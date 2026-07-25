"""
telemetry_pipeline.py
======================
Global Vehicle Telemetry Platform - PySpark Batch Processing Pipeline

This script addresses Part 3 and Part 4 of the Graded Assignment:

  Part 3.1 - Transformations/Actions: avg engine temp per vehicle model,
             narrow vs wide dependencies identified.
  Part 3.2 - Optimization: salting strategy to fix data skew + partitioning
             strategy discussion (Hash vs Range).
  Part 3.3 - Fault Tolerance: RDD/DataFrame lineage explanation (in comments)
             + a demo of lineage-based recovery (no replication needed).
  Part 3.4 - Checkpointing: simulate an iterative job with growing lineage
             depth and truncate the DAG with .checkpoint().
  Part 4   - Execution mechanics: lazy evaluation, DAG -> Stages via wide
             dependencies, data locality, lineage vs replication trade-offs.

Every section below prints something to stdout so that running this file
produces a log you can submit as evidence that the pipeline executes
end-to-end.
"""

import time
import shutil
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# 0. SPARK SESSION SETUP
# ---------------------------------------------------------------------------
# local[*] uses all available cores on this machine to simulate a distributed
# cluster for demo purposes. In production this would point at a YARN/
# Kubernetes/EMR cluster master instead of "local[*]".
spark = (
    SparkSession.builder
    .appName("GlobalVehicleTelemetryPipeline")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")   # small demo cluster -> fewer shuffle partitions
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

# Checkpoint directory is REQUIRED before calling .checkpoint() - Spark writes
# the truncated lineage state here (ideally reliable storage like HDFS/S3;
# local disk is fine for this demo).
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
spark.sparkContext.setCheckpointDir(CHECKPOINT_DIR)

spark.sparkContext.setLogLevel("WARN")  # quiet down noisy INFO logs for a readable submission log

print("=" * 80)
print("SPARK SESSION STARTED")
print(f"Spark version: {spark.version}")
print(f"Default parallelism (cores): {spark.sparkContext.defaultParallelism}")
print("=" * 80)

# ---------------------------------------------------------------------------
# PART 3.1 - INGEST DATA + AVG ENGINE TEMP PER MODEL
# ---------------------------------------------------------------------------
print("\n[PART 3.1] Ingesting telemetry data and computing avg engine temp per model")
print("-" * 80)

t0 = time.time()

# NARROW DEPENDENCY: reading + schema inference + column selection/filter
# below do not require data to move between partitions - each output
# partition depends on exactly one input partition.
raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/vehicle_telemetry.csv")
)

print(f"Ingested row count: {raw_df.count():,}")   # ACTION - triggers execution
raw_df.printSchema()

# NARROW: filter() and select() - each partition transforms independently,
# no shuffle required.
clean_df = (
    raw_df
    .filter(F.col("engine_temp_c").isNotNull())
    .select("vehicle_id", "vehicle_model", "engine_temp_c", "miles_driven")
)

# WIDE DEPENDENCY: groupBy(...).agg(...) requires all rows with the same
# vehicle_model to land in the same partition -> Spark must SHUFFLE data
# across the network/disk to co-locate matching keys. This is the operation
# that will suffer most from data skew (Part 3.2).
avg_temp_per_model = (
    clean_df
    .groupBy("vehicle_model")
    .agg(
        F.avg("engine_temp_c").alias("avg_engine_temp_c"),
        F.count("*").alias("reading_count"),
        F.sum("miles_driven").alias("total_miles_driven"),
    )
    .orderBy("vehicle_model")
)

print("\nAverage engine temperature per vehicle model (ACTION: show()):")
avg_temp_per_model.show(truncate=False)

print(f"Part 3.1 completed in {time.time() - t0:.2f}s")
print("""
Dependency classification:
  NARROW (no shuffle): read/csv partition parsing, filter(), select()
     -> each output partition is computed from exactly one input partition.
  WIDE (forces shuffle): groupBy('vehicle_model').agg(...), orderBy(...)
     -> requires repartitioning by key (or a global sort) across the cluster.
""")

# ---------------------------------------------------------------------------
# PART 3.2 - DATA SKEW MITIGATION VIA SALTING
# ---------------------------------------------------------------------------
print("\n[PART 3.2] Demonstrating and mitigating data skew with salting")
print("-" * 80)

# --- Step A: Show the skew BEFORE salting --------------------------------
# Row count per vehicle_id partition key reveals the "hot" vehicles that
# generate 1000x more logs (VHOT000-VHOT004 in our sample data).
print("Row counts per vehicle_id (top 10 - reveals the skewed 'hot' vehicles):")
skew_check = (
    clean_df.groupBy("vehicle_id")
    .count()
    .orderBy(F.desc("count"))
)
skew_check.show(10, truncate=False)

# --- Step B: SALTING STRATEGY --------------------------------------------
# Problem: groupBy('vehicle_id') sends ALL rows for 'VHOT000' to a single
# reducer/partition. If VHOT000 has 40,000 rows vs 40 for a normal vehicle,
# that one task becomes a straggler and the whole stage waits on it.
#
# Fix: "salt" the skewed key by appending a random suffix (0-N), so the
# heavy key's rows get spread across N sub-keys/partitions during the FIRST
# aggregation (partial aggregation). Then a SECOND aggregation step strips
# the salt and combines the partial results - final answer is identical,
# but the expensive shuffle is now balanced across many tasks instead of one.
SALT_BUCKETS = 8

salted_df = clean_df.withColumn(
    "salted_vehicle_id",
    F.concat(F.col("vehicle_id"), F.lit("_"), (F.rand() * SALT_BUCKETS).cast("int"))
)

# Stage 1 aggregation: groups by the SALTED key -> spreads a hot vehicle's
# 40,000 rows across up to 8 sub-groups instead of 1, balancing shuffle load.
partial_agg = (
    salted_df.groupBy("salted_vehicle_id", "vehicle_id", "vehicle_model")
    .agg(
        F.sum("engine_temp_c").alias("sum_temp"),
        F.count("*").alias("cnt"),
    )
)

# Stage 2 aggregation: strip the salt (group back by the true vehicle_id)
# and combine the partial sums/counts. This second shuffle is CHEAP because
# it operates on the small, pre-aggregated partial_agg result, not the raw
# 220,000-row dataset.
final_avg_by_vehicle = (
    partial_agg.groupBy("vehicle_id", "vehicle_model")
    .agg(
        (F.sum("sum_temp") / F.sum("cnt")).alias("avg_engine_temp_c"),
        F.sum("cnt").alias("total_readings"),
    )
    .orderBy(F.desc("total_readings"))
)

print("\nSalted two-phase aggregation result (top 10 by reading count):")
final_avg_by_vehicle.show(10, truncate=False)

# --- Step C: PARTITIONING STRATEGY ---------------------------------------
# We explicitly repartition using HASH partitioning on vehicle_model before
# the final aggregation/write. Hash partitioning is appropriate here because
# vehicle_model is a low-cardinality, equality-lookup key (4 models) used
# for groupBy - hash distributes keys evenly across a small number of
# partitions. RANGE partitioning would instead be preferred for
# range-scan/ordered queries (e.g. querying telemetry by a timestamp range),
# since it keeps sorted, contiguous key ranges together on the same
# partition and speeds up range filters.
repartitioned = final_avg_by_vehicle.repartition(4, "vehicle_model")  # HASH partitioning
print(f"\nPartitions after repartition(4, 'vehicle_model') [HASH partitioning]: "
      f"{repartitioned.rdd.getNumPartitions()}")

# ---------------------------------------------------------------------------
# PART 3.3 - FAULT TOLERANCE VIA LINEAGE (RDD / DataFrame level)
# ---------------------------------------------------------------------------
print("\n[PART 3.3] Fault tolerance via RDD lineage (no data replication needed)")
print("-" * 80)

# Every DataFrame/RDD in Spark tracks its LINEAGE: the sequence of
# transformations used to derive it from the original source data.
# If a partition is lost (e.g. an executor node crashes), Spark does NOT
# need a replicated copy of that partition (unlike HDFS's 3x block
# replication) - it simply RECOMPUTES the lost partition by re-running the
# lineage graph starting from the nearest available checkpoint or the
# original source file. This trades some recomputation time for a large
# reduction in storage and network I/O.
print("Logical plan (lineage) for the salted aggregation:")
print(final_avg_by_vehicle.explain(mode="simple"))

# ---------------------------------------------------------------------------
# PART 3.4 - CHECKPOINTING TO TRUNCATE A DEEPLY ITERATIVE LINEAGE
# ---------------------------------------------------------------------------
print("\n[PART 3.4] Simulating iterative lineage growth + checkpointing")
print("-" * 80)

# Scenario: an iterative predictive-maintenance style computation that
# repeatedly transforms the DataFrame (e.g. iterative feature smoothing /
# gradient-style updates over many rounds). Each iteration WITHOUT
# checkpointing appends another layer to the logical plan/lineage graph.
# After enough iterations, the plan becomes so deep that:
#   1. The driver can throw a StackOverflowError while recursively
#      analyzing/optimizing the enormous logical plan.
#   2. If a partition is lost late in the job, recovery requires
#      recomputing potentially HUNDREDS of chained transformations,
#      which is extremely slow and defeats the purpose of lineage-based
#      fault tolerance.
iterative_df = clean_df
NUM_ITERATIONS = 60          # deep enough to illustrate the problem
CHECKPOINT_EVERY = 10        # truncate the DAG every N iterations

print(f"Running {NUM_ITERATIONS} iterative transformations "
      f"(checkpointing every {CHECKPOINT_EVERY} iterations)...")

t0 = time.time()
for i in range(1, NUM_ITERATIONS + 1):
    # A trivial per-iteration transformation standing in for an iterative
    # algorithm's update step (e.g. gradient update, smoothing pass).
    iterative_df = iterative_df.withColumn(
        f"temp_adj_{i}", F.col("engine_temp_c") + F.lit(0.001 * i)
    )

    if i % CHECKPOINT_EVERY == 0:
        # .checkpoint() TRUNCATES the DAG: Spark writes the current,
        # already-computed DataFrame to reliable storage (CHECKPOINT_DIR)
        # and Spark then forgets the long chain of transformations that
        # produced it. Any future recovery starts from this checkpoint
        # instead of replaying 10, 20, 30+ chained steps - bounding both
        # recovery time and the depth of the logical plan the driver must
        # analyze.
        iterative_df = iterative_df.checkpoint(eager=True)
        print(f"  Iteration {i}: checkpoint taken -> lineage truncated here.")

elapsed = time.time() - t0
print(f"Completed {NUM_ITERATIONS} iterations with periodic checkpointing "
      f"in {elapsed:.2f}s")
print(f"Final row count after iterative processing (ACTION): {iterative_df.count():,}")

print("""
CACHING vs CHECKPOINTING (Part 4.3):
  cache()/persist(): stores data in memory/disk for reuse, but KEEPS the
     full lineage. If the cached data is lost, Spark still needs the
     original lineage to recompute it. Cache is a performance optimization,
     not a fault-tolerance / lineage-truncation mechanism.
  checkpoint(): writes data to reliable storage AND cuts the lineage graph
     at that point. Recovery after a checkpoint never needs to look further
     back than the checkpoint itself. This is the tool that actually solves
     the "liability of lineage" (StackOverflow risk, unbounded recovery time).
""")

# ---------------------------------------------------------------------------
# PART 4.1 - LAZY EVALUATION AND DAG -> STAGE DECOMPOSITION
# ---------------------------------------------------------------------------
print("\n[PART 4.1] Lazy evaluation and DAG -> Stage decomposition")
print("-" * 80)

# None of the .filter/.select/.groupBy/.withColumn calls above executed
# anything by themselves - Spark only builds up a logical plan (DAG) of
# transformations. Nothing runs until an ACTION (count(), show(), write())
# is called. This laziness lets Spark's Catalyst optimizer look at the
# WHOLE chain of transformations at once and rewrite/optimize it (e.g.
# predicate pushdown, combining filters) before ever touching the data.
print("Physical plan showing Stage boundaries introduced by wide dependencies "
      "(Exchange = shuffle boundary = new Stage):")
final_avg_by_vehicle.explain(mode="formatted")

print("""
How stages are formed:
  Spark walks the logical DAG backwards from the final action. Every time
  it crosses a WIDE dependency (a shuffle-requiring operation, shown as
  'Exchange' in the physical plan above), it closes the current Stage and
  opens a new one. Tasks WITHIN a stage can run back-to-back with no
  network I/O (pure narrow/pipelined transformations); a NEW stage cannot
  start until all tasks of the previous stage have produced their shuffle
  output, since it must read data that other executors produced.
""")

# ---------------------------------------------------------------------------
# PART 4.2 - DATA LOCALITY: "DON'T MOVE DATA, MOVE CODE"
# ---------------------------------------------------------------------------
print("\n[PART 4.2] Data locality demo")
print("-" * 80)

print(f"Input file partitions created by Spark's CSV reader: "
      f"{raw_df.rdd.getNumPartitions()}")
print("""
Spark's scheduler tries to schedule each task on (or as close as possible to)
the executor/node that ALREADY holds the data block it needs (PROCESS_LOCAL >
NODE_LOCAL > RACK_LOCAL > ANY). Since telemetry files can be gigabytes across
500,000 vehicles, shipping the ~KBs of compiled task code to the data is far
cheaper than shipping GBs of raw telemetry across the network to the code -
this is the "don't move data, move code" principle, and it's the main reason
Spark stages that DON'T require a shuffle (narrow dependencies) run so much
faster than the wide-dependency stages that must move data between nodes.
""")

# ---------------------------------------------------------------------------
# WRITE FINAL OUTPUT (demonstrates a full, real, end-to-end batch job)
# ---------------------------------------------------------------------------
OUTPUT_PATH = "output/avg_temp_per_model"
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

print(f"\nWriting final aggregation result to {OUTPUT_PATH} ...")
avg_temp_per_model.write.mode("overwrite").parquet(OUTPUT_PATH)
print("Write complete.")

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)

spark.stop()
