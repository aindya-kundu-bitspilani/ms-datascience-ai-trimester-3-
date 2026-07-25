# Setup & Execution Guide — Global Vehicle Telemetry PySpark Pipeline

This guide gets you from a clean machine to a submittable execution log.
Two paths are given: **Option A (local machine)**, which is what was used
to produce the included `logs/execution_log_clean.txt`, and **Option B
(Google Colab)**, useful if you don't want to install anything locally.

---

## Option A — Run locally (Windows / Mac / Linux)

### 1. Prerequisites
- **Java 11 or 17** (Spark requires a JDK; Java 21 also works with recent PySpark).
  Check with:
  ```bash
  java -version
  ```
  If missing, install via:
  - Ubuntu/Debian: `sudo apt install openjdk-17-jdk`
  - Mac (Homebrew): `brew install openjdk@17`
  - Windows: install Temurin 17 from https://adoptium.net and add it to PATH.

- **Python 3.9+**
  ```bash
  python3 --version
  ```

### 2. Create a project folder and virtual environment
```bash
mkdir telemetry_project && cd telemetry_project
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install PySpark
```bash
pip install pyspark
```
This installs PySpark 4.x (or the latest stable) with `py4j` — no separate
Spark download or `SPARK_HOME` setup needed for local mode.

### 4. Add the project files
Place these two files (provided alongside this guide) into your project
folder:
- `generate_sample_data.py`
- `telemetry_pipeline.py`

### 5. Generate the sample dataset
```bash
python3 generate_sample_data.py
```
Expected output:
```
Generated 220,000 telemetry rows for 505 vehicles.
  Normal vehicles: 500 x 40 rows
  Hot (skewed) vehicles: 5 x 40000 rows
Output written to: data/vehicle_telemetry.csv
```
This creates `data/vehicle_telemetry.csv` — the file the pipeline reads.

### 6. Run the PySpark pipeline
```bash
python3 telemetry_pipeline.py
```
This single run demonstrates, in order:
1. Data ingestion + average engine temperature per model (narrow vs wide ops)
2. Detection of skewed vehicle IDs + a two-phase **salting** fix
3. Hash partitioning strategy applied via `repartition()`
4. Lineage/fault-tolerance explanation via `.explain()`
5. An iterative loop with **checkpointing** every 10 iterations (DAG truncation)
6. Lazy evaluation + DAG → Stage decomposition via `.explain(mode="formatted")`
7. A data-locality discussion
8. A final Parquet write to `output/avg_temp_per_model/`

### 7. Capture the log as evidence
To save a clean, submittable log file:
```bash
python3 telemetry_pipeline.py > execution_log.txt 2>&1
```
Then open `execution_log.txt` — it will contain the full console output,
including the printed DataFrames, physical plans, and checkpoint messages,
proving the code executed successfully end-to-end.

### 8. Verify the output was actually written
```bash
ls output/avg_temp_per_model/
```
You should see a `_SUCCESS` file and one or more `.parquet` part-files —
`_SUCCESS` is Spark's own confirmation that the write completed without
errors, which is strong evidence for your submission.

---

## Option B — Google Colab (no local install required)

1. Go to https://colab.research.google.com and create a new notebook.
2. In the first cell, install PySpark:
   ```python
   !pip install pyspark
   ```
3. Upload `generate_sample_data.py` and `telemetry_pipeline.py` using the
   file browser on the left (folder icon → upload), or paste their contents
   into cells.
4. Run:
   ```python
   !python generate_sample_data.py
   !python telemetry_pipeline.py
   ```
5. Colab will print the entire log inline under the cell — screenshot or
   copy this output as your evidence. You can also redirect to a file and
   download it:
   ```python
   !python telemetry_pipeline.py > execution_log.txt 2>&1
   from google.colab import files
   files.download('execution_log.txt')
   ```

---

## What to actually submit

For each rubric criterion, point to this evidence:

| Rubric Criterion | Evidence in this submission |
|---|---|
| Architectural Design & Scaling Strategy | Your written answers to Part 1 (not code-based) |
| Code Effectiveness & Distributed Optimization | `telemetry_pipeline.py` Part 3.1–3.2 sections + the printed skew table and salted aggregation output in the log |
| Fault Tolerance & Resilience Implementation | Part 3.3–3.4 sections + the `.explain()` output and checkpoint messages in the log |
| Advanced Execution Mechanics & Resilience | Part 4.1–4.2 sections + the formatted physical plan showing `Exchange` (shuffle) boundaries |
| Documentation & Explanation | The inline comments throughout `telemetry_pipeline.py`, written to map directly onto each assignment sub-question |

Submit all of:
1. `generate_sample_data.py`
2. `telemetry_pipeline.py`
3. `data/vehicle_telemetry.csv` (or regenerate it — it's deterministic, seeded with `random.seed(42)`)
4. `execution_log.txt` (your captured run log)
5. Your written narrative answers for Part 1 and Part 2 (pure conceptual questions, no code required)

---

## Troubleshooting

- **`JAVA_HOME is not set` error**: set it explicitly, e.g. on Linux/Mac:
  ```bash
  export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
  ```
- **`Py4JJavaError` on checkpoint**: make sure the `checkpoints/` directory
  is writable — the script creates it automatically, but confirm you have
  write permissions in the project folder.
- **Very slow on a laptop**: reduce `NUM_HOT_VEHICLES` /
  `HOT_READINGS_PER_VEHICLE` in `generate_sample_data.py` to shrink the
  dataset, or lower `NUM_ITERATIONS` in `telemetry_pipeline.py`.
- **Progress bar clutter (`[Stage 3:>...]`) in your log**: this is normal
  Spark console output showing task progress; it's safe to leave in the
  log as further proof of real execution, or strip it for readability.
