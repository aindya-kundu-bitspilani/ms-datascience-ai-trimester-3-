"""
generate_sample_data.py
------------------------
Generates a synthetic fleet-telemetry dataset that mimics the assignment's
scenario: 500,000 vehicles streaming engine heat, speed, location, and
battery efficiency data.

For a laptop-friendly demo we generate a much smaller sample (configurable),
but we DELIBERATELY inject data skew: a handful of "hot" vehicle_ids get
~1000x more rows than normal vehicles. This lets the PySpark script
demonstrate salting / skew mitigation on real, visible skew rather than
a hypothetical one.

Output: data/vehicle_telemetry.csv
Columns:
    vehicle_id        - unique vehicle identifier (string, e.g. "V000123")
    vehicle_model      - one of a small set of truck models
    timestamp          - ISO8601 timestamp of the reading
    engine_temp_c      - engine temperature in Celsius
    speed_kmph         - speed in km/h
    latitude / longitude - GPS coordinates (randomly walked)
    battery_efficiency - 0-1 efficiency score
    miles_driven        - incremental miles logged in this ping
"""

import csv
import random
import datetime
import os

random.seed(42)

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vehicle_telemetry.csv")

# ---- Configuration (scaled down from 500,000 vehicles for a runnable demo) ----
NUM_NORMAL_VEHICLES = 500          # normal fleet vehicles
NUM_HOT_VEHICLES = 5               # small number of vehicles causing skew
NORMAL_READINGS_PER_VEHICLE = 40   # ~40 pings for a normal vehicle
HOT_READINGS_PER_VEHICLE = 40_000  # 1000x more pings -> simulates severe skew

VEHICLE_MODELS = ["FreightMax-9000", "UrbanHauler-500", "LongRoute-750", "EcoFleet-200"]

BASE_LAT, BASE_LON = 12.9716, 77.5946  # arbitrary base coordinate (Bengaluru)
START_TIME = datetime.datetime(2026, 1, 1, 0, 0, 0)


def make_vehicle_pool():
    """Assign each vehicle_id a model and a reading count."""
    vehicles = []

    # Normal vehicles
    for i in range(NUM_NORMAL_VEHICLES):
        vid = f"V{i:06d}"
        model = random.choice(VEHICLE_MODELS)
        vehicles.append((vid, model, NORMAL_READINGS_PER_VEHICLE))

    # "Hot" skewed vehicles - these will dominate the partition if not salted
    for i in range(NUM_HOT_VEHICLES):
        vid = f"VHOT{i:03d}"
        model = random.choice(VEHICLE_MODELS)
        vehicles.append((vid, model, HOT_READINGS_PER_VEHICLE))

    return vehicles


def generate_rows(vehicles):
    for vid, model, n_readings in vehicles:
        lat, lon = BASE_LAT + random.uniform(-1, 1), BASE_LON + random.uniform(-1, 1)
        ts = START_TIME
        cumulative_miles = 0.0
        # baseline engine temp differs slightly per model (for realism)
        base_temp = {"FreightMax-9000": 92, "UrbanHauler-500": 85,
                     "LongRoute-750": 95, "EcoFleet-200": 80}[model]

        for _ in range(n_readings):
            ts = ts + datetime.timedelta(minutes=random.randint(1, 5))
            lat += random.uniform(-0.01, 0.01)
            lon += random.uniform(-0.01, 0.01)
            engine_temp = round(base_temp + random.uniform(-5, 8), 2)
            speed = round(random.uniform(0, 110), 1)
            battery_eff = round(random.uniform(0.55, 0.99), 3)
            miles_increment = round(speed * random.uniform(0.01, 0.08), 3)
            cumulative_miles += miles_increment

            yield [
                vid,
                model,
                ts.isoformat(),
                engine_temp,
                speed,
                round(lat, 6),
                round(lon, 6),
                battery_eff,
                miles_increment,
            ]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vehicles = make_vehicle_pool()

    header = [
        "vehicle_id", "vehicle_model", "timestamp", "engine_temp_c",
        "speed_kmph", "latitude", "longitude", "battery_efficiency",
        "miles_driven",
    ]

    row_count = 0
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in generate_rows(vehicles):
            writer.writerow(row)
            row_count += 1

    total_vehicles = len(vehicles)
    print(f"Generated {row_count:,} telemetry rows for {total_vehicles} vehicles.")
    print(f"  Normal vehicles: {NUM_NORMAL_VEHICLES} x {NORMAL_READINGS_PER_VEHICLE} rows")
    print(f"  Hot (skewed) vehicles: {NUM_HOT_VEHICLES} x {HOT_READINGS_PER_VEHICLE} rows")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
