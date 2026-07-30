"""
data_ingestion.py
------------------
Simple batch / micro-batch ingestion script.

What it does (per the assignment spec):
  1. Reads new data file(s) dropped into data/incoming/ (e.g. a daily CSV export
     from an upstream system).
  2. Validates them against the expected raw schema.
  3. Appends/merges them into the single "training data" table
     (data/processed/training_data.csv), seeded from data/raw/churn_initial.csv
     the first time it runs.
  4. Logs what it ingested (file name, row count, ingestion timestamp) to
     data/processed/ingestion_log.csv, and moves processed files into
     data/incoming/_processed/ so re-running the script is idempotent
     (already-ingested files are not re-appended).

Run:
    python -m src.data_ingestion
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.features import RAW_FEATURE_COLUMNS

CONFIG_PATH = Path("configs/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _validate_schema(df: pd.DataFrame, target_col: str, id_col: str, source: str) -> None:
    required = set(RAW_FEATURE_COLUMNS + [target_col, id_col])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{source}] missing required columns: {sorted(missing)}")


def run_ingestion(cfg: dict) -> pd.DataFrame:
    processed_path = Path(cfg["paths"]["processed_data"])
    raw_path = Path(cfg["paths"]["raw_data"])
    incoming_dir = Path(cfg["paths"]["incoming_dir"])
    log_path = Path(cfg["paths"]["ingestion_log"])
    target_col = cfg["project"]["target_column"]
    id_col = cfg["project"]["id_column"]

    processed_dir_marker = incoming_dir / "_processed"
    processed_dir_marker.mkdir(exist_ok=True)

    # Seed the training table from the raw dataset on first run.
    if not processed_path.exists():
        seed_df = pd.read_csv(raw_path)
        _validate_schema(seed_df, target_col, id_col, source=str(raw_path))
        seed_df.to_csv(processed_path, index=False)
        _append_log(log_path, source=str(raw_path), n_rows=len(seed_df))
        print(f"Seeded {processed_path} with {len(seed_df)} rows from {raw_path}")

    training_df = pd.read_csv(processed_path)

    incoming_files = sorted(p for p in incoming_dir.glob("*.csv") if p.parent == incoming_dir)
    if not incoming_files:
        print("No new files in data/incoming/. Nothing to ingest.")
        return training_df

    new_batches = []
    for file_path in incoming_files:
        batch_df = pd.read_csv(file_path)
        _validate_schema(batch_df, target_col, id_col, source=str(file_path))
        new_batches.append(batch_df)
        _append_log(log_path, source=str(file_path), n_rows=len(batch_df))
        print(f"Ingested {file_path.name}: {len(batch_df)} rows")
        # Move the file so a re-run doesn't double-count it.
        shutil.move(str(file_path), str(processed_dir_marker / file_path.name))

    if new_batches:
        training_df = pd.concat([training_df] + new_batches, ignore_index=True)
        training_df = training_df.drop_duplicates(subset=[id_col], keep="last")
        training_df.to_csv(processed_path, index=False)
        print(f"training_data.csv now has {len(training_df)} rows total")

    return training_df


def _append_log(log_path: Path, source: str, n_rows: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = pd.DataFrame(
        [{
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "source_file": source,
            "n_rows": n_rows,
        }]
    )
    if log_path.exists():
        entry.to_csv(log_path, mode="a", header=False, index=False)
    else:
        entry.to_csv(log_path, mode="w", header=True, index=False)


def main():
    cfg = load_config()
    run_ingestion(cfg)


if __name__ == "__main__":
    main()
