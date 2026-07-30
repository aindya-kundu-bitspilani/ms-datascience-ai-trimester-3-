"""
load_dataset.py
----------------
Loads and cleans the real, open-source "Telco Customer Churn" dataset
(IBM sample dataset, mirrored on Kaggle as
https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The raw, unmodified file is checked into this repo at
data/external/Telco-Customer-Churn-original.csv (7,043 rows, 21 columns)
so the project is reproducible without depending on live internet access
or Kaggle credentials at run time. Use --refresh to re-download the
original file from its public GitHub mirror.

What this script does:
  1. Loads the raw CSV.
  2. Cleans it:
     - `TotalCharges` is stored as a string in the source file and is
       blank for 11 customers, all of whom have tenure == 0 (brand new
       customers who haven't been billed yet). We coerce it to numeric
       and fill those 11 blanks with `MonthlyCharges` (their first
       partial bill), rather than 0, which would be misleading.
     - Column names are renamed to snake_case and category values are
       normalized to the tokens the rest of this project expects
       (e.g. "Month-to-month" -> "month_to_month").
  3. Splits the cleaned data into:
     - data/raw/churn_initial.csv      (first 90% of rows -- the
       "historical" dataset the initial model is trained on)
     - data/incoming/churn_new_batch.csv (last 10% of rows -- held out
       to simulate a new daily/weekly batch arriving later, so the
       ingestion pipeline in src/data_ingestion.py has something real
       to demonstrate on). The source dataset has no timestamp column,
       so a positional split is used as a stand-in for a time-based
       split; this is documented as a simplifying assumption.

Run:
    python -m src.load_dataset
    python -m src.load_dataset --refresh   # re-download the source file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXTERNAL_PATH = Path("data/external/Telco-Customer-Churn-original.csv")

CONTRACT_MAP = {
    "Month-to-month": "month_to_month",
    "One year": "one_year",
    "Two year": "two_year",
}
PAYMENT_MAP = {
    "Electronic check": "electronic_check",
    "Mailed check": "mailed_check",
    "Bank transfer (automatic)": "bank_transfer",
    "Credit card (automatic)": "credit_card",
}
INTERNET_MAP = {
    "DSL": "dsl",
    "Fiber optic": "fiber_optic",
    "No": "none",
}
YES_NO_COLS = [
    "partner", "dependents", "phone_service", "paperless_billing",
]
ADDON_COLS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]


def maybe_refresh(refresh: bool) -> None:
    if not refresh and EXTERNAL_PATH.exists():
        return
    import urllib.request
    EXTERNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading source dataset from {SOURCE_URL} ...")
    urllib.request.urlretrieve(SOURCE_URL, EXTERNAL_PATH)
    print(f"Saved to {EXTERNAL_PATH}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "customerID": "customer_id",
        "tenure": "tenure_months",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Contract": "contract_type",
        "PaymentMethod": "payment_method",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "PaperlessBilling": "paperless_billing",
        "Churn": "churn",
    })

    # --- Data cleaning: TotalCharges blanks (documented in module docstring) ---
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    blank_mask = df["total_charges"].isna()
    df.loc[blank_mask, "total_charges"] = df.loc[blank_mask, "monthly_charges"]

    # --- Normalize categorical value strings to snake_case tokens ---
    df["contract_type"] = df["contract_type"].map(CONTRACT_MAP)
    df["payment_method"] = df["payment_method"].map(PAYMENT_MAP)
    df["internet_service"] = df["internet_service"].map(INTERNET_MAP)

    for col in YES_NO_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # multiple_lines has 3 states ("Yes"/"No"/"No phone service") -> binary flag
    df["multiple_lines"] = (df["multiple_lines"] == "Yes").astype(int)

    # addon columns keep their 3-state string form (Yes/No/No internet service);
    # src/features.py counts "Yes" occurrences directly.
    for col in ADDON_COLS:
        df[col] = df[col]

    df["churn"] = (df["churn"] == "Yes").astype(int)

    keep_cols = [
        "customer_id", "tenure_months", "monthly_charges", "total_charges",
        "contract_type", "payment_method", "internet_service",
        *ADDON_COLS,
        "senior_citizen", "partner", "dependents", "phone_service",
        "multiple_lines", "paperless_billing", "churn",
    ]
    return df[keep_cols]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download the source CSV")
    args = parser.parse_args(argv)

    maybe_refresh(args.refresh)

    raw = pd.read_csv(EXTERNAL_PATH)
    cleaned = clean(raw)

    n_total = len(cleaned)
    n_initial = int(n_total * 0.9)

    initial = cleaned.iloc[:n_initial].reset_index(drop=True)
    incoming = cleaned.iloc[n_initial:].reset_index(drop=True)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/incoming").mkdir(parents=True, exist_ok=True)

    initial.to_csv("data/raw/churn_initial.csv", index=False)
    incoming.to_csv("data/incoming/churn_new_batch.csv", index=False)

    print(f"Source: {EXTERNAL_PATH} ({n_total} rows, real Telco Customer Churn dataset)")
    print(f"Wrote data/raw/churn_initial.csv: {len(initial)} rows")
    print(f"Wrote data/incoming/churn_new_batch.csv: {len(incoming)} rows")
    print(f"Churn rate (initial): {initial['churn'].mean():.2%}")


if __name__ == "__main__":
    main()
