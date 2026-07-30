"""
features.py
------------
Single source of truth for feature engineering, built around the real
Telco Customer Churn dataset schema (see src/load_dataset.py).

This module is imported by BOTH:
  - src/train.py          (offline / training path)
  - api/main.py             (online / serving path)

That is the concrete mechanism this project uses to avoid training-serving
skew: there is exactly one implementation of each transformation, so the
model always sees features computed the same way, whether it's being
trained on the historical CSV or scoring a single live request.

Offline vs online:
  - RAW_FEATURE_COLUMNS below are all attributes of a customer's account
    / subscription (contract, billing, add-on services, tenure). In a
    real deployment these would be read from the customer/billing
    system (e.g. a CRM or subscription-management service) at request
    time; here the API simply accepts them directly in the request body
    for simplicity.
  - All DERIVED features are cheap, stateless, row-local transforms of
    those raw inputs (ratios, bucketing, counting, encoding) -- none of
    them require a lookback window over other customers or a feature
    store, so they can be computed synchronously and identically in
    both the training pipeline and the online request path using this
    module.

Engineered features (7 non-trivial features, on top of raw inputs):
  1. tenure_bucket             - binned tenure (ordinal encoding)
  2. charges_per_tenure_month  - ratio: total_charges / (tenure_months + 1)
  3. monthly_rate_deviation    - derived: monthly_charges - charges_per_tenure_month
                                  (is the customer's current bill higher than
                                  their historical average -> proxy for a
                                  recent price change / plan upgrade)
  4. num_addon_services        - aggregation: count of "Yes" among the 6
                                  add-on service columns (0-6)
  5. is_month_to_month         - binary encoding of contract_type
  6. high_value_customer       - threshold flag on monthly_charges
  7. internet_service_encoded  - ordinal encoding (none/dsl/fiber_optic)
  8. payment_method_encoded    - ordinal encoding of payment_method
"""

from __future__ import annotations

import pandas as pd

# Raw columns the model / API expects as input (see src/load_dataset.py
# for how these map onto the original Kaggle/IBM Telco Churn columns).
RAW_FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "payment_method",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "paperless_billing",
]

ADDON_COLUMNS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]

# Final numeric feature columns fed into the model after engineering
MODEL_FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "paperless_billing",
    "internet_service_encoded",
    "tenure_bucket",
    "charges_per_tenure_month",
    "monthly_rate_deviation",
    "num_addon_services",
    "is_month_to_month",
    "high_value_customer",
    "payment_method_encoded",
]

HIGH_VALUE_THRESHOLD = 80.0  # monthly_charges above this => "high value" flag

_TENURE_BUCKET_EDGES = [-1, 6, 12, 24, 48, 10_000]
_TENURE_BUCKET_LABELS = [0, 1, 2, 3, 4]  # 0-6mo, 6-12mo, 1-2y, 2-4y, 4y+

_PAYMENT_METHOD_MAP = {
    "electronic_check": 0,
    "mailed_check": 1,
    "bank_transfer": 2,
    "credit_card": 3,
}
_INTERNET_SERVICE_MAP = {
    "none": 0,
    "dsl": 1,
    "fiber_optic": 2,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a raw dataframe.

    Accepts a dataframe containing at least RAW_FEATURE_COLUMNS and
    returns a NEW dataframe with MODEL_FEATURE_COLUMNS added/ready.
    Safe to call with a single-row dataframe (online request) or a
    full historical batch (training).
    """
    out = df.copy()

    # Defensive defaults in case of missing raw fields (keeps both
    # training and serving from crashing on partial/malformed input).
    for col in RAW_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0

    out["tenure_months"] = out["tenure_months"].fillna(0).astype(float)
    out["monthly_charges"] = out["monthly_charges"].fillna(0).astype(float)
    out["total_charges"] = out["total_charges"].fillna(0).astype(float)
    out["senior_citizen"] = out["senior_citizen"].fillna(0).astype(int)
    out["partner"] = out["partner"].fillna(0).astype(int)
    out["dependents"] = out["dependents"].fillna(0).astype(int)
    out["phone_service"] = out["phone_service"].fillna(0).astype(int)
    out["multiple_lines"] = out["multiple_lines"].fillna(0).astype(int)
    out["paperless_billing"] = out["paperless_billing"].fillna(0).astype(int)

    # 1. tenure_bucket
    out["tenure_bucket"] = pd.cut(
        out["tenure_months"],
        bins=_TENURE_BUCKET_EDGES,
        labels=_TENURE_BUCKET_LABELS,
    ).astype(int)

    # 2. charges_per_tenure_month (ratio feature)
    out["charges_per_tenure_month"] = out["total_charges"] / (out["tenure_months"] + 1)

    # 3. monthly_rate_deviation (derived: current bill vs historical average)
    out["monthly_rate_deviation"] = out["monthly_charges"] - out["charges_per_tenure_month"]

    # 4. num_addon_services (aggregation over the 6 add-on service columns)
    def _count_yes(row) -> int:
        return sum(1 for col in ADDON_COLUMNS if row.get(col) == "Yes")

    out["num_addon_services"] = out.apply(_count_yes, axis=1)

    # 5. is_month_to_month (categorical encoding)
    out["is_month_to_month"] = (out["contract_type"] == "month_to_month").astype(int)

    # 6. high_value_customer (threshold flag)
    out["high_value_customer"] = (out["monthly_charges"] > HIGH_VALUE_THRESHOLD).astype(int)

    # 7. internet_service_encoded (ordinal encoding)
    out["internet_service_encoded"] = (
        out["internet_service"].map(_INTERNET_SERVICE_MAP).fillna(0).astype(int)
    )

    # 8. payment_method_encoded (ordinal encoding)
    out["payment_method_encoded"] = (
        out["payment_method"].map(_PAYMENT_METHOD_MAP).fillna(-1).astype(int)
    )

    return out


def to_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and return only the columns the model was trained on."""
    engineered = engineer_features(df)
    return engineered[MODEL_FEATURE_COLUMNS]
