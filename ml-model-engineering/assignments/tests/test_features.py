import pandas as pd

from src.features import MODEL_FEATURE_COLUMNS, engineer_features, to_model_matrix


def _sample_row(**overrides):
    row = {
        "customer_id": "CUST1",
        "tenure_months": 2,
        "monthly_charges": 90.0,
        "total_charges": 180.0,
        "contract_type": "month_to_month",
        "payment_method": "electronic_check",
        "internet_service": "fiber_optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "senior_citizen": 0,
        "partner": 0,
        "dependents": 0,
        "phone_service": 1,
        "multiple_lines": 0,
        "paperless_billing": 1,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_engineer_features_adds_expected_columns():
    df = engineer_features(_sample_row())
    for col in [
        "tenure_bucket", "charges_per_tenure_month", "monthly_rate_deviation",
        "num_addon_services", "is_month_to_month", "high_value_customer",
        "internet_service_encoded", "payment_method_encoded",
    ]:
        assert col in df.columns


def test_num_addon_services_counts_yes_values():
    # online_backup=Yes, streaming_tv=Yes -> 2 addons in the sample row
    df = engineer_features(_sample_row())
    assert df.loc[0, "num_addon_services"] == 2


def test_is_month_to_month_encoding():
    df = engineer_features(_sample_row())
    assert df.loc[0, "is_month_to_month"] == 1

    df2 = engineer_features(_sample_row(contract_type="two_year"))
    assert df2.loc[0, "is_month_to_month"] == 0


def test_high_value_customer_threshold():
    df = engineer_features(_sample_row())  # monthly_charges = 90 > 80 threshold
    assert df.loc[0, "high_value_customer"] == 1


def test_internet_service_encoding_ordinal():
    df_none = engineer_features(_sample_row(internet_service="none"))
    df_dsl = engineer_features(_sample_row(internet_service="dsl"))
    df_fiber = engineer_features(_sample_row(internet_service="fiber_optic"))
    assert df_none.loc[0, "internet_service_encoded"] == 0
    assert df_dsl.loc[0, "internet_service_encoded"] == 1
    assert df_fiber.loc[0, "internet_service_encoded"] == 2


def test_to_model_matrix_returns_only_model_columns():
    X = to_model_matrix(_sample_row())
    assert list(X.columns) == MODEL_FEATURE_COLUMNS
    assert len(X) == 1
    assert X.isna().sum().sum() == 0


def test_handles_missing_optional_fields_gracefully():
    partial = pd.DataFrame([{"customer_id": "CUST2", "tenure_months": 10}])
    X = to_model_matrix(partial)
    assert len(X) == 1
    assert X.isna().sum().sum() == 0
