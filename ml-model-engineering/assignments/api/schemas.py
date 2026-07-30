from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    customer_id: str = Field(..., json_schema_extra={"example": "9237-HQITU"})
    tenure_months: int = Field(..., ge=0, le=100, json_schema_extra={"example": 2})
    monthly_charges: float = Field(..., ge=0, json_schema_extra={"example": 70.7})
    total_charges: float = Field(..., ge=0, json_schema_extra={"example": 151.65})
    contract_type: Literal["month_to_month", "one_year", "two_year"] = Field(
        ..., json_schema_extra={"example": "month_to_month"}
    )
    payment_method: Literal["electronic_check", "mailed_check", "bank_transfer", "credit_card"] = Field(
        ..., json_schema_extra={"example": "electronic_check"}
    )
    internet_service: Literal["none", "dsl", "fiber_optic"] = Field(
        ..., json_schema_extra={"example": "fiber_optic"}
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ..., json_schema_extra={"example": "No"}
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ..., json_schema_extra={"example": "No"}
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ..., json_schema_extra={"example": "No"}
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ..., json_schema_extra={"example": "No"}
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ..., json_schema_extra={"example": "No"}
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ..., json_schema_extra={"example": "No"}
    )
    senior_citizen: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    partner: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    dependents: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    phone_service: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    multiple_lines: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    paperless_billing: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    customer_id: str
    churn_prediction: int
    churn_probability: float
    model_version: str
    model_type: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str | None = None
