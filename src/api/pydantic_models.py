from pydantic import BaseModel
from typing import Optional


class CustomerRequest(BaseModel):
    Province: str
    VehicleType: str
    CoverType: str
    Gender: str
    SumInsured: float
    CalculatedPremiumPerTerm: float
    CustomValueEstimate: float
    RegistrationYear: int
    VehicleAge: int


class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int

