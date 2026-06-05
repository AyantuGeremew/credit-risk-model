import pandas as pd
import mlflow
import mlflow.pyfunc

from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import (
    CustomerRequest,
    PredictionResponse
)

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

app = FastAPI(
    title="Insurance Risk Prediction API",
    description="Predict customer claim risk probability",
    version="1.0.0"
)

# ---------------------------------------------------
# Load Model from MLflow Registry
# ---------------------------------------------------

MODEL_NAME = "insurance_risk_model"
MODEL_STAGE = "Production"

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)

    print(f"Loaded model: {MODEL_NAME} ({MODEL_STAGE})")

except Exception as e:
    print(f"Model loading failed: {e}")
    model = None


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------

@app.get("/")
def home():
    return {"message": "Insurance Risk Prediction API is running"}


# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(customer: CustomerRequest):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    try:

        input_df = pd.DataFrame(
            [customer.model_dump()]
        )

        # Predict probability
        probability = model.predict(input_df)

        risk_probability = float(probability[0])

        prediction = (
            1 if risk_probability >= 0.5
            else 0
        )

        return PredictionResponse(
            risk_probability=risk_probability,
            prediction=prediction
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )