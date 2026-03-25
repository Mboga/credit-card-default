import os
from dotenv import load_dotenv
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts.processing import extract_features

# Load Configuration
load_dotenv() #load environment variables from .env

#get the variables

MLRUNS_DIR = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME","CreditDefault_XGB")
MODEL_VERSION = int(os.getenv("MODEL_VERSION",3))

# MLFLOW SetUP
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")


# APP

app = FastAPI(title="Credit Default Prediction API")

# REQUEST SCHEMA

# Pydantic validates every incoming request against this shape.
# If a field is missing or the wrong type, FastAPI returns a 422 automatically.
class CreditRequest(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int;  PAY_2: int;  PAY_3: int
    PAY_4: int;  PAY_5: int;  PAY_6: int
    BILL_AMT1: float; BILL_AMT2: float; BILL_AMT3: float
    BILL_AMT4: float; BILL_AMT5: float; BILL_AMT6: float
    PAY_AMT1: float;  PAY_AMT2: float;  PAY_AMT3: float
    PAY_AMT4: float;  PAY_AMT5: float;  PAY_AMT6: float

# MODEL CACHE

# Load the model once at startup instead of on every request (much faster).
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
        print(f" {MODEL_NAME} v{MODEL_VERSION} loaded.")
    except Exception as e:
        print(f" Model failed to load: {e}")

# ── 5. ROUTES ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check — confirms the API and which model version is active."""
    return {"status": "online", "model": MODEL_NAME, "version": MODEL_VERSION}


@app.post("/predict")
def predict(request: CreditRequest):
    """
    Accepts 23 credit features, returns a default probability and decision.
    Steps: validate → feature engineer → predict → respond.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        # a) Pydantic object → dict → single-row DataFrame
        df_raw = pd.DataFrame([request.model_dump()])

        # b) Run the same feature engineering used during training
        df_featured = extract_features(df_raw)

        # c) Cast to float so XGBoost doesn't complain about int columns
        X = df_featured.astype(float)

        # d) Predict: proba[:,1] is the probability of class 1 (default)
        prob       = float(model.predict_proba(X)[0, 1])
        prediction = 1 if prob > 0.5 else 0

        return {
            "default_probability": round(prob, 4),
            "prediction":          prediction,          # 1 = default, 0 = no default
            "decision":            "REJECTED" if prediction else "APPROVED",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))