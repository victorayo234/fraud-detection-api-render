import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Fraud Detection API")

# Load the model once at startup (fraud_model.pkl must sit next to this file)
try:
    model = joblib.load("fraud_model.pkl")
except Exception as e:
    model = None
    load_error = str(e)


class Transaction(BaseModel):
    # Raw PaySim-style fields — adjust names here if your training columns differ
    type: str = Field(..., description="Transaction type, e.g. TRANSFER, CASH_OUT, PAYMENT")
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


TYPE_MAP = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}


@app.get("/")
def health():
    if model is None:
        return {"status": "error", "detail": load_error}
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {load_error}")

    if tx.type not in TYPE_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown type '{tx.type}'. Expected one of {list(TYPE_MAP)}")

    # Recreate the engineered features from training
    error_balance_orig = tx.oldbalanceOrg - tx.amount - tx.newbalanceOrig
    error_balance_dest = tx.newbalanceDest - (tx.oldbalanceDest + tx.amount)

    row = pd.DataFrame([{
        "type": TYPE_MAP[tx.type],
        "amount": tx.amount,
        "oldbalanceOrg": tx.oldbalanceOrg,
        "newbalanceOrig": tx.newbalanceOrig,
        "oldbalanceDest": tx.oldbalanceDest,
        "newbalanceDest": tx.newbalanceDest,
        "errorBalanceOrig": error_balance_orig,
        "errorBalanceDest": error_balance_dest,
    }])

    try:
        pred = int(model.predict(row)[0])
        proba = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed — check that column names/order match training: {e}")

    return {
        "is_fraud": bool(pred),
        "fraud_probability": proba,
    }
