import joblib
import pandas as pd
from fastapi import FastAPI
from app.schemas import HeartInput

app = FastAPI(title="Heart Disease Prediction API")

model = joblib.load("model/heart_model.joblib")

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"model_type": type(model).__name__, "features": FEATURES}

@app.post("/predict")
def predict(data: HeartInput):
    row = pd.DataFrame([data.model_dump()])[FEATURES]
    prediction = model.predict(row)[0]
    return {"heart_disease": bool(prediction)}