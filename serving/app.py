from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using a trained Random Forest model.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and features
MODEL_PATH = "models/rf_model.joblib"
FEATURES_PATH = "models/features.joblib"

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
else:
    model = None
    features = None
    print("Warning: Model or features not found. Please run train.py first.")

# Define input schema corresponding to the selected features
class HouseFeatures(BaseModel):
    LotArea: float
    OverallQual: int
    YearBuilt: int
    TotalBsmtSF: float
    GrLivArea: float
    FullBath: int
    BedroomAbvGr: int
    GarageCars: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "LotArea": 8450,
                "OverallQual": 7,
                "YearBuilt": 2003,
                "TotalBsmtSF": 856,
                "GrLivArea": 1710,
                "FullBath": 2,
                "BedroomAbvGr": 3,
                "GarageCars": 2
            }
        }
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API. Use /predict to get predictions."}

@app.post("/predict")
def predict_price(house: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert input to DataFrame
    input_data = pd.DataFrame([house.model_dump()])
    
    # Ensure correct column order
    input_data = input_data[features]
    
    # Predict and convert to INR (1 USD = 83 INR)
    prediction_usd = model.predict(input_data)[0]
    prediction_inr = prediction_usd * 83
    
    return {
        "predicted_price": round(float(prediction_inr), 2)
    }

if __name__ == "__main__":
    import uvicorn
    # This allows running the app easily for testing
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
