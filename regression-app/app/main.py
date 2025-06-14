from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import os
from src.predict import load_model, make_prediction


class OceanProximity(str, Enum):
    NEAR_BAY = "NEAR BAY"
    LESS_THAN_1H_OCEAN = "<1H OCEAN"
    INLAND = "INLAND"
    NEAR_OCEAN = "NEAR OCEAN"
    ISLAND = "ISLAND"
    
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: OceanProximity 
    

app = FastAPI()

# Load model when app starts
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
model_path = os.path.join(project_root, "model.pkl")
model = load_model(model_path)

# Endpoints
@app.get("/")
def read_root():
    return {"messsage": "California Housing Price Predictor"}

# Predict endpoint
@app.post("/predict")
def predict(features: HousingFeatures):
    try:
        prediction = make_prediction(model, features.model_dump())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

