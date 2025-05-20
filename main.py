from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="Avian Outbreak Prediction API", 
              description="API to predict whether an avian outbreak will occur in the next 7 days (Yes/No).")

# Load the model from the pickle file
with open('RANDOMFOREST_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define categorical and numerical features based on the dataset
categorical_features = ['Season', 'Holiday', 'State', 'Proximity_to_Water', 
                       'Wild_Bird_Migration', 'Neighboring_Farm_Outbreak', 
                       'Farm_Type', 'Feed_Source', 'Water_Source', 
                       'Other_Animals_Present', 'Protective_Equipment_Used', 
                       'Farm_Accessibility', 'Previous_Outbreak', 'Farm_Size']
numerical_features = ['Latitude', 'Longitude', 'Humidity (%)', 'Vaccination_Rate (%)', 
                     'Temperature (°C)', 'Recent_Farm_Visits', 'Mortality_Rate_Last_Week (%)']

# Define input schema using Pydantic
class PredictionInput(BaseModel):
    Date: str  # Format: YYYY-MM-DD
    Season: str  # Choices: rainy, dry
    Holiday: str  # Choices: Yes, No
    State: str  # Choices: Lagos, Borno, Enugu, Rivers, Kaduna, Ogun, Benue, Niger, Oyo, Kano
    Latitude: float  # Range: 4 to 14
    Longitude: float  # Range: 2 to 15
    Proximity_to_Water: str  # Choices: ≤5km, >5km
    Humidity: float  # Range: 0 to 100
    Wild_Bird_Migration: str  # Choices: low, medium, high
    Neighboring_Farm_Outbreak: str  # Choices: Yes, No
    Farm_Type: str  # Choices: commercial, broiler, backyard, layer
    Vaccination_Rate: float  # Range: 0 to 100
    Farm_Size: str  # Choices: small, medium, large
    Feed_Source: str  # Choices: homegrown, local market, commercial supplier
    Water_Source: str  # Choices: well, municipal supply, river
    Temperature: float  # Range: 0 to 50
    Recent_Farm_Visits: int  # Range: 0 to 50
    Other_Animals_Present: str  # Choices: Yes, No
    Mortality_Rate_Last_Week: float  # Range: 0 to 100
    Protective_Equipment_Used: str  # Choices: Yes, No
    Farm_Accessibility: str  # Choices: easy, moderate, difficult
    Previous_Outbreak: str  # Choices: Yes, No

# Function to preprocess inputs
def preprocess_inputs(data: PredictionInput):
    # Convert inputs to a dictionary
    input_data = {
        'Date': data.Date,
        'Season': data.Season,
        'Holiday': data.Holiday,
        'State': data.State,
        'Latitude': data.Latitude,
        'Longitude': data.Longitude,
        'Proximity_to_Water': data.Proximity_to_Water,
        'Humidity (%)': data.Humidity,
        'Wild_Bird_Migration': data.Wild_Bird_Migration,
        'Neighboring_Farm_Outbreak': data.Neighboring_Farm_Outbreak,
        'Farm_Type': data.Farm_Type,
        'Vaccination_Rate (%)': data.Vaccination_Rate,
        'Farm_Size': data.Farm_Size,
        'Feed_Source': data.Feed_Source,
        'Water_Source': data.Water_Source,
        'Temperature (°C)': data.Temperature,
        'Recent_Farm_Visits': data.Recent_Farm_Visits,
        'Other_Animals_Present': data.Other_Animals_Present,
        'Mortality_Rate_Last_Week (%)': data.Mortality_Rate_Last_Week,
        'Protective_Equipment_Used': data.Protective_Equipment_Used,
        'Farm_Accessibility': data.Farm_Accessibility,
        'Previous_Outbreak': data.Previous_Outbreak
    }
    
    # Create a DataFrame
    df = pd.DataFrame([input_data])
    
    # Process Date feature: Extract year, month, day
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop('Date', axis=1)
    
    # Encode categorical variables (one-hot encoding)
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Ensure all expected columns are present (fill missing with 0)
    expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match model training
    df_encoded = df_encoded[expected_columns]
    
    return df_encoded

# API endpoint for prediction
@app.post("/predict", response_model=dict)
async def predict_outbreak(data: PredictionInput):
    try:
        # Preprocess inputs
        input_df = preprocess_inputs(data)
        
        # Make prediction and map to "Yes" or "No"
        prediction = model.predict(input_df)[0]
        prediction = "Yes" if prediction == 1 else "No"
        return {"Outbreak in next 7 days": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Root endpoint for basic info
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Avian Outbreak Prediction API",
        "endpoint": "/predict",
        "method": "POST",
        "input_format": "JSON with fields: Date (str, YYYY-MM-DD), Season (str), Holiday (str), State (str), Latitude (float), Longitude (float), Proximity_to_Water (str), Humidity (float), Wild_Bird_Migration (str), Neighboring_Farm_Outbreak (str), Farm_Type (str), Vaccination_Rate (float), Farm_Size (str), Feed_Source (str), Water_Source (str), Temperature (float), Recent_Farm_Visits (int), Other_Animals_Present (str), Mortality_Rate_Last_Week (float), Protective_Equipment_Used (str), Farm_Accessibility (str), Previous_Outbreak (str)",
        "output_format": "JSON with field: Outbreak in next 7 days (Yes/No)"
    }
