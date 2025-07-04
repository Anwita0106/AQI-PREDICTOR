from fastapi import FastAPI
from pydantic import BaseModel
from predictor.aqi_model import AQIPredictor  # Import your model

app = FastAPI()


# Load and train the model once
predictor = AQIPredictor()
df = predictor.load_and_preprocess_data("predictor/city_day.csv")
X, y, features = predictor.prepare_features()
predictor.train_model(X, y)

# Define input format
class AQIInput(BaseModel):
    PM25: float = 0
    PM10: float = 0
    NO2: float = 0
    O3: float = 0
    CO: float = 0
    SO2: float = 0
    NH3: float = 0
    City: str = ""

@app.post("/predict")
def predict_aqi(input: AQIInput):
    # Prepare input sample
    sample = {
        "PM2.5": input.PM25,
        "PM10": input.PM10,
        "NO2": input.NO2,
        "O3": input.O3,
        "CO": input.CO,
        "SO2": input.SO2,
        "NH3": input.NH3
    }


    # Encode the city
    for col in predictor.feature_names:
        if col.startswith("City_"):
            sample[col] = 1 if col == f"City_{input.City}" else 0

    
    # Predict AQI
    predicted_aqi = predictor.predict_single_sample(sample)

    return {
        "predicted_AQI": round(predicted_aqi, 2)
    }
