import joblib
import os
import pandas as pd

# Load the pre-trained model from a specified path
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    return model

# Make a prediction using the loaded model
def make_prediction(model, input_data: dict):
    try:
        # Make sure the features match the training data
        expected_columns = [
            "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", 
            "population", "households", "median_income", "ocean_proximity"
        ]
        
        # Convert input to a dataframe
        input_df = pd.DataFrame([input_data], columns=expected_columns)
        
        
        prediction = model.predict(input_df)[0]
        
        return float(prediction)
    except KeyError as missing_feature:
        raise ValueError(f"Missing input feature {missing_feature}")
    
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    model_path = os.path.join(project_root, "model.pkl")
    
    sample_input = {
        'longitude': -122.25,
        'latitude': 37.85,
        'housing_median_age': 52.0,
        'total_rooms': 880.0,
        'total_bedrooms': 160.0,
        'population': 600.0,
        'households': 150.0,
        'median_income': 5.0000,
        'ocean_proximity': 'NEAR BAY'
    }
    
    loaded_model = load_model(model_path)
    prediction = make_prediction(loaded_model, sample_input)
    print(f"Predicted house value: {prediction}")
        