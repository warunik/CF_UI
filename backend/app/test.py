from config import DATASETS, ML_MODELS
from model_manager import ModelManager

# Initialize
model_manager = ModelManager(datasets_config=DATASETS)

# Sample input for "heart" dataset
input_data = {
    "age": 45,
    "sex": 0,
    "cp": 2,
    "trestbps": 112,
    "chol": 149,
    "fbs": 0,
    "restecg": 1,
    "thalach": 125,
    "exang": 0,
    "oldpeak": 1.6,
    "slope": 1,
    "ca": 0,
    "thal": 0
}

# Get prediction
prediction = model_manager.predict(
    dataset_name="heart",
    model_type="mlp",
    input_data=input_data
)[0]

print(f"Prediction for input data: {prediction}")