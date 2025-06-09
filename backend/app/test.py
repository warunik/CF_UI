import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DATASETS, ML_MODELS
from model_manager import ModelManager
from counterfactuals.foil_trees import domain_mappers, contrastive_explanation
import numpy as np

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

feature_names = list(DATASETS["heart"]["feature_types"].keys())
input_array = np.array([input_data[feature] for feature in feature_names])

# Get prediction
prediction = model_manager.predict(
    dataset_name="heart",
    model_type="mlp",
    input_data=input_data
)[0]

print(f"Prediction for input data: {prediction}")

# Get the heart dataset config
heart_config = DATASETS["heart"]

# Extract feature names and class labels
feature_names = list(heart_config["feature_types"].keys())
class_labels = list(heart_config["class_labels"].values())

dm = domain_mappers.DomainMapperTabular(
    train_data=model_manager.get_X_train(dataset_name="heart"),
    feature_names=feature_names,
    contrast_names=class_labels  # Use the class labels
)

exp = contrastive_explanation.ContrastiveExplanation(dm)

model = model_manager.get_model(
    dataset_name="heart",
    model_type="mlp"
)

print("\n", exp.explain_instance_domain(model.predict_proba, input_array), "\n")