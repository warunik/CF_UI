import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DATASETS, ML_MODELS
from model_manager import ModelManager
from counterfactuals.foil_trees import domain_mappers, contrastive_explanation
import numpy as np

name_data = "diabetes"
name_model = "random_forest"

# Initialize
model_manager = ModelManager(datasets_config=DATASETS)

# Sample input for "heart" dataset
input_data = {
    "Pregnancies": 1,
    "Glucose": 189,
    "BloodPressure": 60,
    "SkinThickness": 23,
    "Insulin": 846,
    "BMI": 30.1,
    "DiabetesPedigreeFunction": 0.398,
    "Age": 50,
}

feature_names = list(DATASETS[name_data]["feature_types"].keys())
input_array = np.array([input_data[feature] for feature in feature_names])

# Get prediction
prediction = model_manager.predict(
    dataset_name=name_data,
    model_type=name_model,
    input_data=input_data
)[0]

print(f"Prediction for input data: {prediction}")

# Get the heart dataset config
heart_config = DATASETS[name_data]

# Extract feature names and class labels
feature_names = list(heart_config["feature_types"].keys())
class_labels = list(heart_config["class_labels"].values())

dm = domain_mappers.DomainMapperTabular(
    train_data=model_manager.get_X_train(dataset_name=name_data),
    feature_names=feature_names,
    contrast_names=class_labels  # Use the class labels
)

exp = contrastive_explanation.ContrastiveExplanation(dm)

model = model_manager.get_model(
    dataset_name=name_data,
    model_type=name_model
)

print("\n", exp.explain_instance_domain(model.predict_proba, input_array), "\n")
