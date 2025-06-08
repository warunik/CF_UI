# utils.py
import pandas as pd
from pathlib import Path
import os

def load_dataset(dataset_config):
    dataset_path = Path(dataset_config["path"])
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return pd.read_csv(dataset_path)

def format_counterfactual(counterfactual, dataset_config):
    changes = "\n".join([
        f"- {change['feature']}: {change['current']} → {change['new']}"
        for change in counterfactual["required_changes"]
    ])
    
    user_data_str = "\n".join([
        f"- {feature}: {value}" 
        for feature, value in zip(dataset_config["feature_types"].keys(), counterfactual["user_data"])
    ])
    
    class_labels = dataset_config["class_labels"]
    
    return {
        "original_prediction": counterfactual["original_prediction"],
        "original_class": class_labels.get(counterfactual["original_prediction"], ""),
        "new_prediction": counterfactual["new_prediction"],
        "new_class": class_labels.get(counterfactual["new_prediction"], ""),
        "confidence": counterfactual["confidence"],
        "changes": changes,
        "user_data_str": user_data_str,
        "dataset_name": dataset_config["name"]
    }