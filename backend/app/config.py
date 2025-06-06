# config.py
DATASETS = {
    "heart": {
        "name": "Heart Disease",
        "path": "backend/data/heart.csv",
        "class_labels": {0: "No Heart Disease", 1: "Heart Disease"},
        "feature_types": {
            "age": "numeric",
            "sex": "categorical",
            "cp": "categorical",
            "trestbps": "numeric",
            "chol": "numeric",
            "fbs": "categorical",
            "restecg": "categorical",
            "thalach": "numeric",
            "exang": "categorical",
            "oldpeak": "numeric",
            "slope": "categorical",
            "ca": "numeric",
            "thal": "categorical"
        }
    },
    "diabetes": {
        "name": "Diabetes",
        "path": "data/diabetes.csv",
        "class_labels": {0: "No Diabetes", 1: "Diabetes"},
        "feature_types": {
            "pregnancies": "numeric",
            "glucose": "numeric",
            "blood_pressure": "numeric",
            "skin_thickness": "numeric",
            "insulin": "numeric",
            "bmi": "numeric",
            "diabetes_pedigree": "numeric",
            "age": "numeric"
        }
    }
}

ML_MODELS = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "gradient_boosting": "Gradient Boosting",
    "neural_network": "Neural Network"
}

CF_METHODS = {
    "dice": "DiCE (Diverse Counterfactual Explanations)",
    "wachter": "Wachter's Method",
    "cem": "CEM (Contrastive Explanation Method)",
    "face": "FACE (Feasible and Actionable Counterfactual Explanations)"
}