import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from counterfactuals.foil_trees import domain_mappers, contrastive_explanation

class ModelManager:
    def __init__(self, datasets_config=None, models_dir="models"):
        self.models = {}
        self.datasets = {}
        self.SEED = np.random.RandomState(1994)
        self.datasets_config = datasets_config or {}
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.imputers = {}

    def _model_path(self, dataset_name, model_type):
        return self.models_dir / f"{dataset_name}_{model_type}.pkl"

    def save_model(self, dataset_name, model_type, model):
        with open(self._model_path(dataset_name, model_type), "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {model_type} for '{dataset_name}' to disk.")

    def load_saved_model(self, dataset_name, model_type):
        path = self._model_path(dataset_name, model_type)
        if path.exists():
            with open(path, "rb") as f:
                model = pickle.load(f)
            #print(f"Loaded saved {model_type} for '{dataset_name}'.")
            return model
        return None

    def load_dataset(self, dataset_name):
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        
        config = self.datasets_config[dataset_name]
        df = pd.read_csv(config['path'])
        
        # Handle feature removal
        if 'drop_columns' in config:
            cols_to_drop = [col for col in config['drop_columns'] if col in df.columns]
            df = df.drop(columns=cols_to_drop)
        
        # Encode target variable
        if config['target_column'] not in df.columns:
            raise ValueError(f"Target column '{config['target_column']}' not found in dataset")
        
        le = LabelEncoder()
        y = le.fit_transform(df[config['target_column']])
        
        # Separate features
        X = df.drop(columns=[config['target_column']])
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # One-hot encode categorical features
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        feature_names = X.columns.tolist()
        X = X.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.SEED
        )
        
        dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'label_encoder': le,  # Store for inverse transforms
            'categorical_cols': categorical_cols,  # Store original categorical columns
            'base_columns': list(df.drop(columns=[config['target_column']]).columns)
        }
        self.datasets[dataset_name] = dataset
        return dataset

    def get_model(self, dataset_name, model_type):
        key = (dataset_name, model_type)
        if key in self.models:
            return self.models[key]
        
        model = self.load_saved_model(dataset_name, model_type)
        if model is not None:
            self.models[key] = model
            return model
        
        dataset = self.load_dataset(dataset_name)
        X_train, y_train = dataset['X_train'], dataset['y_train']
        
        # Handle missing values for sensitive models
        if model_type in ["mlp", "logistic_regression"]:
            strategy = 'mean' if model_type == "mlp" else 'most_frequent'
            imputer = SimpleImputer(strategy=strategy)
            X_train = imputer.fit_transform(X_train)
        
        if model_type == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(50,),
                random_state=self.SEED,
                max_iter=1000
            ).fit(X_train, y_train)

        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.SEED
            ).fit(X_train, y_train)

        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.SEED
            ).fit(X_train, y_train)

        elif model_type == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                random_state=self.SEED
            ).fit(X_train, y_train)

        elif model_type == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(
                random_state=self.SEED
            ).fit(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.save_model(dataset_name, model_type, model)
        self.models[key] = model
        return model
        
    def get_X_train(self, dataset_name):
        dataset = self.load_dataset(dataset_name)
        X_train, y_train = dataset['X_train'], dataset['y_train']
        return X_train
    
    def predict(self, dataset_name, model_type, input_data):
        # Load dataset and model
        dataset = self.load_dataset(dataset_name)
        model = self.get_model(dataset_name, model_type)
        config = self.datasets_config[dataset_name]
        
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = input_data.copy()
        
        # Convert data types based on config
        for col in input_df.columns:
            if col in config['feature_types']:
                if config['feature_types'][col] == 'numeric':
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                elif config['feature_types'][col] == 'categorical':
                    input_df[col] = input_df[col].astype(str)
        
        # Drop unwanted columns
        if 'drop_columns' in config:
            cols_to_drop = [col for col in config['drop_columns'] if col in input_df.columns]
            input_df = input_df.drop(columns=cols_to_drop)
        
        # Drop target column if present
        if config['target_column'] in input_df.columns:
            input_df = input_df.drop(columns=[config['target_column']])
        
        # Ensure all base columns are present
        for col in dataset['base_columns']:
            if col not in input_df.columns:
                input_df[col] = np.nan
        
        # One-hot encode categorical features
        categorical_cols = [col for col in dataset['categorical_cols'] if col in input_df.columns]
        if categorical_cols:
            input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # Align columns with training data
        aligned_df = pd.DataFrame(columns=dataset['feature_names'])
        for col in aligned_df.columns:
            if col in input_df.columns:
                aligned_df[col] = input_df[col]
            else:
                aligned_df[col] = 0
        
        # Convert to numpy array
        input_processed = aligned_df.values
        
        # Debug: Print processed features
        # print("Processed Features:")
        # print(aligned_df)
        
        # Apply imputation
        key = (dataset_name, model_type)
        if key not in self.imputers:
            strategy = 'most_frequent'  # Default for all models
            if model_type == "mlp":
                strategy = 'mean'
            self.imputers[key] = SimpleImputer(strategy=strategy)
            # Fit on training data
            X_train = dataset['X_train']
            self.imputers[key].fit(X_train)
        
        input_processed = self.imputers[key].transform(input_processed)
        
        # Make predictions
        preds = model.predict(input_processed)
        
        # Debug: Print probabilities
        # print(f"Prediction probabilities: {model.predict_proba(input_processed)}")
        
        # Get config reference
        config = self.datasets_config[dataset_name]
        return preds
        
        # # Convert numerical predictions to class names
        # if 'class_labels' in config:
        #     # Map numerical predictions to string labels
        #     class_labels = config['class_labels']
        #     # Ensure predictions are integers for mapping
        #     preds = preds.astype(int)
        #     # Map each prediction to its class name
        #     class_names = [class_labels[pred] for pred in preds]
        #     return class_names
        # else:
        #     # Fallback to numerical values if no class_labels mapping exists
        #     le = dataset['label_encoder']
        #     return le.inverse_transform(preds)

    def generate_counterfactual(self, model, dataset, instance, method="foiltrees"):
        """Generate counterfactual explanation"""
        # 1. Load your processed dataset info
        data_config = self.load_dataset(dataset)
        feature_names = data_config['feature_names']      # <-- the actual columns used by your model
        # class_labels comes from your CONFIG, not from data_config
        class_labels = list(self.datasets_config[dataset]['class_labels'].values())

        # 2. Build the input array in the same order as your training features
        input_array = np.array([instance.get(f, 0) for f in feature_names])

        # 3. Get your trained model
        model_ = self.get_model(dataset_name=dataset, model_type=model)

        # 4. Prepare the domain mapper (needs to know the feature_names and the train data)
        dm = domain_mappers.DomainMapperTabular(
            train_data=self.get_X_train(dataset_name=dataset),
            feature_names=feature_names,
            contrast_names=class_labels
        )

        # 5. Build the foil-trees explainer and generate the CF
        exp = contrastive_explanation.ContrastiveExplanation(dm)
        if method == "foiltrees":
            return exp.explain_instance_domain(model_.predict_proba, input_array)

        raise ValueError(f"Unsupported CF method: {method}")


# if __name__ == "__main__":
#     from config import DATASETS
#     import numpy as np

#     # Create an instance of ModelManager
#     manager = ModelManager(datasets_config=DATASETS)

#     input_data = {
#         "Pregnancies": 3,
#         "Glucose": 110,
#         "BloodPressure": 92,
#         "SkinThickness": 0,
#         "Insulin": 2,
#         "BMI": 34,
#         "DiabetesPedigreeFunction": 0.191,
#         "Age": 40,
#     }

#     # Call the method through the manager instance
#     cf = manager.generate_counterfactual(
#         model="random_forest",  # Fixed typo in model name
#         dataset="diabetes",
#         instance=input_data,  # Pass as dictionary
#         method="foiltrees"
#     )
    
#     print("\nCounterfactual Explanation:\n", cf, "\n")
    