import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#from Foil_Trees import domain_mappers, contrastive_explanation

class ModelManager:
    def __init__(self, datasets_config=None):
        self.models = {}
        self.datasets = {}
        self.SEED = np.random.RandomState(1994)
        self.datasets_config = datasets_config or {}


    def load_dataset(self, dataset_name):
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        
        config = self.datasets_config[dataset_name]
        df = pd.read_csv(config['path'])
        
        # Check if target column exists
        if config['target_column'] not in df.columns:
            raise ValueError(f"Target column '{config['target_column']}' not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[config['target_column']])
        y = df[config['target_column']].values
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # One-hot encode categorical features
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Store feature names BEFORE converting to array
        feature_names = X.columns.tolist()
        
        # Convert to numpy array
        X = X.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.SEED
        )
        
        # Store dataset
        dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names  # Use pre-stored names
        }
        self.datasets[dataset_name] = dataset
        return dataset

    def get_model(self, dataset_name, model_type):
        """Get or train model"""
        key = (dataset_name, model_type)
        if key in self.models:
            return self.models[key]
        
        # Load dataset
        dataset = self.load_dataset(dataset_name)
        
        # Train model
        if model_type == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(50,),
                random_state=self.SEED,
                max_iter=1000
            ).fit(dataset['X_train'], dataset['y_train'])

        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.SEED
            ).fit(dataset['X_train'], dataset['y_train'])

        elif model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                max_iter=1000,
                random_state=self.SEED
            ).fit(dataset['X_train'], dataset['y_train'])

        elif model_type == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                random_state=self.SEED
            ).fit(dataset['X_train'], dataset['y_train'])

        elif model_type == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(
                random_state=self.SEED
            ).fit(dataset['X_train'], dataset['y_train'])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Cache and return
        self.models[key] = model
        return model

    # def generate_counterfactual(self, model, dataset, instance, method="foil_trees"):
    #     """Generate counterfactual explanation"""
    #     if method == "foil_trees":
    #         # Prepare domain mapper
    #         dm = domain_mappers.DomainMapperTabular(
    #             train_data=dataset['X_train'],
    #             feature_names=dataset['feature_names'],
    #             contrast_names=dataset['target_names']
    #         )
            
    #         # Generate explanation
    #         exp = contrastive_explanation.ContrastiveExplanation(dm)
    #         return exp.explain_instance_domain(model.predict_proba, instance)
        
    #     # Add other CF methods here
    #     raise ValueError(f"Unsupported CF method: {method}")