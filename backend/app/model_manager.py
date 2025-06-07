import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#from Foil_Trees import domain_mappers, contrastive_explanation

class ModelManager:
    def __init__(self):
        self.models = {}
        self.datasets = {}
        self.SEED = np.random.RandomState(1994)
    
    def load_dataset(self, dataset_name):
        """Load and cache dataset"""
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        
        # Get dataset config
        from .config import DATASETS
        config = DATASETS[dataset_name]
        
        # Load and preprocess data
        df = pd.read_csv(config['path'])
        X = df.drop(config['target_column'], axis=1).values
        y = df[config['target_column']].values
        
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
            'feature_names': df.drop(config['target_column'], axis=1).columns.tolist(),
            'target_names': config['class_labels']
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
            print('F1 Score:', metrics.f1_score(dataset['y_test'], model.predict(dataset['X_test']), average='weighted'))
        # Add other models here
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