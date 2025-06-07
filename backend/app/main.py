from config import DATASETS, ML_MODELS
from model_manager import ModelManager
import pandas as pd
from sklearn import metrics

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    return {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred, average="weighted"),
        "recall": metrics.recall_score(y_test, y_pred, average="weighted"),
        "f1": metrics.f1_score(y_test, y_pred, average="weighted")
    }

def main():
    # Initialize model manager
    manager = ModelManager()
    
    # Add our datasets config to the manager
    manager.datasets_config = DATASETS
    
    # Store results
    results = []
    
    # Iterate through all datasets and models
    for dataset_name in DATASETS.keys():
        for model_type in ML_MODELS.keys():
            print(f"\n{'='*50}")
            print(f"Training {ML_MODELS[model_type]} on {DATASETS[dataset_name]['name']}")
            print(f"{'='*50}")
            
            try:
                # Get trained model
                model = manager.get_model(dataset_name, model_type)
                
                # Load dataset
                dataset = manager.load_dataset(dataset_name)
                
                # Evaluate
                metrics = evaluate_model(
                    model, 
                    dataset['X_test'], 
                    dataset['y_test']
                )
                
                # Store results
                results.append({
                    "dataset": DATASETS[dataset_name]['name'],
                    "model": ML_MODELS[model_type],
                    **metrics
                })
                
                # Print current results
                print(f"\nResults for {ML_MODELS[model_type]} on {dataset_name}:")
                for metric, value in metrics.items():
                    print(f"{metric.capitalize()}: {value:.4f}")
                    
            except Exception as e:
                print(f"Error processing {model_type} on {dataset_name}: {str(e)}")
                results.append({
                    "dataset": DATASETS[dataset_name]['name'],
                    "model": ML_MODELS[model_type],
                    "error": str(e)
                })
    
    # Create and save results dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_evaluation_results.csv", index=False)
    print("\nSaved results to model_evaluation_results.csv")
    
    # Print final results table
    print("\nFinal Evaluation Results:")
    print(results_df)

if __name__ == "__main__":
    main()