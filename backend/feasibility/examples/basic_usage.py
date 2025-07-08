import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
from feasibility.core.refiner import CounterfactualFeasibilityRefiner
from feasibility.config.schema import ConstraintsConfig 
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config(r'C:\Users\SINGER\CF_UI\backend\feasibility\config\default_constraints.yaml')
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 5)  # 100 samples, 5 features
    
    # Create refiner instance
    refiner = CounterfactualFeasibilityRefiner(
        X_train=X_train,
        constraints_config=config
    )
    
    # Original instance and raw counterfactuals
    original = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    raw_cfs = [
        np.array([0.1, 0.1, 0.3, 1.5, 0.5]),  # Violates lower bound constraint
        np.array([0.1, 0.2, 0.2, 0.4, 0.5]),  # Violates immutable feature
        np.array([0.1, 0.3, 0.3, 0.4, 0.6])   # Valid
    ]
    
    # Refine counterfactuals
    feasible_cfs = refiner.refine(original, raw_cfs)
    
    # Print results
    print("Original:", original)
    for i, (raw, feasible) in enumerate(zip(raw_cfs, feasible_cfs)):
        print(f"\nRaw CF {i}:", raw)
        print(f"Feasible CF {i}:", feasible)
        print(f"Changes: {np.where(np.abs(raw - feasible) > 1e-5)[0]}")

if __name__ == "__main__":
    main()