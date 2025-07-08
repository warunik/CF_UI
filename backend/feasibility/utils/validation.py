import numpy as np
from typing import List, Union
from pydantic import validate_arguments

@validate_arguments
def validate_feature_vector(x: Union[List[float], np.ndarray], expected_dim: int):
    """Validate a feature vector"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Feature vector must be 1D, got {x.ndim}D")
    if len(x) != expected_dim:
        raise ValueError(f"Expected {expected_dim} features, got {len(x)}")
    return x

@validate_arguments
def validate_counterfactuals(cfs: List[Union[List[float], np.ndarray]], 
                            original: np.ndarray):
    """Validate counterfactuals against original instance"""
    validated = []
    for cf in cfs:
        cf_arr = validate_feature_vector(cf, len(original))
        if cf_arr.shape != original.shape:
            raise ValueError("Counterfactual shape doesn't match original")
        validated.append(cf_arr)
    return validated

@validate_arguments
def validate_config(config: dict):
    """Validate configuration using Pydantic model"""
    from feasibility.config.schema import ConstraintsConfig
    return ConstraintsConfig(**config).dict()

def validate_data_ranges(X_train: np.ndarray, counterfactuals: List[np.ndarray]):
    """Ensure counterfactuals stay within training data ranges"""
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)
    
    violations = []
    for i, cf in enumerate(counterfactuals):
        out_of_bounds = np.where((cf < min_vals) | (cf > max_vals))[0]
        if len(out_of_bounds) > 0:
            violations.append({
                'cf_index': i,
                'features': out_of_bounds.tolist(),
                'min': min_vals[out_of_bounds],
                'max': max_vals[out_of_bounds],
                'values': cf[out_of_bounds]
            })
    return violations