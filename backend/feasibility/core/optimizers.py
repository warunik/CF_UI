import numpy as np

def feasibility_loss(cf, original, raw_cf, constraint_mgr, density_validator):
    # Proximity loss (L1 distance)
    proximity = np.linalg.norm(cf - original, 1)
    
    # Density loss (negative log likelihood)
    density = -density_validator.score(cf)
    
    # Fidelity loss (distance from raw CF)
    fidelity = np.linalg.norm(cf - raw_cf, 2)
    
    # Constraint violation penalty
    violations = len(constraint_mgr.check_causal_constraints(original, cf))
    
    return (
        0.5 * proximity +
        0.3 * density +
        0.1 * fidelity +
        0.1 * violations * 10  # Penalize each violation
    )