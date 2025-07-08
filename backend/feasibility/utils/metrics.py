import numpy as np

def calculate_feasibility_score(original, cf, constraint_mgr, density_validator):
    score = 0
    
    # Plausibility
    score += 0.4 if density_validator.is_plausible(cf) else 0
    
    # Constraint satisfaction
    violations = constraint_mgr.check_causal_constraints(original, cf)
    score += 0.3 * (1 - min(len(violations)/5, 1))  # Max 5 violations
    
    # Sparsity
    changed = np.sum(np.abs(cf - original) > 1e-5)
    score += 0.2 * (1 - min(changed/10, 1))  # Max 10 changes
    
    # Action cost
    action_cost = np.mean(np.abs(cf - original))
    score += 0.1 * (1 - min(action_cost/100, 1))  # Normalized
    
    return score