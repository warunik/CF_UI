HEART_CONSTRAINTS = {
    'age': lambda orig, new: new > orig,
    'sex': lambda orig, new: orig == new,
    'cp': lambda orig, new: new >= 0 and new <= 3,
    'trestbps': lambda orig, new: new >= 90 and new <= 200,
    'chol': lambda orig, new: new >= 100 and new <= 600,
    'fbs': lambda orig, new: new >= 0 and new <= 1,
    'thalach': lambda orig, new: new >= 60 and new <= 220,
    'exang': lambda orig, new: new >= 0 and new <= 1,
    'oldpeak': lambda orig, new: new >= 0,
    'ca': lambda orig, new: new >= 0 and new <= 4,
    'thal': lambda orig, new: new >= 0 and new <= 3
}

def check_feasibility(original, counterfactual):
    """
    Checks if a counterfactual is feasible based on predefined constraints
    
    Args:
        original (dict): Original instance with feature values
        counterfactual (dict): Proposed counterfactual with feature values
    
    Returns:
        tuple: (bool, list) indicating overall feasibility and list of violated constraints
    """
    violations = []
    
    for feature, constraint in HEART_CONSTRAINTS.items():
        orig_val = original[feature]
        cf_val = counterfactual[feature]
        
        if not constraint(orig_val, cf_val):
            violations.append(feature)
    
    is_feasible = len(violations) == 0
    return (is_feasible, violations)

original_patient = {
    'age': 54,
    'sex': 1,
    'cp': 2,
    'trestbps': 125,
    'chol': 240,
    'fbs': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.8,
    'ca': 0,
    'thal': 2
}

counterfactual_patient = {
    'age': 50,          # VIOLATION (must be greater than original)
    'sex': 0,           # VIOLATION (must match original)
    'cp': 2,            # Valid
    'trestbps': 85,     # VIOLATION (below 90)
    'chol': 650,        # VIOLATION (above 600)
    'fbs': 0,           # Valid
    'thalach': 230,     # VIOLATION (above 220)
    'exang': 1,         # Valid
    'oldpeak': -0.5,    # VIOLATION (negative)
    'ca': 3,            # Valid
    'thal': 7           # VIOLATION (above 3)
}

feasible, violations = check_feasibility(original_patient, counterfactual_patient)

print(f"Overall feasible: {feasible}")
print(f"Violations: {violations}")