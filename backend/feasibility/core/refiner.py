import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
from scipy.optimize import minimize
from feasibility.core.constraints import ConstraintManager
from feasibility.core.density_estimator import DensityValidator
from feasibility.core.optimizers import feasibility_loss

class CounterfactualFeasibilityRefiner:
    def __init__(self, X_train, constraints_config, density_threshold=0.1):
        self.constraint_mgr = ConstraintManager(constraints_config)
        self.density_validator = DensityValidator(X_train, density_threshold)
        self.X_train = X_train
        self.bounds = [(np.min(X_train[:, i]), np.max(X_train[:, i])) 
                       for i in range(X_train.shape[1])]
    
    def refine(self, original, raw_cfs):
        return [self._refine_single(original, cf) for cf in raw_cfs]
    
    def _refine_single(self, original, raw_cf):
        # Apply hard constraints
        cf = self.constraint_mgr.apply_hard_constraints(original, raw_cf.copy())
        
        # Return immediately if plausible
        if self.density_validator.is_plausible(cf):
            return cf
        
        # Optimization-based refinement
        result = minimize(
            fun=feasibility_loss,
            x0=cf,
            args=(original, raw_cf, self.constraint_mgr, self.density_validator),
            bounds=self.bounds,
            method='SLSQP',
            options={'maxiter': 100}
        )
        return result.x