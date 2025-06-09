import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
import scipy.optimize
from backend.app.config import DATASETS
from backend.app.model_manager import ModelManager

# Initialize
model_manager = ModelManager(datasets_config=DATASETS)

class FOCUSCounterfactual:
    def __init__(self, model, feature_names, sigma=1.0, tau=1.0, lambda_param=0.1, max_iter=1000):
        self.model = model
        self.feature_names = feature_names
        self.sigma = sigma  # For sigmoid approximation
        self.tau = tau      # For softmax temperature
        self.lambda_param = lambda_param
        self.max_iter = max_iter
    
    def sigmoid_approx(self, x, threshold):
        """Sigmoid approximation of step function"""
        return 1.0 / (1.0 + np.exp(-self.sigma * (x - threshold)))
    
    def approximate_tree_prediction(self, x, tree):
        """Approximate prediction of a single decision tree"""
        # This is a simplified approximation since we don't have access to tree internals
        # In practice, you would need to access the tree structure directly
        try:
            # Get actual prediction from tree
            if hasattr(tree, 'predict_proba'):
                proba = tree.predict_proba(x.reshape(1, -1))[0]
            else:
                pred = tree.predict(x.reshape(1, -1))[0]
                proba = np.array([1-pred, pred])
            
            # Apply sigmoid smoothing to make it differentiable
            # This is a simplified version - full FOCUS would approximate tree structure
            smoothed_proba = np.exp(self.tau * proba) / np.sum(np.exp(self.tau * proba))
            return smoothed_proba
        except:
            return np.array([0.5, 0.5])
    
    def approximate_ensemble_prediction(self, x):
        """Approximate prediction of the entire ensemble"""
        if hasattr(self.model, 'estimators_'):
            # For ensemble models, approximate each tree
            all_probas = []
            for tree in self.model.estimators_:
                tree_proba = self.approximate_tree_prediction(x, tree)
                all_probas.append(tree_proba)
            
            # Average predictions with softmax
            avg_proba = np.mean(all_probas, axis=0)
            smoothed_proba = np.exp(self.tau * avg_proba) / np.sum(np.exp(self.tau * avg_proba))
            return smoothed_proba
        else:
            # For single models, use direct approximation
            try:
                proba = self.model.predict_proba(x.reshape(1, -1))[0]
                smoothed_proba = np.exp(self.tau * proba) / np.sum(np.exp(self.tau * proba))
                return smoothed_proba
            except:
                return np.array([0.5, 0.5])
    
    def focus_loss(self, x_cf, x_orig, desired_class):
        """FOCUS loss function"""
        x_cf = np.clip(x_cf, x_orig - 5, x_orig + 5)  # Add bounds
        
        # Prediction loss using approximated probabilities
        approx_proba = self.approximate_ensemble_prediction(x_cf)
        pred_loss = (1.0 - approx_proba[desired_class]) ** 2
        
        # Distance loss
        dist_loss = np.sum((x_cf - x_orig) ** 2)
        
        return pred_loss + self.lambda_param * dist_loss
    
    def generate_counterfactual(self, x_orig, desired_class=None):
        """Generate FOCUS counterfactual explanation"""
        x_orig = np.array(x_orig).flatten()
        
        # Get original prediction
        orig_pred = self.model.predict(x_orig.reshape(1, -1))[0]
        
        # Set desired class (opposite of original)
        if desired_class is None:
            desired_class = 1 - orig_pred
        
        # Try multiple optimizations with different initializations
        best_result = None
        best_loss = float('inf')
        
        for attempt in range(5):
            # Initialize with small perturbation
            x_cf = x_orig + np.random.normal(0, 0.1, size=x_orig.shape)
            
            # Optimization with bounds
            bounds = [(x_orig[i] - 3, x_orig[i] + 3) for i in range(len(x_orig))]
            
            try:
                result = scipy.optimize.minimize(
                    fun=lambda x: self.focus_loss(x, x_orig, desired_class),
                    x0=x_cf,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.max_iter}
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
            except:
                continue
        
        if best_result is None:
            # If optimization failed, return original
            x_cf_final = x_orig
        else:
            x_cf_final = best_result.x
        
        # Check if counterfactual is valid
        try:
            cf_pred = self.model.predict(x_cf_final.reshape(1, -1))[0]
        except:
            cf_pred = orig_pred
        
        # Create changes dictionary
        changes = {}
        for i, feature in enumerate(self.feature_names):
            if abs(x_cf_final[i] - x_orig[i]) > 1e-3:
                changes[feature] = {
                    'original': float(x_orig[i]),
                    'counterfactual': float(x_cf_final[i]),
                    'change': float(x_cf_final[i] - x_orig[i])
                }
        
        return {
            'method': 'FOCUS',
            'counterfactual': x_cf_final,
            'changes': changes,
            'original_prediction': int(orig_pred),
            'counterfactual_prediction': int(cf_pred),
            'valid': cf_pred != orig_pred,
            'distance': float(np.linalg.norm(x_cf_final - x_orig))
        }

model = model_manager.get_model(
    dataset_name="heart",
    model_type="mlp"
)
heart_config = DATASETS["heart"]
feature_names = list(heart_config["feature_types"].keys())


# Test FOCUS method
focus = FOCUSCounterfactual(model, feature_names, sigma=2.0, tau=2.0, lambda_param=0.05)
test_instance = [44,1,2,130,233,0,1,179,1,0.4,2,0,2]
focus_result = focus.generate_counterfactual(test_instance)

print("=== FOCUS METHOD ===")
print(f"Original prediction: {focus_result['original_prediction']}")
print(f"Counterfactual prediction: {focus_result['counterfactual_prediction']}")
print(f"Valid counterfactual: {focus_result['valid']}")
print(f"Distance: {focus_result['distance']:.3f}")
print(f"Number of features changed: {len(focus_result['changes'])}")

if focus_result['changes']:
    print("\nTop 3 feature changes:")
    sorted_changes = sorted(focus_result['changes'].items(), key=lambda x: abs(x[1]['change']), reverse=True)
    for feat, change_info in sorted_changes[:3]:
        print(f"  {feat}: {change_info['original']:.3f} → {change_info['counterfactual']:.3f} (Δ{change_info['change']:.3f})")