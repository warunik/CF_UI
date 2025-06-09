# Let me improve the Wachter implementation and try with different parameters
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
import scipy.optimize
from backend.app.config import DATASETS
from backend.app.model_manager import ModelManager

# Initialize
model_manager = ModelManager(datasets_config=DATASETS)

class WachterCounterfactual:
    def __init__(self, model, feature_names, lambda_param=1.0, max_iter=1000, lr=0.01):
        self.model = model
        self.feature_names = feature_names
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.lr = lr
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            pred = self.model.predict(X)
            proba = np.zeros((len(pred), 2))
            proba[pred == 0, 0] = 1
            proba[pred == 1, 1] = 1
            return proba
    
    def loss_function(self, x_cf, x_orig, desired_class):
        """Wachter loss function with better formulation"""
        x_cf = x_cf.reshape(1, -1)
        
        # Prediction loss - maximize probability of desired class
        pred_proba = self.predict_proba(x_cf)[0]
        pred_loss = (1.0 - pred_proba[desired_class]) ** 2
        
        # Distance loss - L2 distance from original
        dist_loss = np.sum((x_cf.flatten() - x_orig) ** 2)
        
        return pred_loss + self.lambda_param * dist_loss
    
    def generate_counterfactual(self, x_orig, desired_class=None):
        """Generate counterfactual explanation"""
        x_orig = np.array(x_orig).flatten()
        
        # Get original prediction
        orig_pred = self.model.predict(x_orig.reshape(1, -1))[0]
        
        # Set desired class (opposite of original)
        if desired_class is None:
            desired_class = 1 - orig_pred
        
        # Try multiple random initializations
        best_result = None
        best_loss = float('inf')
        
        for attempt in range(5):  # Try 5 different initializations
            # Initialize with small random perturbation
            x_cf = x_orig + np.random.normal(0, 0.1, size=x_orig.shape)
            
            # Optimization using scipy
            result = scipy.optimize.minimize(
                fun=lambda x: self.loss_function(x, x_orig, desired_class),
                x0=x_cf,
                method='L-BFGS-B',
                options={'maxiter': self.max_iter}
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        
        x_cf_final = best_result.x
        
        # Check if counterfactual is valid
        cf_pred = self.model.predict(x_cf_final.reshape(1, -1))[0]
        
        # Create result dictionary
        changes = {}
        for i, feature in enumerate(self.feature_names):
            if abs(x_cf_final[i] - x_orig[i]) > 1e-4:  # Lower threshold for changes
                changes[feature] = {
                    'original': float(x_orig[i]),
                    'counterfactual': float(x_cf_final[i]),
                    'change': float(x_cf_final[i] - x_orig[i])
                }
        
        return {
            'method': 'Wachter',
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

# Test improved Wachter method
wachter = WachterCounterfactual(model, feature_names, lambda_param=0.1)
test_instance = [44,1,2,130,233,0,1,179,1,0.4,2,0,2]
wachter_result = wachter.generate_counterfactual(test_instance)

print("=== IMPROVED WACHTER METHOD ===")
print(f"Original prediction: {wachter_result['original_prediction']}")
print(f"Counterfactual prediction: {wachter_result['counterfactual_prediction']}")
print(f"Valid counterfactual: {wachter_result['valid']}")
print(f"Distance: {wachter_result['distance']:.3f}")
print(f"Number of features changed: {len(wachter_result['changes'])}")
print("\nTop 3 feature changes:")
sorted_changes = sorted(wachter_result['changes'].items(), key=lambda x: abs(x[1]['change']), reverse=True)
for feat, change_info in sorted_changes[:3]:
    print(f"  {feat}: {change_info['original']:.3f} → {change_info['counterfactual']:.3f} (Δ{change_info['change']:.3f})")