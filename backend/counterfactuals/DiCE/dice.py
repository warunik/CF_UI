import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
import scipy.optimize
from backend.app.config import DATASETS
from backend.app.model_manager import ModelManager

# Initialize
model_manager = ModelManager(datasets_config=DATASETS)

class DiCECounterfactual:
    def __init__(self, model, feature_names, num_cfs=2, proximity_weight=0.5, diversity_weight=0.1, max_iter=500):
        self.model = model
        self.feature_names = feature_names
        self.num_cfs = num_cfs
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.max_iter = max_iter
    
    def predict_proba(self, X):
        """Get prediction probabilities with safety checks"""
        try:
            # Clip values to reasonable range
            X = np.clip(X, -10, 10)
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                pred = self.model.predict(X)
                proba = np.zeros((len(pred), 2))
                proba[pred == 0, 0] = 1
                proba[pred == 1, 1] = 1
                return proba
        except:
            # Fallback for problematic inputs
            return np.array([[0.5, 0.5]])
    
    def generate_counterfactual(self, x_orig, desired_class=None):
        """Generate counterfactual explanations using a simpler approach"""
        x_orig = np.array(x_orig).flatten()
        
        # Get original prediction
        orig_pred = self.model.predict(x_orig.reshape(1, -1))[0]
        
        # Set desired class (opposite of original)
        if desired_class is None:
            desired_class = 1 - orig_pred
        
        results = []
        
        # Generate multiple counterfactuals
        for cf_idx in range(self.num_cfs):
            best_cf = None
            best_distance = float('inf')
            best_valid = False
            
            # Try multiple random seeds
            for seed in range(20):
                np.random.seed(seed + cf_idx * 100)
                
                # Start with small perturbations in different directions
                perturbation_scale = 0.1 + cf_idx * 0.1  # Different scales for diversity
                x_cf = x_orig + np.random.normal(0, perturbation_scale, size=x_orig.shape)
                
                # Iterative improvement
                for iteration in range(50):
                    # Clip to reasonable bounds
                    x_cf = np.clip(x_cf, x_orig - 3, x_orig + 3)
                    
                    # Check current prediction
                    try:
                        current_pred = self.model.predict(x_cf.reshape(1, -1))[0]
                    except:
                        break
                    
                    if current_pred == desired_class:
                        # Found valid counterfactual
                        distance = np.linalg.norm(x_cf - x_orig)
                        if distance < best_distance:
                            best_cf = x_cf.copy()
                            best_distance = distance
                            best_valid = True
                        break
                    
                    # Move towards desired class
                    try:
                        proba = self.predict_proba(x_cf.reshape(1, -1))[0]
                        gradient_direction = np.random.normal(0, 0.05, size=x_orig.shape)
                        
                        # If not enough probability for desired class, move more aggressively
                        if proba[desired_class] < 0.7:
                            gradient_direction *= 2
                        
                        x_cf = x_cf + gradient_direction
                    except:
                        break
                
                if best_valid:
                    break
            
            # Create result
            if best_cf is not None:
                try:
                    cf_pred = self.model.predict(best_cf.reshape(1, -1))[0]
                except:
                    cf_pred = orig_pred
                
                # Create changes dictionary
                changes = {}
                for i, feature in enumerate(self.feature_names):
                    if abs(best_cf[i] - x_orig[i]) > 1e-3:
                        changes[feature] = {
                            'original': float(x_orig[i]),
                            'counterfactual': float(best_cf[i]),
                            'change': float(best_cf[i] - x_orig[i])
                        }
                
                result_dict = {
                    'method': f'DiCE_CF_{cf_idx+1}',
                    'counterfactual': best_cf,
                    'changes': changes,
                    'original_prediction': int(orig_pred),
                    'counterfactual_prediction': int(cf_pred),
                    'valid': cf_pred != orig_pred,
                    'distance': float(best_distance) if best_distance != float('inf') else 0.0
                }
            else:
                # No valid counterfactual found
                result_dict = {
                    'method': f'DiCE_CF_{cf_idx+1}',
                    'counterfactual': x_orig,
                    'changes': {},
                    'original_prediction': int(orig_pred),
                    'counterfactual_prediction': int(orig_pred),
                    'valid': False,
                    'distance': 0.0
                }
            
            results.append(result_dict)
        
        return results

model = model_manager.get_model(
    dataset_name="heart",
    model_type="mlp"
)
heart_config = DATASETS["heart"]
feature_names = list(heart_config["feature_types"].keys())

# Test the improved DiCE method
dice = DiCECounterfactual(model, feature_names, num_cfs=2)
test_instance = [44,1,2,130,233,0,1,179,1,0.4,2,0,2]
dice_results = dice.generate_counterfactual(test_instance)

print("=== IMPROVED DICE METHOD ===")
for i, result in enumerate(dice_results):
    print(f"\nCounterfactual {i+1}:")
    print(f"  Original prediction: {result['original_prediction']}")
    print(f"  Counterfactual prediction: {result['counterfactual_prediction']}")
    print(f"  Valid counterfactual: {result['valid']}")
    print(f"  Distance: {result['distance']:.3f}")
    print(f"  Number of features changed: {len(result['changes'])}")
    
    if result['changes']:
        print("  Top 3 feature changes:")
        sorted_changes = sorted(result['changes'].items(), key=lambda x: abs(x[1]['change']), reverse=True)
        for feat, change_info in sorted_changes[:3]:
            print(f"    {feat}: {change_info['original']:.3f} → {change_info['counterfactual']:.3f} (Δ{change_info['change']:.3f})")