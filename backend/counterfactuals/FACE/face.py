import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
import scipy.optimize
from backend.app.config import DATASETS
from backend.app.model_manager import ModelManager

# Initialize
model_manager = ModelManager(datasets_config=DATASETS)

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor

class FACECounterfactual:
    def __init__(self, model, feature_names, X_train, k_neighbors=5, density_weight=0.5):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.k_neighbors = k_neighbors
        self.density_weight = density_weight
        
        # Fit nearest neighbors for density estimation
        self.nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        self.nn.fit(X_train)
        
        # Fit LOF for outlier detection
        self.lof = LocalOutlierFactor(n_neighbors=k_neighbors, novelty=True)
        self.lof.fit(X_train)
    
    def get_density_score(self, x):
        """Get density score based on distance to k nearest neighbors"""
        x = x.reshape(1, -1)
        distances, _ = self.nn.kneighbors(x)
        # Higher density = lower average distance to neighbors
        avg_distance = np.mean(distances)
        return 1.0 / (1.0 + avg_distance)
    
    def get_feasibility_score(self, x):
        """Get feasibility score based on LOF (Local Outlier Factor)"""
        x = x.reshape(1, -1)
        try:
            lof_score = self.lof.decision_function(x)[0]
            # Convert to positive score (higher = more feasible)
            return max(0, lof_score + 1)
        except:
            return 0.5
    
    def find_path_to_target(self, x_start, target_class, max_steps=50, step_size=0.1):
        """Find a feasible path from x_start to target class"""
        x_current = x_start.copy()
        path = [x_current.copy()]
        
        for step in range(max_steps):
            # Check if we've reached the target class
            try:
                current_pred = self.model.predict(x_current.reshape(1, -1))[0]
                if current_pred == target_class:
                    break
            except:
                break
            
            # Find high-density neighbors
            distances, neighbor_indices = self.nn.kneighbors(x_current.reshape(1, -1), 
                                                           n_neighbors=min(20, len(self.X_train)))
            
            best_direction = None
            best_score = -float('inf')
            
            # Try directions towards high-density neighbors
            for neighbor_idx in neighbor_indices[0]:
                neighbor = self.X_train[neighbor_idx]
                direction = neighbor - x_current
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    direction = direction / direction_norm
                    
                    # Test step in this direction
                    x_test = x_current + step_size * direction
                    
                    # Score based on density and progress toward target
                    density_score = self.get_density_score(x_test)
                    feasibility_score = self.get_feasibility_score(x_test)
                    
                    # Check if this direction helps with classification
                    try:
                        test_proba = self.model.predict_proba(x_test.reshape(1, -1))[0]
                        progress_score = test_proba[target_class]
                    except:
                        progress_score = 0.0
                    
                    total_score = (self.density_weight * density_score + 
                                 self.density_weight * feasibility_score + 
                                 (1 - 2 * self.density_weight) * progress_score)
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_direction = direction
            
            # Move in best direction or random if no good direction found
            if best_direction is not None:
                x_current = x_current + step_size * best_direction
            else:
                # Random small step
                x_current = x_current + np.random.normal(0, step_size * 0.5, size=x_current.shape)
            
            path.append(x_current.copy())
        
        return path
    
    def generate_counterfactual(self, x_orig, desired_class=None):
        """Generate FACE counterfactual explanation"""
        x_orig = np.array(x_orig).flatten()
        
        # Get original prediction
        orig_pred = self.model.predict(x_orig.reshape(1, -1))[0]
        
        # Set desired class (opposite of original)
        if desired_class is None:
            desired_class = 1 - orig_pred
        
        # Find path to target class
        path = self.find_path_to_target(x_orig, desired_class)
        
        # Use the last point in the path as the counterfactual
        x_cf = path[-1]
        
        # Verify the counterfactual
        try:
            cf_pred = self.model.predict(x_cf.reshape(1, -1))[0]
        except:
            cf_pred = orig_pred
        
        # Create changes dictionary
        changes = {}
        for i, feature in enumerate(self.feature_names):
            if abs(x_cf[i] - x_orig[i]) > 1e-3:
                changes[feature] = {
                    'original': float(x_orig[i]),
                    'counterfactual': float(x_cf[i]),
                    'change': float(x_cf[i] - x_orig[i])
                }
        
        # Additional FACE-specific metrics
        density_score = self.get_density_score(x_cf)
        feasibility_score = self.get_feasibility_score(x_cf)
        
        return {
            'method': 'FACE',
            'counterfactual': x_cf,
            'changes': changes,
            'original_prediction': int(orig_pred),
            'counterfactual_prediction': int(cf_pred),
            'valid': cf_pred != orig_pred,
            'distance': float(np.linalg.norm(x_cf - x_orig)),
            'density_score': float(density_score),
            'feasibility_score': float(feasibility_score),
            'path_length': len(path)
        }

model = model_manager.get_model(
    dataset_name="heart",
    model_type="mlp"
)
heart_config = DATASETS["heart"]
feature_names = list(heart_config["feature_types"].keys())

X_train = model_manager.get_X_train(
    dataset_name="heart"
)

# Test FACE method
face = FACECounterfactual(model, feature_names, X_train)
test_instance = [44,1,2,130,233,0,1,179,1,0.4,2,0,2]
face_result = face.generate_counterfactual(test_instance)

print("=== FACE METHOD ===")
print(f"Original prediction: {face_result['original_prediction']}")
print(f"Counterfactual prediction: {face_result['counterfactual_prediction']}")
print(f"Valid counterfactual: {face_result['valid']}")
print(f"Distance: {face_result['distance']:.3f}")
print(f"Density score: {face_result['density_score']:.3f}")
print(f"Feasibility score: {face_result['feasibility_score']:.3f}")
print(f"Path length: {face_result['path_length']}")
print(f"Number of features changed: {len(face_result['changes'])}")

if face_result['changes']:
    print("\nTop 3 feature changes:")
    sorted_changes = sorted(face_result['changes'].items(), key=lambda x: abs(x[1]['change']), reverse=True)
    for feat, change_info in sorted_changes[:3]:
        print(f"  {feat}: {change_info['original']:.3f} → {change_info['counterfactual']:.3f} (Δ{change_info['change']:.3f})")