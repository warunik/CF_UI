import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
import scipy.optimize
from backend.app.config import DATASETS
from backend.app.model_manager import ModelManager

# Initialize
model_manager = ModelManager(datasets_config=DATASETS)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CCHVAECounterfactual:
    def __init__(self, model, feature_names, X_train, latent_dim=5, reconstruction_weight=1.0, prediction_weight=1.0):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.latent_dim = min(latent_dim, X_train.shape[1])
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        
        # Use PCA as a simple autoencoder approximation
        self.encoder = PCA(n_components=self.latent_dim)
        self.encoder.fit(X_train)
        
        # Scaler for latent space
        X_train_latent = self.encoder.transform(X_train)
        self.latent_scaler = StandardScaler()
        self.latent_scaler.fit(X_train_latent)
    
    def encode(self, x):
        """Encode to latent space"""
        x = x.reshape(1, -1)
        z = self.encoder.transform(x)
        z_scaled = self.latent_scaler.transform(z)
        return z_scaled.flatten()
    
    def decode(self, z):
        """Decode from latent space"""
        z = z.reshape(1, -1)
        z_unscaled = self.latent_scaler.inverse_transform(z)
        x_reconstructed = self.encoder.inverse_transform(z_unscaled)
        return x_reconstructed.flatten()
    
    def cchvae_loss(self, z_cf, x_orig, desired_class):
        """CCHVAE loss function"""
        # Decode counterfactual
        x_cf = self.decode(z_cf)
        
        # Clip to reasonable bounds
        x_cf = np.clip(x_cf, x_orig - 5, x_orig + 5)
        
        # Reconstruction loss (stay close to data manifold)
        x_orig_reconstructed = self.decode(self.encode(x_orig))
        reconstruction_loss = np.sum((x_cf - x_orig_reconstructed) ** 2)
        
        # Prediction loss
        try:
            pred_proba = self.model.predict_proba(x_cf.reshape(1, -1))[0]
            prediction_loss = (1.0 - pred_proba[desired_class]) ** 2
        except:
            prediction_loss = 1.0
        
        # Proximity loss in latent space
        z_orig = self.encode(x_orig)
        latent_proximity_loss = np.sum((z_cf - z_orig) ** 2)
        
        return (self.prediction_weight * prediction_loss + 
                self.reconstruction_weight * reconstruction_loss + 
                0.1 * latent_proximity_loss)
    
    def generate_counterfactual(self, x_orig, desired_class=None):
        """Generate CCHVAE counterfactual explanation"""
        x_orig = np.array(x_orig).flatten()
        
        # Get original prediction
        orig_pred = self.model.predict(x_orig.reshape(1, -1))[0]
        
        # Set desired class (opposite of original)
        if desired_class is None:
            desired_class = 1 - orig_pred
        
        # Encode original to latent space
        z_orig = self.encode(x_orig)
        
        # Optimize in latent space
        best_result = None
        best_loss = float('inf')
        
        for attempt in range(5):
            # Initialize with small perturbation in latent space
            z_cf = z_orig + np.random.normal(0, 0.1, size=z_orig.shape)
            
            # Set bounds in latent space
            bounds = [(z_orig[i] - 2, z_orig[i] + 2) for i in range(len(z_orig))]
            
            try:
                result = scipy.optimize.minimize(
                    fun=lambda z: self.cchvae_loss(z, x_orig, desired_class),
                    x0=z_cf,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500}
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
            # Decode the optimized latent representation
            x_cf_final = self.decode(best_result.x)
        
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
            'method': 'CCHVAE',
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

X_train = model_manager.get_X_train(
    dataset_name="heart"
)

# Test CCHVAE method
cchvae = CCHVAECounterfactual(model, feature_names, X_train, latent_dim=3)
test_instance = [44,1,2,130,233,0,1,179,1,0.4,2,0,2]
cchvae_result = cchvae.generate_counterfactual(test_instance)

print("=== CCHVAE METHOD ===")
print(f"Original prediction: {cchvae_result['original_prediction']}")
print(f"Counterfactual prediction: {cchvae_result['counterfactual_prediction']}")
print(f"Valid counterfactual: {cchvae_result['valid']}")
print(f"Distance: {cchvae_result['distance']:.3f}")
print(f"Number of features changed: {len(cchvae_result['changes'])}")

if cchvae_result['changes']:
    print("\nTop 3 feature changes:")
    sorted_changes = sorted(cchvae_result['changes'].items(), key=lambda x: abs(x[1]['change']), reverse=True)
    for feat, change_info in sorted_changes[:3]:
        print(f"  {feat}: {change_info['original']:.3f} → {change_info['counterfactual']:.3f} (Δ{change_info['change']:.3f})")