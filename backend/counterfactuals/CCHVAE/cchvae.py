# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# import numpy as np
# import scipy.optimize
# from backend.app.config import DATASETS
# from backend.app.model_manager import ModelManager

# # Initialize
# model_manager = ModelManager(datasets_config=DATASETS)

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# class CCHVAECounterfactual:
#     def __init__(self, model, feature_names, X_train, latent_dim=5, reconstruction_weight=1.0, prediction_weight=1.0):
#         self.model = model
#         self.feature_names = feature_names
#         self.X_train = X_train
#         self.latent_dim = min(latent_dim, X_train.shape[1])
#         self.reconstruction_weight = reconstruction_weight
#         self.prediction_weight = prediction_weight
        
#         # Use PCA as a simple autoencoder approximation
#         self.encoder = PCA(n_components=self.latent_dim)
#         self.encoder.fit(X_train)
        
#         # Scaler for latent space
#         X_train_latent = self.encoder.transform(X_train)
#         self.latent_scaler = StandardScaler()
#         self.latent_scaler.fit(X_train_latent)
    
#     def encode(self, x):
#         """Encode to latent space"""
#         x = x.reshape(1, -1)
#         z = self.encoder.transform(x)
#         z_scaled = self.latent_scaler.transform(z)
#         return z_scaled.flatten()
    
#     def decode(self, z):
#         """Decode from latent space"""
#         z = z.reshape(1, -1)
#         z_unscaled = self.latent_scaler.inverse_transform(z)
#         x_reconstructed = self.encoder.inverse_transform(z_unscaled)
#         return x_reconstructed.flatten()
    
#     def cchvae_loss(self, z_cf, x_orig, desired_class):
#         """CCHVAE loss function"""
#         # Decode counterfactual
#         x_cf = self.decode(z_cf)
        
#         # Clip to reasonable bounds
#         x_cf = np.clip(x_cf, x_orig - 5, x_orig + 5)
        
#         # Reconstruction loss (stay close to data manifold)
#         x_orig_reconstructed = self.decode(self.encode(x_orig))
#         reconstruction_loss = np.sum((x_cf - x_orig_reconstructed) ** 2)
        
#         # Prediction loss
#         try:
#             pred_proba = self.model.predict_proba(x_cf.reshape(1, -1))[0]
#             prediction_loss = (1.0 - pred_proba[desired_class]) ** 2
#         except:
#             prediction_loss = 1.0
        
#         # Proximity loss in latent space
#         z_orig = self.encode(x_orig)
#         latent_proximity_loss = np.sum((z_cf - z_orig) ** 2)
        
#         return (self.prediction_weight * prediction_loss + 
#                 self.reconstruction_weight * reconstruction_loss + 
#                 0.1 * latent_proximity_loss)
    
#     def generate_counterfactual(self, x_orig, desired_class=None):
#         """Generate CCHVAE counterfactual explanation"""
#         x_orig = np.array(x_orig).flatten()
        
#         # Get original prediction
#         orig_pred = self.model.predict(x_orig.reshape(1, -1))[0]
        
#         # Set desired class (opposite of original)
#         if desired_class is None:
#             desired_class = 1 - orig_pred
        
#         # Encode original to latent space
#         z_orig = self.encode(x_orig)
        
#         # Optimize in latent space
#         best_result = None
#         best_loss = float('inf')
        
#         for attempt in range(5):
#             # Initialize with small perturbation in latent space
#             z_cf = z_orig + np.random.normal(0, 0.1, size=z_orig.shape)
            
#             # Set bounds in latent space
#             bounds = [(z_orig[i] - 2, z_orig[i] + 2) for i in range(len(z_orig))]
            
#             try:
#                 result = scipy.optimize.minimize(
#                     fun=lambda z: self.cchvae_loss(z, x_orig, desired_class),
#                     x0=z_cf,
#                     method='L-BFGS-B',
#                     bounds=bounds,
#                     options={'maxiter': 500}
#                 )
                
#                 if result.fun < best_loss:
#                     best_loss = result.fun
#                     best_result = result
#             except:
#                 continue
        
#         if best_result is None:
#             # If optimization failed, return original
#             x_cf_final = x_orig
#         else:
#             # Decode the optimized latent representation
#             x_cf_final = self.decode(best_result.x)
        
#         # Check if counterfactual is valid
#         try:
#             cf_pred = self.model.predict(x_cf_final.reshape(1, -1))[0]
#         except:
#             cf_pred = orig_pred
        
#         # Create changes dictionary
#         changes = {}
#         for i, feature in enumerate(self.feature_names):
#             if abs(x_cf_final[i] - x_orig[i]) > 1e-3:
#                 changes[feature] = {
#                     'original': float(x_orig[i]),
#                     'counterfactual': float(x_cf_final[i]),
#                     'change': float(x_cf_final[i] - x_orig[i])
#                 }
        
#         return {
#             'method': 'CCHVAE',
#             'counterfactual': x_cf_final,
#             'changes': changes,
#             'original_prediction': int(orig_pred),
#             'counterfactual_prediction': int(cf_pred),
#             'valid': cf_pred != orig_pred,
#             'distance': float(np.linalg.norm(x_cf_final - x_orig))
#         }

# model = model_manager.get_model(
#     dataset_name="heart",
#     model_type="mlp"
# )
# heart_config = DATASETS["heart"]
# feature_names = list(heart_config["feature_types"].keys())

# X_train = model_manager.get_X_train(
#     dataset_name="heart"
# )

# # Test CCHVAE method
# cchvae = CCHVAECounterfactual(model, feature_names, X_train, latent_dim=3)
# test_instance = [44,1,2,130,233,0,1,179,1,0.4,2,0,2]
# cchvae_result = cchvae.generate_counterfactual(test_instance)

# print("=== CCHVAE METHOD ===")
# print(f"Original prediction: {cchvae_result['original_prediction']}")
# print(f"Counterfactual prediction: {cchvae_result['counterfactual_prediction']}")
# print(f"Valid counterfactual: {cchvae_result['valid']}")
# print(f"Distance: {cchvae_result['distance']:.3f}")
# print(f"Number of features changed: {len(cchvae_result['changes'])}")

# if cchvae_result['changes']:
#     print("\nTop 3 feature changes:")
#     sorted_changes = sorted(cchvae_result['changes'].items(), key=lambda x: abs(x[1]['change']), reverse=True)
#     for feat, change_info in sorted_changes[:3]:
#         print(f"  {feat}: {change_info['original']:.3f} → {change_info['counterfactual']:.3f} (Δ{change_info['change']:.3f})")



import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import scipy.optimize

class CCHVAE:
    def __init__(self, feature_names, immutable_features, mutable_feature_types, 
                 latent_dim=8, n_components=3, beta=0.5):
        self.feature_names = feature_names
        self.immutable_idx = [feature_names.index(f) for f in immutable_features]
        self.mutable_idx = [i for i in range(len(feature_names)) if i not in self.immutable_idx]
        self.mutable_types = mutable_feature_types
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.beta = beta  # Weight for KL divergence
        self._build_models()
        
    def _build_models(self):
        # Inputs
        input_dim = len(self.feature_names)
        immutable_dim = len(self.immutable_idx)
        mutable_dim = len(self.mutable_idx)
        inputs = Input(shape=(input_dim,))
        x_p = inputs[:, self.immutable_idx]
        x_f = inputs[:, self.mutable_idx]
        
        # Encoder
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        
        # Mixture components (c)
        c_logits = Dense(self.n_components)(x)
        c_probs = tf.nn.softmax(c_logits)
        
        # Latent distribution parameters
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        
        # Sampling layer
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # Decoder
        decoder_input = Concatenate()([z, x_p])
        d = Dense(32, activation='relu')(decoder_input)
        d = Dense(64, activation='relu')(d)
        
        # Heterogeneous outputs
        outputs = []
        self.recon_loss_fn = []
        for i, idx in enumerate(self.mutable_idx):
            if self.mutable_types[i] == 'continuous':
                out_layer = Dense(1, activation='linear', name=f'out_{idx}')(d)
                outputs.append(out_layer)
                self.recon_loss_fn.append(self._gaussian_loss)
            elif self.mutable_types[i] == 'binary':
                out_layer = Dense(1, activation='sigmoid', name=f'out_{idx}')(d)
                outputs.append(out_layer)
                self.recon_loss_fn.append(self._bernoulli_loss)
        
        # Full model
        self.encoder = Model(inputs, [z_mean, z_log_var, c_probs])
        self.decoder = Model([inputs, z], outputs)
        
        # VAE for training
        outputs = self.decoder([inputs, z])
        self.vae = Model(inputs, outputs)
        
        # Loss function
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        self.vae.add_loss(self.beta * kl_loss)
        
        # Compile
        self.vae.compile(optimizer=Adam(0.001))
    
    def _gaussian_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def _bernoulli_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    
    def train(self, X, epochs=100, batch_size=32):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.vae.fit(X_scaled, X_scaled[:, self.mutable_idx], 
                    epochs=epochs, 
                    batch_size=batch_size,
                    verbose=0)
    
    def encode(self, x):
        x_scaled = self.scaler.transform([x])
        z_mean, _, _ = self.encoder.predict(x_scaled, verbose=0)
        return z_mean[0]
    
    def decode(self, z, x_immutable):
        # Create dummy input for decoder
        dummy_input = np.zeros((1, len(self.feature_names)))
        dummy_input[0, self.immutable_idx] = x_immutable
        recon = self.decoder.predict([dummy_input, [z]], verbose=0)
        # Reconstruct full feature vector
        x_rec = np.zeros(len(self.feature_names))
        x_rec[self.immutable_idx] = x_immutable
        for i, idx in enumerate(self.mutable_idx):
            x_rec[idx] = recon[i][0]
        return self.scaler.inverse_transform([x_rec])[0]
    
    def generate_counterfactual(self, x_orig, model, desired_class=1, max_iter=100):
        # Prepare original data
        x_orig = np.array(x_orig)
        x_immutable = x_orig[self.immutable_idx]
        z_orig = self.encode(x_orig)
        
        # Loss function for optimization
        def loss(z_pert):
            x_cf = self.decode(z_pert, x_immutable)
            pred = model.predict_proba([x_cf])[0]
            # Encourage class change and proximity
            class_loss = (1 - pred[desired_class])**2
            prox_loss = np.sum((z_pert - z_orig)**2)
            return class_loss + 0.1 * prox_loss
        
        # Latent space optimization
        bounds = [(z_orig[i]-1, z_orig[i]+1) for i in range(self.latent_dim)]
        result = scipy.optimize.minimize(
            loss, 
            z_orig, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        # Generate counterfactual
        x_cf = self.decode(result.x, x_immutable)
        changes = {}
        for i, name in enumerate(self.feature_names):
            if i in self.mutable_idx and abs(x_cf[i] - x_orig[i]) > 1e-3:
                changes[name] = {
                    'original': x_orig[i],
                    'counterfactual': x_cf[i],
                    'change': x_cf[i] - x_orig[i]
                }
        
        return {
            'counterfactual': x_cf.tolist(),
            'changes': changes,
            'original_prediction': model.predict([x_orig])[0],
            'counterfactual_prediction': model.predict([x_cf])[0],
            'valid': model.predict([x_cf])[0] == desired_class,
            'latent_distance': np.linalg.norm(result.x - z_orig)
        }
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.config import DATASETS, ML_MODELS
from backend.app.model_manager import ModelManager
# Example usage:
if __name__ == "__main__":

    model_manager = ModelManager(datasets_config=DATASETS)

    input_data = {
        "Pregnancies": 3,
        "Glucose": 110,
        "BloodPressure": 92,
        "SkinThickness": 0,
        "Insulin": 2,
        "BMI": 34,
        "DiabetesPedigreeFunction": 0.191,
        "Age": 40,
    }

    feature_names = list(DATASETS["diabetes"]["feature_types"].keys())
    input_array = np.array([input_data[feature] for feature in feature_names])
    immutable = ['Age']  # Protected features
    mutable_types = ['continuous']*7 # All mutable features are continuous
    
    X_train = model_manager.get_X_train(
        dataset_name="diabetes"
    )
    # Initialize and train C-CHVAE
    cchvae = CCHVAE(feature_names, immutable, mutable_types, latent_dim=5)
    cchvae.train(X_train, epochs=200)
    
    model = model_manager.get_model(
        dataset_name="diabetes",
        model_type="random_forest"
    )

    # Generate counterfactual
    result = cchvae.generate_counterfactual(input_array, model, desired_class=0)
    
    print("C-CHVAE Counterfactual:")
    print(f"Original prediction: {result['original_prediction']}")
    print(f"CF prediction: {result['counterfactual_prediction']}")
    print(f"Valid: {result['valid']}")
    print("Changes:")
    for feat, vals in result['changes'].items():
        print(f"  {feat}: {vals['original']:.2f} → {vals['counterfactual']:.2f} (Δ{vals['change']:.2f})")