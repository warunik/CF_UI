import numpy as np
from sklearn.neighbors import KernelDensity

class DensityValidator:
    def __init__(self, X_train, threshold=0.1):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        self.kde.fit(X_train)
        self.threshold = np.percentile(
            self.kde.score_samples(X_train), 
            threshold * 100
        )
    
    def score(self, cf):
        return self.kde.score_samples([cf])[0]
    
    def is_plausible(self, cf):
        return self.score(cf) >= self.threshold