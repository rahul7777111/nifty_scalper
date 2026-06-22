"""Feature Stabilizer and Online Standardization Engine.

Implements Welford's algorithm for rolling and exponentially decaying online Z-score
standardization, ensuring feature stability under severe NIFTY regime drift.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squares of differences from the current mean
    var: float = 1.0

class WelfordStabilizer:
    """Exponentially decaying and standard rolling Welford algorithm for online Z-score tracking."""
    def __init__(self, decay: float = 0.999, min_samples: int = 20):
        self.decay = decay
        self.min_samples = min_samples
        self.count = 0
        self.mean = 0.0
        self.var = 1.0
        self.M2 = 0.0
        
    def update(self, x: float) -> float:
        """Update running metrics and return standardized Z-score.
        
        Args:
            x: Raw feature value.
            
        Returns:
            Standardized value (Z-score).
        """
        if not np.isfinite(x):
            return 0.0
            
        self.count += 1
        
        if self.count == 1:
            self.mean = x
            self.var = 1.0
            self.M2 = 0.0
            return 0.0
            
        # Exponentially decaying variant for trading feature adaptation
        diff = x - self.mean
        # Adjust adaptation speed based on number of samples
        alpha = 1.0 - self.decay if self.count > self.min_samples else 1.0 / self.count
        
        self.mean += alpha * diff
        self.M2 = (1.0 - alpha) * (self.M2 + alpha * diff ** 2)
        
        # M2 acts as variance tracking
        self.var = self.M2
        std = np.sqrt(self.var) if self.var > 1e-8 else 1.0
        
        return (x - self.mean) / std

    def transform(self, x: float) -> float:
        """Standardize a value without updating running statistics."""
        if not np.isfinite(x):
            return 0.0
        std = np.sqrt(self.var) if self.var > 1e-8 else 1.0
        return (x - self.mean) / std


class OnlineFeaturePipeline:
    """Manages online standardization for a multi-feature matrix."""
    def __init__(self, feature_names: List[str], decay: float = 0.999):
        self.feature_names = feature_names
        self.stabilizers = {name: WelfordStabilizer(decay=decay) for name in feature_names}
        
    def process_row(self, row: Dict[str, float]) -> Dict[str, float]:
        """Standardizes a single real-time ticker stream feature dictionary."""
        processed = {}
        for name in self.feature_names:
            val = row.get(name, 0.0)
            processed[name] = self.stabilizers[name].update(val)
        return processed

    def process_matrix(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Process and standardize a batch matrix."""
        n_samples, n_features = X.shape
        standardized = np.zeros_like(X)
        for j in range(n_features):
            name = feature_names[j]
            stabilizer = self.stabilizers.setdefault(name, WelfordStabilizer())
            for i in range(n_samples):
                standardized[i, j] = stabilizer.update(X[i, j])
        return standardized
