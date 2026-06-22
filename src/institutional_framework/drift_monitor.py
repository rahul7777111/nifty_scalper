"""Concept Drift, Model Calibration, and Retraining Monitor.

Tracks real-time prediction decay, rolling Brier scores, and feature distribution shift
using Population Stability Index (PSI), triggering automated retraining when thresholds breach.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class DriftReport:
    psi_scores: Dict[str, float]
    mean_brier_score: float
    is_drifted: bool
    trigger_retrain: bool
    stability_score: float = 1.0

class ConceptDriftMonitor:
    """Monitors live data against training statistics to detect concept and feature drift."""
    
    def __init__(
        self,
        reference_features: Dict[str, Tuple[float, float]],  # Feature name to (mean, std)
        psi_threshold: float = 0.25,
        brier_threshold: float = 0.22,
        window_size: int = 500
    ):
        self.reference_features = reference_features
        self.psi_threshold = psi_threshold
        self.brier_threshold = brier_threshold
        self.window_size = window_size
        
        self.live_features: Dict[str, List[float]] = {name: [] for name in reference_features}
        self.live_predictions: List[float] = []
        self.live_realizations: List[int] = []
        
    def record_prediction(self, pred_prob: float, actual_label: Optional[int] = None):
        """Records model prediction probabilities and their eventual real-world labels.
        
        Args:
            pred_prob: The ensemble prediction probability for the positive class.
            actual_label: Realized label (0 or 1) once the triple barrier finishes.
        """
        self.live_predictions.append(pred_prob)
        if actual_label is not None:
            self.live_realizations.append(actual_label)
            
        if len(self.live_predictions) > self.window_size:
            self.live_predictions.pop(0)
        if len(self.live_realizations) > self.window_size:
            self.live_realizations.pop(0)

    def record_features(self, row: Dict[str, float]):
        """Records live feature updates to evaluate distribution shift."""
        for name, val in row.items():
            if name in self.live_features:
                self.live_features[name].append(val)
                if len(self.live_features[name]) > self.window_size:
                    self.live_features[name].pop(0)

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
        """Calculates the Population Stability Index between reference and live windows."""
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
            
        # Set up quantile-based bins using the reference distribution
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(expected, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        # Calculate frequency distribution
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)
        
        # Convert to fractions with Laplace smoothing to prevent division by zero
        expected_pct = (expected_counts + 0.5) / (len(expected) + 0.5 * num_bins)
        actual_pct = (actual_counts + 0.5) / (len(actual) + 0.5 * num_bins)
        
        # Calculate PSI
        psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_value)

    def calculate_brier_score(self) -> float:
        """Calculates the rolling Brier Score (mean squared error of probability predictions)."""
        if not self.live_realizations or len(self.live_predictions) < len(self.live_realizations):
            return 0.0
            
        n_eval = len(self.live_realizations)
        # Match latest predictions to their realizations
        eval_preds = self.live_predictions[-n_eval:]
        
        preds = np.array(eval_preds)
        realizations = np.array(self.live_realizations)
        
        brier = np.mean((preds - realizations) ** 2)
        return float(brier)

    def calculate_model_stability_score(self, current_importances: np.ndarray, baseline_importances: np.ndarray) -> float:
        """Calculates a Model Stability Score (MSS) between 0.0 and 1.0.
        
        Uses Cosine Similarity of feature importances or coefficients to measure stability.
        An MSS near 1.0 represents a stable, non-overfit model, while a low score indicates drift/instability.
        """
        if len(current_importances) == 0 or len(baseline_importances) == 0:
            return 1.0
        
        # Ensure correct shapes
        n1 = np.linalg.norm(current_importances)
        n2 = np.linalg.norm(baseline_importances)
        
        if n1 == 0 or n2 == 0:
            return 1.0
            
        cosine_sim = np.dot(current_importances, baseline_importances) / (n1 * n2)
        # Scale to 0-1 range
        mss = float((cosine_sim + 1.0) / 2.0)
        return mss

    def evaluate_drift(
        self,
        training_data: Dict[str, np.ndarray],
        current_importances: Optional[np.ndarray] = None,
        baseline_importances: Optional[np.ndarray] = None
    ) -> DriftReport:
        """Evaluates whether features or probabilities have drifted significantly.
        
        Args:
            training_data: Dictionary containing training distribution arrays for comparison.
            current_importances: Optional current model feature importances.
            baseline_importances: Optional baseline training feature importances.
        """
        psi_scores = {}
        is_drifted = False
        
        # Calculate PSI for each tracked feature
        for name, live_vals in self.live_features.items():
            if name in training_data and len(live_vals) >= 100:
                psi = self.calculate_psi(training_data[name], np.array(live_vals))
                psi_scores[name] = psi
                if psi > self.psi_threshold:
                    is_drifted = True
            else:
                psi_scores[name] = 0.0
                
        # Calculate Brier score for probability tracking
        brier = self.calculate_brier_score()
        
        # Calculate Model Stability Score
        mss = 1.0
        if current_importances is not None and baseline_importances is not None:
            mss = self.calculate_model_stability_score(current_importances, baseline_importances)
        
        # Retrain if Brier score breaches threshold (accuracy decay), multiple features drift, or MSS degrades
        drifted_count = sum(1 for p in psi_scores.values() if p > self.psi_threshold)
        trigger_retrain = brier > self.brier_threshold or drifted_count >= 3 or mss < 0.70
        
        return DriftReport(
            psi_scores=psi_scores,
            mean_brier_score=brier,
            is_drifted=is_drifted,
            trigger_retrain=trigger_retrain,
            stability_score=mss
        )
