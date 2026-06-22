"""Production-Grade Options ML Trading Pipeline Coordinator.

Coordinates feature calculation, feature stabilization, regime routing, ensemble inference,
capital risk management, low-latency execution, and non-blocking telemetry logging.
"""

from __future__ import annotations

import time
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ml_pipeline import TARGET_FEATURES
from .feature_stabilizer import OnlineFeaturePipeline
from .regime_router import RegimeRouter, MarketRegime
from .ml_signals import DynamicConsensusEnsemble, EnsembleSignal
from .risk_engine import OptionsRiskEngine, PositionSizeReport
from .execution_engine import IndianOptionsExecutionEngine, FillReport
from .telemetry import TelemetryTracker, ExecutionTelemetryRecord
from .drift_monitor import ConceptDriftMonitor
from .database import AsyncTelemetryDatabase

# Compatibility alias retained; the canonical production schema lives in ml_pipeline.TARGET_FEATURES.
TARGET_FEATURES_24 = TARGET_FEATURES

class InstitutionalMLTradingPipeline:
    """End-to-end NIFTY options trading pipeline orchestrator."""
    
    def __init__(
        self,
        db_path: str,
        initial_capital: float = 100000.0,
        base_brokerage: float = 20.0
    ):
        self.db = AsyncTelemetryDatabase(db_path=db_path)
        self.db.start()
        
        self.feature_pipeline = OnlineFeaturePipeline(feature_names=TARGET_FEATURES)
        self.regime_router = RegimeRouter()
        self.ensemble = DynamicConsensusEnsemble()
        self.risk_engine = OptionsRiskEngine(starting_capital=initial_capital)
        self.execution_engine = IndianOptionsExecutionEngine(base_brokerage=base_brokerage)
        self.telemetry = TelemetryTracker(db_writer=self.db)
        
        # Concept Drift tracking setup
        ref_stats = {name: (0.0, 1.0) for name in TARGET_FEATURES}
        self.drift_monitor = ConceptDriftMonitor(reference_features=ref_stats)
        
        self.lot_size = 75  # Standard NSE NIFTY option lot size
        self.is_trading_suspended = False

    def process_tick(
        self,
        raw_features: Dict[str, float],
        spot_price: float,
        opt_bid: float,
        opt_ask: float,
        spot_velocity: float,
        is_news_window: bool = False
    ) -> Optional[Tuple[EnsembleSignal, Optional[PositionSizeReport], Optional[FillReport]]]:
        """Processes a single real-time options market tick through the low-latency pipeline.
        
        Args:
            raw_features: Dictionary containing current values of the 24 target features.
            spot_price: Current NIFTY spot index level.
            opt_bid: Option contract bid price.
            opt_ask: Option contract ask price.
            spot_velocity: Rate of spot price change (momentum).
            is_news_window: Safety flag indicating active news event windows.
        """
        t_start = time.perf_counter()
        
        if self.is_trading_suspended:
            return None
            
        # 1. Feature Stabilization (Online Z-Score)
        stabilized_row = self.feature_pipeline.process_row(raw_features)
        self.drift_monitor.record_features(stabilized_row)
        
        # 2. Regime Detection
        current_regime = self.regime_router.classify_regime(
            adx=raw_features.get("rsi_14", 20.0),  # Proxy for trend strength
            atr=raw_features.get("atr_14", 5.0),
            bb_bandwidth=raw_features.get("bollinger_bandwidth", 0.05),
            current_iv=raw_features.get("implied_volatility_atm", 0.16),
            current_vix=raw_features.get("india_vix", 15.0),
            is_news_window=is_news_window
        )
        
        # 3. Dynamic Ensemble Routing
        weights = self.regime_router.get_routing_weights(current_regime)
        
        # Veto immediately if trading is disabled in current regime (e.g. NEWS_EVENT)
        if not weights.trading_enabled:
            return None
            
        # Convert stabilized features dict to an array matching target shape
        feature_vector = np.array([[stabilized_row.get(name, 0.0) for name in TARGET_FEATURES]])
        t_features = time.perf_counter()
        
        # 4. Platt Scaled Ensemble Prediction
        signal = self.ensemble.predict_consensus(
            X=feature_vector,
            w_lr=weights.w_lr,
            w_rf=weights.w_rf,
            w_xgb=weights.w_xgb,
            regime_name=current_regime.value
        )
        t_signal = time.perf_counter()
        
        # 5. Concept Drift Monitoring
        self.drift_monitor.record_prediction(pred_prob=signal.consensus_prob)
        
        # Gate signals: Exit early if consensus veto is triggered
        if signal.is_vetoed:
            return signal, None, None
            
        # 6. Capital Risk Position Sizing
        risk_report = self.risk_engine.calculate_kelly_size(
            prob=signal.consensus_prob,
            risk_reward=1.5,  # Reward to Risk baseline target
            spot_volatility=raw_features.get("realized_volatility_30m", 0.15),
            target_volatility=0.15,  # 15% Annualized target
            option_premium=(opt_bid + opt_ask) / 2.0,
            lot_size=self.lot_size
        )
        
        if not risk_report.is_allowed:
            return signal, risk_report, None
            
        # 7. Low-Latency Execution Fill Simulation
        t_order_sent = time.perf_counter()
        fill_report = self.execution_engine.simulate_fill(
            target_price=(opt_bid + opt_ask) / 2.0,
            qty=risk_report.recommended_qty,
            bid=opt_bid,
            ask=opt_ask,
            spot_velocity=spot_velocity,
            current_iv=raw_features.get("implied_volatility_atm", 0.16),
            is_buy=(signal.signal_direction > 0)
        )
        t_fill = time.perf_counter()
        
        # 8. Non-Blocking Telemetry & Shortfall Logging
        self.telemetry.measure_execution(
            t_tick_arrival=t_start,
            t_signal_generated=t_signal,
            t_order_sent=t_order_sent,
            t_broker_fill=t_fill,
            spot=spot_price,
            opt_bid=opt_bid,
            opt_ask=opt_ask,
            fill_price=fill_report.fill_price,
            order_type="BUY" if signal.signal_direction > 0 else "SELL"
        )
        
        # Update risk engine active book
        pos_id = f"NIFTY_OPT_{int(time.time())}"
        self.risk_engine.active_positions[pos_id] = {
            "premium": fill_report.fill_price,
            "qty": fill_report.filled_qty,
            "direction": signal.signal_direction
        }
        
        return signal, risk_report, fill_report

    def shutdown(self):
        """Safely stops threads and flushes log queues."""
        self.db.stop()
