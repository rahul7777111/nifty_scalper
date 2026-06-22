"""
Hard Risk Firewall — runs after preset selector, before paper trade creation.
ML/router cannot bypass these checks.
Fail-closed: any uncertainty -> block.
"""

from typing import Any, Optional


class HardRiskFirewall:
    """
    Stateless firewall that checks hard risk rules.
    Called on every decision before paper trade creation.
    
    Fail-closed: any uncertainty (missing data, True kill switch, etc.) -> block.
    This firewall cannot be bypassed by ML/router — it is independent.
    """

    # Spread limits by tier
    SPREAD_LIMIT_TIER_MAX: dict[str, float] = {
        "tight": 0.001,
        "normal": 0.002,
        "wide": 0.004,
    }

    def check(
        self,
        decision: dict[str, Any],
        snapshot: dict[str, Any],
        live_features: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        """
        Evaluate all hard risk blocks.
        
        Parameters
        ----------
        decision : dict
            Router+preset merged decision dict containing candidate selection,
            preset name, thresholds, and limits.
        snapshot : dict
            Live snapshot containing system state (real_trading_enabled,
            broker_place_order_allowed, kill_switch_active, daily_pnl, etc.).
        live_features : dict
            Aligned live features including bid/ask, spread_pct, feature_coverage,
            stale_quote, etc.
        mode : str
            "shadow" | "paper" | "live"
            
        Returns
        -------
        dict with keys:
            hard_risk_allowed : bool
            hard_block_reason : str | None
            final_action : "TRADE" | "BLOCK"
        """
        allowed, block_reason = self._check_hard_blocks(
            decision, snapshot, live_features, mode
        )
        
        return {
            "hard_risk_allowed": allowed,
            "hard_block_reason": block_reason,
            "final_action": "TRADE" if allowed else "BLOCK",
        }

    def _check_hard_blocks(
        self,
        decision: dict[str, Any],
        snapshot: dict[str, Any],
        live_features: dict[str, Any],
        mode: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Returns (allowed, block_reason). (False, reason) = blocked.
        
        Implements 15 hard blocks:
        1. Real trading mode — block
        2. real_trading_enabled must be False in shadow/paper
        3. broker_place_order_allowed must be False in shadow/paper
        4. Kill switch active — block
        5. Daily loss limit breached — block
        6. Max open positions breached
        7. Max trades per day breached
        8. Feature coverage too low
        9. Missing bid/ask in paper mode
        10. Stale quote — block
        11. Spread too wide for spread_limit_tier
        12. No candidate selected
        13. Probability below effective threshold
        14. Preset is BLOCK — block regardless
        15. AGGRESSIVE_SHADOW_ONLY in paper — block
        """
        # -----------------------------------------------------------------
        # Block 1: Real trading mode — block
        # -----------------------------------------------------------------
        if mode == "live":
            return False, "Dynamic presets not allowed in live mode"

        # -----------------------------------------------------------------
        # Block 2: real_trading_enabled must be False in shadow/paper
        # -----------------------------------------------------------------
        if snapshot.get("real_trading_enabled") is True:
            return False, "real_trading_enabled is True — hard block"

        # -----------------------------------------------------------------
        # Block 3: broker_place_order_allowed must be False in shadow/paper
        # -----------------------------------------------------------------
        if snapshot.get("broker_place_order_allowed") is True:
            return False, "broker_place_order_allowed is True — hard block"

        # -----------------------------------------------------------------
        # Block 4: Kill switch active — block
        # -----------------------------------------------------------------
        if snapshot.get("kill_switch_active") is True:
            return False, "Kill switch active"

        # -----------------------------------------------------------------
        # Block 5: Daily loss limit breached — block
        # -----------------------------------------------------------------
        daily_pnl = snapshot.get("daily_pnl", 0.0)
        max_daily_loss = snapshot.get("max_daily_loss_hard_cap", -10000.0)
        if daily_pnl <= max_daily_loss:
            return False, f"Daily loss limit breached: {daily_pnl} <= {max_daily_loss}"

        # -----------------------------------------------------------------
        # Block 6: Max open positions breached
        # -----------------------------------------------------------------
        open_positions = snapshot.get("open_positions_count", 0)
        max_open = decision.get("max_open_positions", 1)
        if open_positions >= max_open:
            return False, f"Max open positions: {open_positions} >= {max_open}"

        # -----------------------------------------------------------------
        # Block 7: Max trades per day breached
        # -----------------------------------------------------------------
        trades_today = snapshot.get("trades_today", 0)
        max_trades = decision.get("max_trades_per_day", 0)
        if max_trades > 0 and trades_today >= max_trades:
            return False, f"Max trades per day: {trades_today} >= {max_trades}"

        # -----------------------------------------------------------------
        # Block 8: Feature coverage too low (from preset minimum)
        # -----------------------------------------------------------------
        coverage = live_features.get("feature_coverage", 0.0)
        min_coverage = decision.get("feature_coverage_minimum", 0.95)
        if coverage < min_coverage:
            return False, f"Feature coverage {coverage:.3f} < minimum {min_coverage:.3f}"

        # -----------------------------------------------------------------
        # Block 9: Missing bid/ask in paper mode
        # -----------------------------------------------------------------
        if mode == "paper":
            bid = live_features.get("bid")
            ask = live_features.get("ask")
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                return False, "Missing or invalid bid/ask in paper mode"

        # -----------------------------------------------------------------
        # Block 10: Stale quote — block
        # -----------------------------------------------------------------
        if live_features.get("stale_quote") is True:
            return False, "Stale quote"

        # -----------------------------------------------------------------
        # Block 11: Spread too wide for the preset's spread_limit_tier
        # -----------------------------------------------------------------
        spread_pct = live_features.get("spread_pct", 0.0)
        spread_limit_tier = decision.get("spread_limit_tier", "normal")
        max_spread = self.SPREAD_LIMIT_TIER_MAX.get(spread_limit_tier, 0.001)
        if spread_pct > max_spread:
            return False, f"Spread {spread_pct:.4f} > {spread_limit_tier} tier max {max_spread:.4f}"

        # -----------------------------------------------------------------
        # Block 12: Candidate filter failed (no selected candidate)
        # -----------------------------------------------------------------
        selected_id = decision.get("selected_candidate_id")
        if selected_id is None:
            return False, "No candidate selected"

        # -----------------------------------------------------------------
        # Block 13: Probability below effective threshold
        # -----------------------------------------------------------------
        prob = decision.get("probability", 0.0)
        eff_thresh = decision.get("effective_threshold", 0.5)
        if prob <= eff_thresh:
            return False, f"Probability {prob:.4f} <= effective threshold {eff_thresh:.4f}"

        # -----------------------------------------------------------------
        # Block 14: Preset is BLOCK — block regardless
        # -----------------------------------------------------------------
        selected_preset = decision.get("selected_preset", "")
        if selected_preset == "BLOCK":
            preset_reason = decision.get("preset_reason", "no reason")
            return False, f"Preset BLOCK: {preset_reason}"

        # -----------------------------------------------------------------
        # Block 15: AGGRESSIVE_SHADOW_ONLY in paper — block
        # -----------------------------------------------------------------
        if selected_preset == "AGGRESSIVE_SHADOW_ONLY" and mode == "paper":
            return False, "AGGRESSIVE_SHADOW_ONLY not allowed in paper mode"

        # -----------------------------------------------------------------
        # All blocks passed
        # -----------------------------------------------------------------
        return True, None


# ---------------------------------------------------------------------------
# Convenience function for integration with routing pipelines
# ---------------------------------------------------------------------------

def check_hard_risk(
    decision: dict[str, Any],
    snapshot: dict[str, Any],
    live_features: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """
    Convenience wrapper around HardRiskFirewall.check().
    
    Example integration with route_with_presets():
    
        decision = route_with_presets(snapshot, live_features, mode)
        
        # Apply hard risk firewall
        firewall_result = check_hard_risk(decision, snapshot, live_features, mode)
        decision.update(firewall_result)
        
        if not decision["hard_risk_allowed"]:
            return decision  # Blocked, do not create paper trade
        
        # Continue to paper trade creation...
    """
    firewall = HardRiskFirewall()
    return firewall.check(decision, snapshot, live_features, mode)