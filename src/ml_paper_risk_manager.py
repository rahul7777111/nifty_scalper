from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List


@dataclass
class MLPaperRiskManager:
    max_trades_per_day: int = 5
    cooldown_minutes: int = 15
    max_open_paper_positions: int = 1
    max_daily_paper_loss: float = 5000.0
    max_consecutive_paper_losses: int = 3
    min_option_price: float = 5.0
    max_bid_ask_spread_pct: float = 5.0
    avoid_first_minutes: int = 5
    avoid_last_minutes: int = 5
    allow_ce: bool = True
    allow_pe: bool = True
    trade_timestamps: List[datetime] = field(default_factory=list)
    open_positions: int = 0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0

    def evaluate(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        ts = snapshot.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        option_type = str(snapshot.get("option_type") or "").upper()
        ltp = float(snapshot.get("option_ltp") or snapshot.get("ltp") or 0.0)
        spread_pct = float(snapshot.get("bid_ask_spread_pct") or 0.0) * (100.0 if float(snapshot.get("bid_ask_spread_pct") or 0.0) <= 1.0 else 1.0)
        if option_type == "CE" and not self.allow_ce:
            return {"allowed": False, "blocking_reason": "ce_disabled", "current_limits_state": self._state()}
        if option_type == "PE" and not self.allow_pe:
            return {"allowed": False, "blocking_reason": "pe_disabled", "current_limits_state": self._state()}
        if ltp < self.min_option_price:
            return {"allowed": False, "blocking_reason": "min_option_price", "current_limits_state": self._state()}
        if spread_pct > self.max_bid_ask_spread_pct:
            return {"allowed": False, "blocking_reason": "max_bid_ask_spread_pct", "current_limits_state": self._state()}
        if self.open_positions >= self.max_open_paper_positions:
            return {"allowed": False, "blocking_reason": "max_open_paper_positions", "current_limits_state": self._state()}
        if self.daily_pnl <= -abs(self.max_daily_paper_loss):
            return {"allowed": False, "blocking_reason": "max_daily_paper_loss", "current_limits_state": self._state()}
        if self.consecutive_losses >= self.max_consecutive_paper_losses:
            return {"allowed": False, "blocking_reason": "max_consecutive_paper_losses", "current_limits_state": self._state()}
        if ts is not None:
            today = ts.date()
            same_day = [item for item in self.trade_timestamps if item.date() == today]
            if len(same_day) >= self.max_trades_per_day:
                return {"allowed": False, "blocking_reason": "max_daily_trades", "current_limits_state": self._state()}
            if same_day and ts - max(same_day) < timedelta(minutes=self.cooldown_minutes):
                return {"allowed": False, "blocking_reason": "cooldown", "current_limits_state": self._state()}
            minutes = ts.hour * 60 + ts.minute
            if minutes < 9 * 60 + 15 + self.avoid_first_minutes:
                return {"allowed": False, "blocking_reason": "avoid_opening_minutes", "current_limits_state": self._state()}
            if minutes > 15 * 60 + 30 - self.avoid_last_minutes:
                return {"allowed": False, "blocking_reason": "avoid_closing_minutes", "current_limits_state": self._state()}
        return {"allowed": True, "blocking_reason": "", "current_limits_state": self._state()}

    def record_entry(self, timestamp: datetime) -> None:
        self.trade_timestamps.append(timestamp)
        self.open_positions += 1

    def record_exit(self, pnl: float) -> None:
        self.open_positions = max(0, self.open_positions - 1)
        self.daily_pnl += float(pnl)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def _state(self) -> Dict[str, Any]:
        return {
            "trades_today": len(self.trade_timestamps),
            "open_positions": self.open_positions,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
        }
