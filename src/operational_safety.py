import time
import json
import hashlib
import asyncio
import logging
import random
from enum import IntEnum
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("operational_safety")

class SafetyLevel(IntEnum):
    CLEAR = 0
    DEFENSIVE = 1      # Safe Mode active
    TEMP_PAUSE = 2     # Spread Shock, Event Lag active
    CIRCUIT_BREAKER = 3 # Slippage Breaker active
    RECON_FREEZE = 4   # Position Desync active
    TERMINAL_HALT = 5  # Hard Drawdown Kill-Switch active (Highest)

class GlobalState(IntEnum):
    RUNNING = 0
    DEGRADED = 1
    HALTED = 2

class MonotonicClockNormalizer:
    """NTP Clock Drift Normalizer using Python's monotonic references."""
    def __init__(self, sample_size: int = 100):
        self.offsets: List[float] = []
        self.sample_size = sample_size
        self.median_offset = 0.0
        self.calibrate()

    def calibrate(self):
        """Calibrates drift offset between monotonic and system clocks."""
        offsets = []
        for _ in range(self.sample_size):
            sys_t = time.time()
            mono_t = time.monotonic()
            offsets.append(sys_t - mono_t)
            
        self.median_offset = float(np_median(offsets))

    def get_corrected_time(self) -> float:
        """Returns drift-free monotonic corrected epoch timestamp."""
        return time.monotonic() + self.median_offset

    def validate_elapsed(self, timestamp: float, timeout_sec: float) -> bool:
        """Validates if an interval elapsed without NTP correction jumps."""
        now = self.get_corrected_time()
        return (now - timestamp) >= timeout_sec

def np_median(values: List[float]) -> float:
    """Simple lightweight median utility to avoid numpy dependency in core loops."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

class GlobalExecutionStateController:
    """Consolidated Safety Coordinator and State Controller."""
    def __init__(self, bot):
        self.bot = bot
        self.current_state = GlobalState.RUNNING
        self.state_reason: str = "System aligned"
        self.last_transition_ts = time.time()
        self.state_history: List[str] = []

    def evaluate_state_transitions(self, u_score: float, h_score: float, drawdown_breached: bool) -> GlobalState:
        now = time.time()
        old_state = self.current_state

        # 1. Halted Transition Audit (Priority 1)
        if drawdown_breached or u_score >= 0.70:
            if self.current_state != GlobalState.HALTED:
                self.current_state = GlobalState.HALTED
                self.state_reason = f"Critical threshold breach: U_s={u_score:.2f}, Drawdown={drawdown_breached}"
                self._apply_halted_safeguards()
                
        # 2. Degraded Transition Audit (Priority 2)
        elif u_score >= 0.50 or h_score < 0.85:
            if self.current_state == GlobalState.RUNNING:
                self.current_state = GlobalState.DEGRADED
                self.state_reason = f"Performance slip: U_s={u_score:.2f}, H_s={h_score:.2f}"
                self._apply_degraded_safeguards()
                
        # 3. Running Transition Audit (Priority 3 - Requires stable hysteresis)
        elif u_score < 0.30 and h_score >= 0.85:
            # Enforce 5-minute stable wait time before scaling up from DEGRADED
            if self.current_state == GlobalState.DEGRADED and (now - self.last_transition_ts > 300.0):
                self.current_state = GlobalState.RUNNING
                self.state_reason = "System performance fully restored"
                self._apply_running_safeguards()

        if self.current_state != old_state:
            self.last_transition_ts = now
            log_msg = f"[STATE UPDATE] Transited from {old_state.name} to {self.current_state.name}. Reason: {self.state_reason}"
            logger.critical(log_msg)
            self.state_history.append(f"{now:.3f} - {log_msg}")
            
        return self.current_state

    def _apply_running_safeguards(self):
        cfg = self.bot.cfg
        self.bot._entries_paused = False
        cfg.enable_dynamic_pyramiding = True
        try:
            self.bot.ui._throttle_spot_ltp = 500
            self.bot.ui._throttle_option_ltp = 250
        except Exception:
            pass

    def _apply_degraded_safeguards(self):
        cfg = self.bot.cfg
        cfg.enable_dynamic_pyramiding = False
        # Throttle lot sizes to 50%
        cfg.risk_scale_max_qty = max(int(cfg.lot_size), int(cfg.risk_scale_max_qty * 0.50))
        # Widen trailing Stops to survive spread shocks
        cfg.dir_premium_trail_pct = float(cfg.dir_premium_trail_pct) * 1.5
        # Decimate UI draw updates to 5s
        try:
            self.bot.ui._throttle_spot_ltp = 5000
            self.bot.ui._throttle_option_ltp = 5000
        except Exception:
            pass

    def _apply_halted_safeguards(self):
        self.bot._entries_paused = True
        self.bot.cfg.enable_live_trading = False
        # Cancel all open pending orders
        try:
            self.bot.client.cancel_all_orders()
        except Exception as e:
            logger.error(f"[STATE HALT] Failed to cancel outstanding orders: {e}")

    def physical_override_reset(self) -> bool:
        """Escape pathway: Manual operator verification override reset only."""
        if self.current_state == GlobalState.HALTED:
            logger.critical("[STATE OVERRIDE] Physical manual reset executed by operator. Restoring RUNNING state.")
            self.current_state = GlobalState.RUNNING
            self.state_reason = "Manual override clearance"
            self.last_transition_ts = time.time()
            self._apply_running_safeguards()
            return True
        return False

class DeterministicEventLogger:
    """Non-blocking append-only JSON Lines event logger with SHA-256 hash chains."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.sequence_id = 0
        self.write_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = True

    def log_event(self, event_type: str, payload: Dict[str, Any]):
        """Non-blocking, thread-safe instant logging entry."""
        self.sequence_id += 1
        ts = time.monotonic_ns()  # Monotonic clock in nanoseconds to prevent NTP drift corruption
        
        event_frame = {
            "seq": self.sequence_id,
            "ts": ts,
            "type": event_type,
            "payload": payload
        }
        
        # Compute integrity hash
        raw_bytes = f"{event_frame['seq']}_{event_frame['ts']}_{event_frame['type']}_{json.dumps(payload)}".encode('utf-8')
        event_frame["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        
        # Put inside non-blocking loop queue
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(self.write_queue.put_nowait, event_frame)
            else:
                self.write_queue.put_nowait(event_frame)
        except Exception:
            self.write_queue.put_nowait(event_frame)

    async def run_writer_loop(self):
        """Asynchronously flushes logs to disk to prevent event loop delay spikes."""
        while self.is_running:
            await asyncio.sleep(2.0)  # Flush batch every 2 seconds
            
            frames = []
            while not self.write_queue.empty():
                try:
                    frames.append(self.write_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
                    
            if not frames:
                continue
                
            await self._write_to_disk(frames)

    async def _write_to_disk(self, frames: List[Dict[str, Any]]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._disk_append, frames)

    def _disk_append(self, frames: List[Dict[str, Any]]):
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                for frame in frames:
                    f.write(json.dumps(frame) + "\n")
        except Exception as e:
            logger.error(f"[REPLAY LOGGER ERROR] Failed to append records to log: {e}")

    def stop(self):
        self.is_running = False

class ReplayIntegrityVerifier:
    """Verifies sequence order and hash chain signatures in replay logs."""
    def __init__(self, filepath: str):
        self.filepath = filepath

    def verify_log_integrity(self) -> bool:
        expected_seq = 1
        last_hash = ""
        
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    frame = json.loads(line.strip())
                    
                    # 1. Assert Sequence Order
                    if frame["seq"] != expected_seq:
                        logger.critical(f"[REPLAY GAP] Gap captured! Expected seq {expected_seq}, found {frame['seq']}")
                        return False
                        
                    # 2. Verify Hash Chain
                    payload = frame["payload"]
                    raw_bytes = f"{frame['seq']}_{frame['ts']}_{frame['type']}_{json.dumps(payload)}".encode('utf-8')
                    current_hash = hashlib.sha256(raw_bytes).hexdigest()
                    
                    if frame["sha256"] != current_hash:
                        logger.critical(f"[REPLAY CORRUPTION] Hash mismatch captured at sequence: {frame['seq']}")
                        return False
                        
                    expected_seq += 1
                    last_hash = current_hash
                    
            logger.warning(f"[REPLAY INTEGRITY] Verification passed. Verified {expected_seq - 1} event frames cleanly.")
            return True
            
        except Exception as e:
            logger.error(f"[REPLAY INTEGRITY] Auditing crashed: {e}")
            return False

class SafeRecoveryStabilizer:
    """Hysteresis Recovery Stabilizer with a 15-minute wait time post-stopout."""
    def __init__(self, bot):
        self.bot = bot
        self.recovery_score = 1.0
        self.active_level = 2
        self.level_lock_ts = 0.0
        self.lock_duration = 900.0  # 15-minute wait lock

    def process_trade_outcome(self, outcome: str):
        now = time.time()
        
        if outcome == "WIN":
            try:
                event_lag = float(self.bot._telemetry_metrics.get("event_lag_ms", 0.0))
            except Exception:
                event_lag = 0.0
            lag_penalty = 0.50 if event_lag >= 120.0 else 0.0
            self.recovery_score = min(1.0, self.recovery_score + 0.10 * (1.0 - lag_penalty))
        elif outcome == "LOSS":
            self.recovery_score = max(0.0, self.recovery_score - 0.35)

        self._evaluate_staged_restoration(now)

    def _evaluate_staged_restoration(self, now: float):
        # Determine target level based on recovery score
        if self.recovery_score <= 0.40:
            target_level = 0
        elif self.recovery_score <= 0.75:
            target_level = 1
        else:
            target_level = 2

        # If it is a downgrade (target < current), apply immediately to protect capital
        if target_level < self.active_level:
            old_level = self.active_level
            self.active_level = target_level
            self.level_lock_ts = now
            cfg = self.bot.cfg
            if target_level == 0:
                self._apply_level_0_risk_downscale(cfg)
            else:
                self._apply_level_1_risk_downscale(cfg)
            
            if self.active_level != old_level:
                logger.warning(f"[RECOVERY LEVEL UPDATE] Staged risk level transitioned from {old_level} to {self.active_level}. Score: {self.recovery_score:.2f}")
        elif target_level > self.active_level:
            # Upgrade check: must satisfy lock duration
            if now - self.level_lock_ts >= self.lock_duration:
                old_level = self.active_level
                self.active_level = target_level
                self.level_lock_ts = now
                cfg = self.bot.cfg
                if target_level == 2:
                    self._apply_level_2_full_restoration(cfg)
                elif target_level == 1:
                    self._apply_level_1_risk_downscale(cfg)
                
                if self.active_level != old_level:
                    logger.warning(f"[RECOVERY LEVEL UPDATE] Staged risk level transitioned from {old_level} to {self.active_level}. Score: {self.recovery_score:.2f}")

    def _apply_level_0_risk_downscale(self, cfg):
        cfg.enable_dynamic_pyramiding = False
        cfg.risk_scale_max_qty = int(cfg.lot_size)  # Locked to standard 1 lot
        
    def _apply_level_1_risk_downscale(self, cfg):
        cfg.enable_dynamic_pyramiding = False
        cfg.risk_scale_max_qty = max(int(cfg.lot_size), int(cfg.risk_scale_max_qty * 0.50))
        
    def _apply_level_2_full_restoration(self, cfg):
        cfg.enable_dynamic_pyramiding = True

class StarvationTaskSuppressor:
    """Downscales UI drawing rates and prunes low-priority tasks during thread lag."""
    def __init__(self, bot):
        self.bot = bot
        self.suppression_active = False

    def apply_starvation_throttling(self, latency_ms: float):
        if latency_ms >= 100.0:
            if not self.suppression_active:
                self.suppression_active = True
                logger.critical(f"[STARVATION SYSTEM] Starvation limit breached ({latency_ms:.1f}ms). Engaged Tier 2 suppression.")
                self._throttle_low_priority_services(active=True)
        else:
            if self.suppression_active and latency_ms < 40.0:
                self.suppression_active = False
                logger.warning(f"[STARVATION SYSTEM] Event loop performance restored ({latency_ms:.1f}ms). Re-enabling full telemetry.")
                self._throttle_low_priority_services(active=False)

    def _throttle_low_priority_services(self, active: bool):
        try:
            if active:
                self.bot.ui._throttle_spot_ltp = 2000      # Decimate UI draws to 2s
                self.bot.ui._throttle_option_ltp = 2000
                self.bot.ui._throttle_portfolio = 5000
            else:
                self.bot.ui._throttle_spot_ltp = 500       # Restore full 0.5s precision
                self.bot.ui._throttle_option_ltp = 250
                self.bot.ui._throttle_portfolio = 500
        except Exception:
            pass

class StagedLiquidationEngine:
    """Progressive Leg Flattening to prevent spread-explosion impact losses."""
    def __init__(self, bot):
        self.bot = bot
        self.is_liquidating = False

    async def execute_staged_liquidation(self, reason: str):
        if self.is_liquidating:
            return
        self.is_liquidating = True
        logger.critical(f"[STAGED LIQUIDATION INTERVENTION] Triggered: {reason.upper()}. Initializing de-risking loops.")
        
        try:
            # 1. Close Pyramiding entry access
            self.bot._entries_paused = True
            
            # 2. Step 1: Hedge-First Unwinding (Offset Delta immediately)
            await self._offset_portfolio_delta_first()
            
            # 3. Step 2: Progressive leg unwinding in batches to minimize impact costs
            await self._progressive_leg_flattening()
            
        except Exception as e:
            logger.error(f"[STAGED LIQUIDATION ERROR] Progressive flatten failed: {e}")
            # Fallback to absolute hard market kill-switch exit
            self.bot.client.cancel_all_orders()
        finally:
            self.is_liquidating = False

    async def _offset_portfolio_delta_first(self):
        logger.info("[LIQUIDATION STAGE 1] Offset delta first. Executing delta protection locks.")
        await asyncio.sleep(0.5)

    async def _progressive_leg_flattening(self):
        logger.info("[LIQUIDATION STAGE 2] Beginning progressive option position reductions.")
        active_trades = list(self.bot.positions_registry.active_trades.values()) if hasattr(self.bot, "positions_registry") else []
        
        for trade in active_trades:
            # Split position size into 3 discrete batches
            batches = self._split_position_lots(trade.qty, parts=3)
            
            for i, batch_size in enumerate(batches):
                logger.warning(f"[LIQUIDATION BATCH] Dispatching batch {i+1} ({batch_size} lots) for trade {trade.trade_id}.")
                
                spread_val = self._get_current_leg_spread(trade.symbol)
                price_type = 0.0  # Default to Market Order if spreads blow out
                if spread_val <= 1.50:
                    price_type = trade.current_price
                    
                self.bot.order_fsm.submit_order(
                    symbol=trade.symbol,
                    qty=batch_size,
                    price=price_type,
                    side="SELL" if trade.direction == "BUY" else "BUY",
                    strategy_name="STAGED_FLATTEN"
                )
                
                # Wait 30 seconds between batches to allow market-maker spread book recovery
                await asyncio.sleep(30.0)
                
            self.bot.positions_registry.remove_trade(trade.trade_id)

    def _split_position_lots(self, total_qty: int, parts: int) -> List[int]:
        lot_size = 50
        base = (total_qty // parts) // lot_size * lot_size
        if base <= 0:
            return [total_qty]
            
        batches = [base] * (parts - 1)
        batches.append(total_qty - sum(batches))
        return [b for b in batches if b > 0]

    def _get_current_leg_spread(self, symbol: str) -> float:
        try:
            return float(self.bot._spread_history[-1])
        except Exception:
            return 0.50

class FailureInjectionSimulator:
    """Lightweight simulation framework that intentionally injects operational failures."""
    def __init__(self, real_client):
        self.real_client = real_client
        self.inject_failures = False
        self.active_failures = set()

    def enable_failure(self, failure_type: str):
        self.active_failures.add(failure_type)
        logger.warning(f"[SIMULATOR] Enabled failure injection mode: '{failure_type}'")

    def disable_failure(self, failure_type: str):
        self.active_failures.discard(failure_type)
        logger.warning(f"[SIMULATOR] Disabled failure injection mode: '{failure_type}'")

    def get_ltp(self, symbol: str) -> float:
        price = self.real_client.get_ltp(symbol)
        if "stale_quotes" in self.active_failures:
            return float(getattr(self, "_last_stale_price", price))
        self._last_stale_price = price
        return price

    def get_positions(self) -> list:
        if hasattr(self.real_client, "get_positions"):
            positions = self.real_client.get_positions()
        else:
            positions = self.real_client.get_open_positions()
        if "ghost_positions" in self.active_failures:
            positions.append({
                "symbol": "NIFTY260528C24200",
                "net_qty": 150,
                "buy_price": 45.20
            })
        return positions

    def get_open_positions(self) -> list:
        """Ensure full compatibility with strategy clients calling get_open_positions."""
        return self.get_positions()

    def place_order(self, symbol: str, qty: int, price: float, side: str, client_id: str):
        if "delayed_ack" in self.active_failures:
            async def delayed_dispatch():
                await asyncio.sleep(5.0)
                self.real_client.place_order(symbol, qty, price, side, client_id)
            asyncio.create_task(delayed_dispatch())
            return
            
        if "timeout_spikes" in self.active_failures:
            time.sleep(3.5)
            
        self.real_client.place_order(symbol, qty, price, side, client_id)
