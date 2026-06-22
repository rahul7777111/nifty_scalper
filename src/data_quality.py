import logging
from datetime import datetime, timedelta
from typing import List, Tuple
from market_data import Candle

logger = logging.getLogger(__name__)

class DataQualityPipeline:
    """Lightweight Data Quality Pipeline.
    
    Performs streaming-only rolling validations on index and option candle structures.
    Validates timestamps, prices, premiums, and provides repair mechanisms.
    """

    def __init__(self, outlier_atr_mult: float = 3.0):
        self.outlier_atr_mult = outlier_atr_mult
        self.last_clean_confidence = 1.0

    def validate_and_repair(self, candles: List[Candle], timeframe_minutes: int = 1, *, allow_synthetic_repair: bool = True) -> Tuple[List[Candle], float]:
        """Validates incoming candle lists, repairs missing gaps, and returns confidence.
        
        Returns:
            Tuple of [repaired_candles_list, confidence_score] (0.0 to 1.0)
        """
        if not candles:
            return [], 0.0
            
        try:
            # 1. Sort by time just in case
            sorted_candles = sorted(candles, key=lambda c: c.time)
            
            repaired = []
            duplicates = 0
            gaps_repaired = 0
            outliers = 0
            
            # Simple rolling statistics for outlier detection (Close vs rolling mean)
            rolling_closes = []
            
            for i, candle in enumerate(sorted_candles):
                # Outlier detection: price deviation check
                is_outlier = False
                if rolling_closes:
                    avg_close = sum(rolling_closes) / len(rolling_closes)
                    # Reject if price moves more than 10% in a single minute/timeframe (shock protection)
                    if avg_close > 0 and abs(candle.close - avg_close) / avg_close > 0.10:
                        is_outlier = True
                        outliers += 1
                        logger.warning(f"[DATA QUALITY] Outlier detected at {candle.time}: Price {candle.close} vs MA {avg_close:.1f}")
                
                if not is_outlier:
                    rolling_closes.append(candle.close)
                    if len(rolling_closes) > 10:
                        rolling_closes.pop(0)

                # Check for duplicates
                if repaired and repaired[-1].time == candle.time:
                    duplicates += 1
                    continue  # skip duplicate candles
                
                # Check for missing gaps (greater than timeframe interval)
                if repaired and i > 0:
                    prev_time = repaired[-1].time
                    curr_time = candle.time
                    delta = curr_time - prev_time
                    expected_delta = timedelta(minutes=timeframe_minutes)
                    
                    if delta > expected_delta:
                        missing_count = int(delta / expected_delta) - 1
                        if 0 < missing_count <= 5:
                            gaps_repaired += missing_count
                            if allow_synthetic_repair:
                                last_c = repaired[-1]
                                for j in range(1, missing_count + 1):
                                    synth_time = prev_time + (expected_delta * j)
                                    repaired.append(Candle(
                                        time=synth_time,
                                        open=last_c.close,
                                        high=last_c.close,
                                        low=last_c.close,
                                        close=last_c.close,
                                        volume=0.0
                                    ))
                            else:
                                logger.warning(
                                    f"[DATA QUALITY] Gap detected at {candle.time}: missing {missing_count} candles; synthetic repair disabled"
                                )
                
                if not is_outlier:
                    repaired.append(candle)
                    
            # 5. Compute Confidence Score
            total_processed = len(sorted_candles)
            penalties = (duplicates * 0.1) + (gaps_repaired * 0.05) + (outliers * 0.2)
            confidence = max(0.0, min(1.0, 1.0 - (penalties / total_processed if total_processed > 0 else 0)))
            
            self.last_clean_confidence = confidence
            return repaired, confidence
            
        except Exception as e:
            logger.error(f"[DATA QUALITY] Error in validation pipeline: {e}")
            return candles, 0.5

    def validate_option_premiums(self, bid: float, ask: float) -> bool:
        """Sanity checks on active option quote bid-asks."""
        if bid < 0 or ask < 0:
            logger.warning(f"[DATA QUALITY] Negative premium detected: Bid {bid}, Ask {ask}")
            return False
        if ask < bid:
            logger.warning(f"[DATA QUALITY] Ask price less than bid: Bid {bid}, Ask {ask}")
            return False
        return True
