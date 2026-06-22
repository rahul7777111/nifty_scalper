#!/usr/bin/env python3
"""Unit test paper-forward readiness reconciliation."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_forward_engine import reconcile_paper_forward_readiness, PaperForwardDataStatus


def main() -> int:
    errors: list[str] = []

    quality, predict = reconcile_paper_forward_readiness(
        data_quality="WAITING_FOR_CANDLES",
        candle_rows=100,
        option_rows=82,
        spot=24500.5,
        missing_features=[],
        artifact_ok=True,
        candidate_id="test_cand",
    )
    if quality != "DATA_OK":
        errors.append(f"expected DATA_OK got {quality}")
    if not predict:
        errors.append("expected predict_allowed=true")

    status = PaperForwardDataStatus(
        spot=24500.0,
        candle_count=100,
        option_rows=82,
        data_quality_status="WAITING_FOR_CANDLES",
        broker_auth="AUTH_OK",
    )
    if status.data_quality_status != "DATA_OK":
        errors.append(f"PaperForwardDataStatus reconcile failed: {status.data_quality_status}")

    quality2, predict2 = reconcile_paper_forward_readiness(
        data_quality="DATA_OK",
        candle_rows=0,
        option_rows=82,
        spot=24500.5,
        missing_features=[],
        artifact_ok=True,
    )
    if predict2:
        errors.append("predict should be false when candle_rows=0")
    if quality2 == "DATA_OK" and not predict2:
        pass
    elif quality2 != "DATA_OK":
        pass
    else:
        errors.append("candle_rows=0 should not allow prediction")

    if errors:
        print("verify_paper_forward_readiness FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("verify_paper_forward_readiness PASSED")
    print(f"  data_quality={quality} predict_allowed={predict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())