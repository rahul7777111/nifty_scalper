from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_shadow_mode as esm


def test_shadow_evaluation_generates_metrics(tmp_path: Path) -> None:
    log = tmp_path / "ml_shadow_predictions_20260605.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-06-05T10:00:00+05:30", "shadow_decision": "WOULD_ENTER", "option_type": "CE", "realized_return": 0.1}),
                json.dumps({"timestamp": "2026-06-05T10:30:00+05:30", "shadow_decision": "WOULD_ENTER", "option_type": "PE", "realized_return": -0.05}),
            ]
        ),
        encoding="utf-8",
    )
    payload = esm.evaluate_shadow_rows(esm._load_jsonl(log))
    assert payload["number_of_predictions"] == 2
    assert payload["number_of_would_enter_signals"] == 2
