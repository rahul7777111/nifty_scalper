"""Small helper module for GPT diagnostics.

This exists mainly so you can do:

    import gpt
    gpt.health_check()

without wiring the full strategy/UI.
"""

from __future__ import annotations

from typing import Dict, Optional

# Support both:
# - package-style imports: `from src import gpt`
# - script-style imports (UI adds src/ to sys.path): `import gpt`
try:
    from .gpt_advisor import health_check as _health_check  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    from gpt_advisor import health_check as _health_check


def health_check(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_sec: float = 45.0,
    print_result: bool = True,
) -> Dict[str, object]:
    """Run a minimal request against the configured GPT endpoint.

    Never prints the raw API key.
    """

    result = _health_check(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_sec=timeout_sec,
    )

    if print_result:
        try:
            ok = bool(result.get("ok"))
            status = result.get("status")
            url = result.get("url")
            chosen_model = result.get("model")
            detail = result.get("detail")
            print(f"[GPT health] ok={ok} status={status} model={chosen_model} url={url} :: {detail}")
        except Exception:
            print(f"[GPT health] {result}")

    return result
