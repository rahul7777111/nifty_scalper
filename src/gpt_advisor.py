from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
import urllib.parse
import re
from datetime import date, datetime
from decimal import Decimal
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, Dict, List, Optional, Tuple


def _normalize_model_name(name: str) -> str:
    """Normalize common user typos in model identifiers.

    OpenAI-compatible model ids use '.' for decimals. Some locales/users may
    type 'gpt-5,0' or 'gpt 4.0 mini'. We normalize those common variants.
    """

    s = str(name or "").strip()
    if not s:
        return s
    try:
        s_l = s.lower()
        if s_l in {"gpt 4", "gpt4", "gpt-4.0", "gpt 4.0"}:
            return "gpt-4"
        if s_l in {"gpt 4.0 mini", "gpt-4.0-mini", "gpt 4o mini", "gpt-4o mini", "gpt4o-mini"}:
            return "gpt-4o-mini"
        s2 = re.sub(r"(\d),(\d)", r"\1.\2", s)
        # Most GPT ids don't use a ".0" suffix (e.g. "gpt-5", not "gpt-5.0").
        s2 = re.sub(r"(gpt-\d+)\.0\b", r"\1", s2)
        return s2
    except Exception:
        return s


def _default_model_for_url(url: str) -> str:
    return "gpt-4o-mini"


def _effective_model_name(model: Optional[str], url: str) -> str:
    normalized = _normalize_model_name(str(model or "").strip())
    if not normalized:
        return _default_model_for_url(url)
    return normalized


def _use_max_completion_tokens(model: str) -> bool:
    """Return True when the model expects max_completion_tokens instead of max_tokens."""

    m = _normalize_model_name(model)
    try:
        return "gpt-5" in (m or "")
    except Exception:
        return False


def _effective_base_url(base_url: Optional[str], api_key: str) -> Optional[str]:
    _ = api_key
    explicit = str(base_url or os.getenv("MSTOCK_GPT_API_BASE_URL") or "").strip()
    if explicit:
        return explicit
    return _default_api_base_url()


def _omit_temperature(model: str) -> bool:
    """Return True when we should omit the temperature parameter.

    Some models (notably GPT-5 family) only support the default temperature and
    reject explicit values like 0.
    """

    try:
        return _use_max_completion_tokens(model)
    except Exception:
        return False


def health_check(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_sec: float = 45.0,
    http_post: Optional["HttpPost"] = None,
) -> Dict[str, object]:
    """Best-effort connectivity/auth check for the configured chat.completions endpoint.

    Returns a dict safe to print/log (never includes the raw api_key).
    """

    key = (api_key if api_key is not None else (os.getenv("MSTOCK_GPT_API_KEY") or os.getenv("OPENAI_API_KEY") or "")).strip()

    url = _chat_completions_url(_effective_base_url(base_url, key))

    # Choose a sensible default model when the user hasn't configured one.
    chosen_model = _effective_model_name(
        model if model is not None else (os.getenv("MSTOCK_GPT_MODEL") or ""),
        url,
    )

    if not key:
        return {
            "ok": False,
            "status": 0,
            "url": url,
            "model": chosen_model,
            "detail": "Missing MSTOCK_GPT_API_KEY/OPENAI_API_KEY",
        }

    token_key = "max_completion_tokens" if _use_max_completion_tokens(chosen_model) else "max_tokens"
    payload: Dict[str, Any] = {
        "model": chosen_model,
        token_key: 8,
        "messages": [
            {"role": "system", "content": "Respond with a single word: OK"},
            {"role": "user", "content": "OK"},
        ],
    }
    if not _omit_temperature(chosen_model):
        payload["temperature"] = 0.0

    headers = _auth_headers_for_url(url, key)

    post = http_post or _urllib_post_json
    status, resp_text = post(url, headers, payload, float(timeout_sec))
    safe_text = _redact_secrets(resp_text or "")

    ok = status == 200
    detail = "OK" if ok else f"HTTP {status}: {safe_text[:200]}"
    if (not ok) and status in {401, 403}:
        detail += " (verify API key permissions and base URL)"

    return {
        "ok": ok,
        "status": int(status),
        "url": url,
        "model": chosen_model,
        "detail": detail,
    }


def _is_github_models_url(url: str) -> bool:
    try:
        host = (urllib.parse.urlparse(str(url or "")).hostname or "").lower()
    except Exception:
        return False
    return host.endswith("models.github.ai")


def _is_azure_inference_url(url: str) -> bool:
    try:
        host = (urllib.parse.urlparse(str(url or "")).hostname or "").lower()
    except Exception:
        return False
    return host.endswith("models.inference.ai.azure.com")


def _auth_headers_for_url(url: str, api_key: str) -> Dict[str, str]:
    """Return auth + required headers for a given endpoint.

    - OpenAI-compatible endpoints: Authorization: Bearer <key>
    - Azure AI Inference endpoint (models.inference.ai.azure.com): uses api-key header.
    """

    k = str(api_key or "").strip()
    if not k:
        return {}

    headers: Dict[str, str] = {}

    if _is_azure_inference_url(url):
        # Azure AI Inference convention.
        headers["api-key"] = k
        # Some gateways also accept Bearer tokens; include for compatibility.
        headers["Authorization"] = f"Bearer {k}"
        return headers

    # Default: OpenAI-compatible Bearer.
    headers["Authorization"] = f"Bearer {k}"

    if _is_github_models_url(url):
        # Matches GitHub Models REST docs/quickstart.
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"

    return headers

def _redact_secrets(text: str) -> str:
    """Best-effort redaction for key/token-like substrings in logs.

    Some upstream APIs echo a partially-redacted key inside error messages.
    This keeps console/UI logs safer without trying to be perfect.
    """

    if not text:
        return text

    t = str(text)

    # Common OpenAI key shapes.
    t = re.sub(r"\bsk-[A-Za-z0-9]{8,}\b", "sk-***", t)
    t = re.sub(r"\bsk-proj-[A-Za-z0-9]{8,}\b", "sk-proj-***", t)

    # GitHub tokens commonly mistaken for OPENAI_API_KEY.
    t = re.sub(r"\bgithub_pat_[A-Za-z0-9_]{8,}\b", "github_pat_***", t)
    t = re.sub(r"\bghp_[A-Za-z0-9]{8,}\b", "ghp_***", t)

    return t


def _chat_completions_url(base_url: Optional[str]) -> str:
    """Return a usable chat.completions endpoint URL.

    Accepts either:
    - a base URL like "https://api.aicredits.in/v1" (we append /chat/completions)
    - a full endpoint URL like "https://api.aicredits.in/v1/chat/completions"
    """

    u = (base_url or os.getenv("MSTOCK_GPT_API_BASE_URL") or _default_api_base_url()).strip()
    if not u:
        u = _default_api_base_url()

    u = u.rstrip("/")
    u_l = u.lower()
    if u_l.endswith("/chat/completions"):
        return u
    return f"{u}/chat/completions"


@dataclass(frozen=True)
class GPTAdvice:
    decision: str  # "TAKE" | "SKIP" | "UNKNOWN"
    reason: str = ""
    confidence: Optional[float] = None
    raw_text: str = ""

    # Optional extra structured keys returned by the model.
    # This allows the strategy/UI to surface richer context (e.g. CE/PE bias,
    # suggested strategy, greeks interpretation) without changing the decision schema.
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def allow(self) -> bool:
        return str(self.decision).strip().upper() == "TAKE"

    @property
    def signals_not_profitable(self) -> bool:
        """Return True when the model explicitly indicates the trade is unprofitable.

        Used by strategy gating to hard-block entries that GPT flags as
        "not profitable" / negative EV, even when confidence is low.
        """

        try:
            rsn = str(self.reason or "").strip().lower()
        except Exception:
            rsn = ""

        # Heuristic text patterns (keep conservative to avoid false blocks).
        if rsn:
            patterns = [
                r"\bnot\s+profitable\b",
                r"\bunprofitable\b",
                r"\bnegative\s+(?:ev|expected\s*value|expectancy)\b",
                r"\bnegative\s+edge\b",
                r"\bno\s+edge\b",
                r"\bno\s+profit\b",
                r"\bexpected\s+loss\b",
                r"\bpoor\s+risk[- ]?reward\b",
                r"\bunfavorable\s+risk[- ]?reward\b",
            ]
            try:
                for p in patterns:
                    if re.search(p, rsn):
                        return True
            except Exception:
                pass

        extras = self.extras if isinstance(self.extras, dict) else {}

        # Structured profitability hints.
        try:
            prof = extras.get("profitability")
            if prof is None:
                prof = extras.get("expected_profitability")
            if prof is None:
                prof = extras.get("profit")
            if isinstance(prof, str):
                p = prof.strip().lower()
                if p in {"not_profitable", "unprofitable", "negative", "loss", "losing"}:
                    return True
        except Exception:
            pass

        # Numeric EV/edge hints.
        for k in ("expected_value", "expected_ev", "ev", "expected_pnl", "expected_profit", "edge", "expected_edge"):
            try:
                v = extras.get(k)
                if v is None:
                    continue
                vf = float(v)
                if vf < 0:
                    return True
            except Exception:
                continue

        # Sign-only hints.
        for k in ("expected_value_sign", "ev_sign", "edge_sign"):
            try:
                v = extras.get(k)
                if v is None:
                    continue
                if isinstance(v, str):
                    vl = v.strip().lower()
                    if vl in {"-1", "neg", "negative", "loss", "losing"}:
                        return True
                else:
                    if float(v) < 0:
                        return True
            except Exception:
                continue

        return False


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _default_api_base_url() -> str:
    return "https://api.aicredits.in/v1"


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    s = str(text)

    # Remove common markdown fences.
    s = s.replace("```json", "```").replace("```JSON", "```")
    if "```" in s:
        # Keep everything inside the first fenced block if present.
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]

    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    return s[start : end + 1]


def _json_default(obj: Any) -> Any:
    """Best-effort JSON encoding for common Python types.

    The GPT advisor request payload is user-constructed and may include
    non-JSON-native values (e.g., datetime/date). We normalize them here
    so the advisor call doesn't crash and hard-block trading.
    """

    if isinstance(obj, (datetime, date)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)

    if isinstance(obj, set):
        return list(obj)

    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)

    # Last-resort: stringify to preserve information without failing serialization.
    return str(obj)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def parse_gpt_advice(text: str) -> GPTAdvice:
    """Parse model output into a structured GPTAdvice.

    Expected JSON format:
        {"decision": "TAKE"|"SKIP", "reason": "...", "confidence": 0.0-1.0}

    Robust to extra text / fenced code blocks.
    """

    raw = str(text or "")
    blob = _extract_first_json_object(raw)
    if not blob:
        return GPTAdvice(decision="UNKNOWN", reason="No JSON found in response", raw_text=raw)

    try:
        obj = json.loads(blob)
    except Exception as exc:
        return GPTAdvice(decision="UNKNOWN", reason=f"Invalid JSON in response: {exc}", raw_text=raw)

    decision = str(obj.get("decision") or obj.get("action") or "UNKNOWN").strip().upper()
    if decision not in {"TAKE", "SKIP", "UNKNOWN"}:
        decision = "UNKNOWN"

    reason = str(obj.get("reason") or obj.get("rationale") or "").strip()

    conf = obj.get("confidence")
    confidence: Optional[float]
    try:
        confidence = float(conf) if conf is not None else None
    except Exception:
        confidence = None

    # Preserve any non-core fields as extras.
    extras: Dict[str, Any] = {}
    try:
        for k, v in obj.items():
            if k in {"decision", "action", "reason", "rationale", "confidence"}:
                continue
            extras[str(k)] = v
    except Exception:
        extras = {}

    return GPTAdvice(decision=decision, reason=reason, confidence=confidence, raw_text=raw, extras=extras)


@dataclass(frozen=True)
class GPTMarketAnalysis:
    ce_pe_bias: str  # "CE" | "PE" | "NEUTRAL" | "UNKNOWN"
    recommended_strategy: str = ""
    # Optional preset request based on market conditions.
    # One of: "aggressive", "conservative", "".
    preset_request: str = ""
    preset_reason: str = ""
    # Optional: For directional trades, how many strike steps away from ATM to pick.
    # Example: if strike_step=50 and directional_strike_offset_steps=2, target is 100 points OTM.
    directional_strike_offset_steps: Optional[int] = None
    greeks_interpretation: str = ""
    option_chain_analysis: str = ""
    reason: str = ""
    confidence: Optional[float] = None
    # Optional per-strategy runtime tuning hints:
    # {"short_strangle":{"distance":55,"theta":-7.2}, "iron_condor":{"distance":45,"wing":70}}
    # Supported keys: distance, wing, ratio, expiry_spread, theta, delta.
    strategy_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_text: str = ""


def parse_gpt_market_analysis(text: str) -> GPTMarketAnalysis:
    """Parse model output for market snapshot analysis.

    Expected JSON format (keys are flexible):
        {
          "ce_pe_bias": "CE"|"PE"|"NEUTRAL",
          "recommended_strategy": "directional"|...,
          "greeks_interpretation": "...",
          "option_chain_analysis": "...",
          "reason": "...",
          "confidence": 0.0-1.0
        }

    Robust to extra text / fenced code blocks.
    """

    raw = str(text or "")
    blob = _extract_first_json_object(raw)
    if not blob:
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason="No JSON found in response", raw_text=raw)

    try:
        obj = json.loads(blob)
    except Exception as exc:
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason=f"Invalid JSON in response: {exc}", raw_text=raw)

    bias = str(
        obj.get("ce_pe_bias")
        or obj.get("bias")
        or obj.get("direction")
        or obj.get("cepe_bias")
        or "UNKNOWN"
    ).strip().upper()
    # Models sometimes respond with market-direction words instead of CE/PE.
    # Map common synonyms into the expected CE/PE/NEUTRAL bucket so the strategy
    # can apply GPT voting weight reliably.
    if bias in {"BULL", "BULLISH", "UP", "UPTREND", "LONG", "BUY"}:
        bias = "CE"
    elif bias in {"BEAR", "BEARISH", "DOWN", "DOWNTREND", "SHORT", "SELL"}:
        bias = "PE"
    elif bias in {"SIDEWAYS", "RANGE", "RANGING", "FLAT"}:
        bias = "NEUTRAL"
    if bias in {"CALL", "C"}:
        bias = "CE"
    elif bias in {"PUT", "P"}:
        bias = "PE"
    if bias not in {"CE", "PE", "NEUTRAL", "UNKNOWN"}:
        bias = "UNKNOWN"

    rec = str(obj.get("recommended_strategy") or obj.get("strategy") or "").strip()
    preset_raw = str(
        obj.get("preset_request")
        or obj.get("preset")
        or obj.get("suggested_preset")
        or ""
    ).strip().lower()
    if preset_raw in {"aggressive", "aggresive"}:
        preset_raw = "aggressive"
    elif preset_raw in {"conservative", "safe", "defensive"}:
        preset_raw = "conservative"
    else:
        preset_raw = ""
    preset_reason = str(
        obj.get("preset_reason")
        or obj.get("preset_change_reason")
        or obj.get("preset_rationale")
        or ""
    ).strip()
    # Optional strike guidance for directional trades.
    raw_steps = (
        obj.get("directional_strike_offset_steps")
        or obj.get("strike_offset_steps")
        or obj.get("directional_otm_steps")
        or obj.get("otm_steps")
    )
    directional_steps: Optional[int]
    try:
        if raw_steps is None or (isinstance(raw_steps, str) and not str(raw_steps).strip()):
            directional_steps = None
        else:
            directional_steps = int(float(raw_steps))
    except Exception:
        directional_steps = None
    if directional_steps is not None:
        # Keep it sane; negative doesn't make sense for this field.
        directional_steps = max(0, min(int(directional_steps), 20))

    greeks_txt = str(obj.get("greeks_interpretation") or obj.get("greeks") or "").strip()
    chain_txt = str(obj.get("option_chain_analysis") or obj.get("option_chain") or obj.get("chain") or "").strip()
    reason = str(obj.get("reason") or obj.get("rationale") or "").strip()

    conf = obj.get("confidence")
    confidence: Optional[float]
    try:
        confidence = float(conf) if conf is not None else None
    except Exception:
        confidence = None

    # Optional per-strategy runtime tuning payload.
    strategy_parameters: Dict[str, Dict[str, float]] = {}
    try:
        raw_params = (
            obj.get("strategy_parameters")
            or obj.get("strategy_params")
            or obj.get("parameter_overrides")
            or {}
        )
        if isinstance(raw_params, dict):
            for k, v in raw_params.items():
                if not isinstance(v, dict):
                    continue
                key = str(k or "").strip()
                if not key:
                    continue
                out_row: Dict[str, float] = {}
                for pkey in ("distance", "wing", "ratio", "expiry_spread", "theta", "delta"):
                    try:
                        pv = v.get(pkey)
                        if pv is None:
                            continue
                        out_row[pkey] = float(pv)
                    except Exception:
                        continue
                if out_row:
                    strategy_parameters[key] = out_row
    except Exception:
        strategy_parameters = {}

    return GPTMarketAnalysis(
        ce_pe_bias=bias,
        recommended_strategy=rec,
        preset_request=preset_raw,
        preset_reason=preset_reason,
        directional_strike_offset_steps=directional_steps,
        greeks_interpretation=greeks_txt,
        option_chain_analysis=chain_txt,
        reason=reason,
        confidence=confidence,
        strategy_parameters=strategy_parameters,
        raw_text=raw,
    )


def analyze_market(
    *,
    snapshot: Dict[str, Any],
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    timeout_sec: float = 45.0,
    temperature: float = 0.0,
    max_tokens: int = 450,
    http_post: Optional[HttpPost] = None,
) -> GPTMarketAnalysis:
    """Call OpenAI Chat Completions API for market snapshot analysis.

    Returns GPTMarketAnalysis with robust parsing. If the API call fails,
    returns ce_pe_bias=UNKNOWN.
    """

    if not api_key:
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason="Missing API key")

    url = _chat_completions_url(_effective_base_url(base_url, api_key))

    system = (
        "You are an options trading assistant. "
        "You MUST respond with ONLY valid JSON (no markdown). "
        "Given the market snapshot, provide: "
        "(1) ce_pe_bias as CE|PE|NEUTRAL, "
        "(2) recommended_strategy as one of the provided allowed_strategies (pick EXACTLY from the list; do not invent new names), "
        "(2a) Never return aliases/umbrella labels like directional, short_premium, long_premium, short_volatility, or long_volatility; always return an exact allowed strategy key, "
        "(2b) If you think a directional trade is best, you MUST choose an explicit option-type strategy such as long_call, long_put, short_call, or short_put (avoid returning 'directional'), "
        "(2c) In range-bound conditions with chain_available=true, prefer neutral multi-leg structures (iron_condor/iron_fly/iron_butterfly/short_strangle/short_straddle) over directional picks when justified, "
        "(3) directional_strike_offset_steps as an integer >=0 when a directional option-type strategy is plausible (0=ATM, 1=1 step OTM, etc), "
        "(3b) optional strategy_parameters object for runtime tuning where each key is a strategy and values can include distance, wing, ratio, expiry_spread, theta, delta, "
        "(3c) for strike selection, use live greek context (especially theta, delta, IV and liquidity) and the current preset/risk posture; prefer JSON fields over prose, "
        "(4) greeks_interpretation (short, practical), "
        "(5) option_chain_analysis (short, practical, mention theta decay context), "
        "(6) optional preset_request as aggressive|conservative when a risk-profile switch is advisable for current market, with preset_reason, "
        "(7) reason and confidence (0-1). "
        "Use snapshot.patterns and snapshot.pattern_groups (bullish/bearish/neutral candlestick detections) as first-class context alongside indicators/greeks. "
        "If candlestick groups conflict strongly with indicators/greeks, reduce confidence and bias toward NEUTRAL. "
        "If information is missing/ambiguous, choose NEUTRAL and be conservative. "
        "Output schema: {\"ce_pe_bias\":string,\"recommended_strategy\":string,\"preset_request\":string,\"preset_reason\":string,\"directional_strike_offset_steps\":number,\"strategy_parameters\":object,\"greeks_interpretation\":string,\"option_chain_analysis\":string,\"reason\":string,\"confidence\":number}"
    )

    user = {
        "ts": int(time.time()),
        "snapshot": snapshot,
    }

    model_name = _effective_model_name(model, url)
    token_key = "max_completion_tokens" if _use_max_completion_tokens(model_name) else "max_tokens"
    payload: Dict[str, Any] = {
        "model": model_name,
        token_key: int(max_tokens),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _json_dumps(user)},
        ],
    }
    if not _omit_temperature(model_name):
        payload["temperature"] = float(temperature)

    headers = _auth_headers_for_url(url, api_key)

    post = http_post or _urllib_post_json
    status, resp_text = post(url, headers, payload, float(timeout_sec))

    if _bool_env("MSTOCK_GPT_DEBUG", False):
        print(f"[GPT] market POST {url} status={status} bytes={len(resp_text or '')}")

    if status != 200:
        safe_text = _redact_secrets(resp_text or "")
        hint = ""
        if status in {401, 403}:
            hint = " (verify API key permissions and base URL)"
        return GPTMarketAnalysis(
            ce_pe_bias="UNKNOWN",
            reason=f"HTTP {status}: {safe_text[:200]}{hint}",
            raw_text=safe_text,
        )

    try:
        data = json.loads(resp_text)
    except Exception as exc:
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason=f"Bad JSON from API: {exc}", raw_text=resp_text)

    content = None
    try:
        choices = data.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content")
    except Exception:
        content = None

    if not content:
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason="Empty model response", raw_text=resp_text)

    return parse_gpt_market_analysis(str(content))


HttpPost = Callable[[str, Dict[str, str], Dict[str, Any], float], Tuple[int, str]]


def _urllib_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_sec: float) -> Tuple[int, str]:
    body = _json_dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "OpenAI/Python 1.x")

    try:
        with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            data = resp.read().decode("utf-8", errors="replace")
            return status, data
    except urllib.error.HTTPError as e:
        try:
            data = e.read().decode("utf-8", errors="replace")
        except Exception:
            data = str(e)
        return int(getattr(e, "code", 500) or 500), data
    except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
        reason = getattr(e, "reason", e)
        reason_text = str(reason or e).strip().lower()
        if "timed out" in reason_text or isinstance(reason, (TimeoutError, socket.timeout)):
            timeout_text = f"Request timed out after {float(timeout_sec):g}s"
            return 504, timeout_text
        return 0, str(e)
    except Exception as e:
        return 0, str(e)


def advise_trade(
    *,
    proposal: Dict[str, Any],
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    timeout_sec: float = 45.0,
    temperature: float = 0.0,
    max_tokens: int = 300,
    http_post: Optional[HttpPost] = None,
) -> GPTAdvice:
    """Call OpenAI Chat Completions API to approve/deny a proposed trade.

    This is intentionally conservative:
    - If the API call fails or output can't be parsed, returns decision=UNKNOWN.
    - Use your strategy's risk controls; do not rely on GPT alone.

    Environment variables supported (optional):
    - MSTOCK_GPT_DEBUG=1 : prints request/response metadata (never prints api_key)
    """

    if not api_key:
        return GPTAdvice(decision="UNKNOWN", reason="Missing API key")

    url = _chat_completions_url(_effective_base_url(base_url, api_key))

    system = (
        str(system_prompt).strip()
        if system_prompt is not None and str(system_prompt).strip()
        else (
            "You are a trading risk manager for an automated strategy. "
            "You MUST respond with ONLY valid JSON (no markdown). "
            "Decide whether to TAKE or SKIP the proposed trade, given the context. "
            "Be conservative: if information is missing or ambiguous, SKIP. "
            "Output schema: {\"decision\":\"TAKE\"|\"SKIP\",\"reason\":string,\"confidence\":number}. "
            "When relevant, include optional keys like profitability=profitable|not_profitable|unknown and expected_value_sign=-1|0|1. "
            "You MAY include extra JSON keys when helpful (e.g. side/qty/target)."
        )
    )

    user = {
        "ts": int(time.time()),
        "proposal": proposal,
    }

    model_name = _effective_model_name(model, url)
    token_key = "max_completion_tokens" if _use_max_completion_tokens(model_name) else "max_tokens"
    payload: Dict[str, Any] = {
        "model": model_name,
        token_key: int(max_tokens),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _json_dumps(user)},
        ],
    }
    if not _omit_temperature(model_name):
        payload["temperature"] = float(temperature)

    headers = _auth_headers_for_url(url, api_key)

    post = http_post or _urllib_post_json
    status, resp_text = post(url, headers, payload, float(timeout_sec))

    if _bool_env("MSTOCK_GPT_DEBUG", False):
        print(f"[GPT] POST {url} status={status} bytes={len(resp_text or '')}")

    if status != 200:
        safe_text = _redact_secrets(resp_text or "")
        hint = ""
        if status in {401, 403}:
            hint = " (verify API key permissions and base URL)"
        return GPTAdvice(decision="UNKNOWN", reason=f"HTTP {status}: {safe_text[:200]}{hint}", raw_text=safe_text)

    try:
        data = json.loads(resp_text)
    except Exception as exc:
        return GPTAdvice(decision="UNKNOWN", reason=f"Bad JSON from API: {exc}", raw_text=resp_text)

    # OpenAI chat.completions format
    content = None
    try:
        choices = data.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content")
    except Exception:
        content = None

    if not content:
        return GPTAdvice(decision="UNKNOWN", reason="Empty model response", raw_text=resp_text)

    return parse_gpt_advice(str(content))
