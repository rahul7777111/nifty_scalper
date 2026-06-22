from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
import urllib.parse
import re
from pathlib import Path
from datetime import date, datetime
from decimal import Decimal
from dataclasses import dataclass
from dataclasses import field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading


def _decode_env_value(raw: str) -> str:
    v = str(raw or "").strip()
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        v = v[1:-1]
        v = v.replace("\\n", "\n").replace("\\\\", "\\").replace('\\"', '"').replace("\\'", "'")
    return v


def _load_env_file(path: Path, *, override_existing: bool = False) -> Dict[str, str]:
    try:
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return {}

    loaded: Dict[str, str] = {}
    for line in raw.splitlines():
        s = str(line or "").strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        if not key:
            continue
        val = _decode_env_value(v)
        loaded[key] = val
        if override_existing or key not in os.environ:
            os.environ[key] = val
    return loaded


def _hydrate_gpt_env() -> None:
    """Load repo GPT env settings without requiring python-dotenv.

    Strategy/UI paths normally load config first, but direct GPT health checks and
    tests import this module standalone. Missing base URL made valid alternate
    provider keys get sent to OpenAI, causing misleading 401 errors.
    """

    try:
        root = Path(__file__).resolve().parent.parent
    except Exception:
        return

    # `.scalper.env` is what the UI persists. `.env` keeps older/manual setups
    # working. Existing process env values still win.
    _load_env_file(root / ".scalper.env", override_existing=False)
    _load_env_file(root / ".env", override_existing=False)


_hydrate_gpt_env()


def _normalize_model_name(name: str) -> str:
    """Normalize common user typos in model identifiers."""
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
    """Return True when we should omit the temperature parameter."""
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
    """Best-effort connectivity/auth check for the configured chat.completions endpoint."""
    key = (api_key if api_key is not None else (os.getenv("MSTOCK_GPT_API_KEY") or os.getenv("OPENAI_API_KEY") or "")).strip()
    url = _chat_completions_url(_effective_base_url(base_url, key))
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


# UI integration hook
_ui_callback: Optional[Callable] = None


def register_ui_callback(fn: Callable) -> None:
    """Register a UI callback to receive GPT events."""
    global _ui_callback
    try:
        _ui_callback = fn
    except Exception:
        _ui_callback = None


def _notify_ui(event: str, payload: dict) -> None:
    try:
        if _ui_callback:
            try:
                _ui_callback(str(event), dict(payload or {}))
            except Exception:
                pass
    except Exception:
        pass


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
    """Return auth + required headers for a given endpoint."""
    k = str(api_key or "").strip()
    if not k:
        return {}
    headers: Dict[str, str] = {}
    if _is_azure_inference_url(url):
        headers["api-key"] = k
        headers["Authorization"] = f"Bearer {k}"
        return headers
    headers["Authorization"] = f"Bearer {k}"
    if _is_github_models_url(url):
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    return headers


def _redact_secrets(text: str) -> str:
    """Best-effort redaction for key/token-like substrings in logs."""
    if not text:
        return text
    t = str(text)
    t = re.sub(r"\bsk-[A-Za-z0-9]{8,}\b", "sk-***", t)
    t = re.sub(r"\bsk-proj-[A-Za-z0-9]{8,}\b", "sk-proj-***", t)
    t = re.sub(r"\bgithub_pat_[A-Za-z0-9_]{8,}\b", "github_pat_***", t)
    t = re.sub(r"\bghp_[A-Za-z0-9]{8,}\b", "ghp_***", t)
    return t


def _chat_completions_url(base_url: Optional[str]) -> str:
    """Return a usable chat.completions endpoint URL."""
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
    decision: str
    reason: str = ""
    confidence: Optional[float] = None
    raw_text: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def allow(self) -> bool:
        return str(self.decision).strip().upper() == "TAKE"

    @property
    def signals_not_profitable(self) -> bool:
        try:
            rsn = str(self.reason or "").strip().lower()
        except Exception:
            rsn = ""
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
        try:
            prof = extras.get("profitability") or extras.get("expected_profitability") or extras.get("profit")
            if isinstance(prof, str):
                p = prof.strip().lower()
                if p in {"not_profitable", "unprofitable", "negative", "loss", "losing"}:
                    return True
        except Exception:
            pass
        for k in ("expected_value", "expected_ev", "ev", "expected_pnl", "expected_profit", "edge", "expected_edge"):
            try:
                v = extras.get(k)
                if v is None:
                    continue
                if float(v) < 0:
                    return True
            except Exception:
                continue
        for k in ("expected_value_sign", "ev_sign", "edge_sign"):
            try:
                v = extras.get(k)
                if v is None:
                    continue
                if isinstance(v, str):
                    if v.strip().lower() in {"-1", "neg", "negative", "loss", "losing"}:
                        return True
                elif float(v) < 0:
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
    env_url = (os.getenv("MSTOCK_GPT_API_BASE_URL") or "").strip()
    if env_url:
        return env_url
    return "https://api.openai.com/v1"


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    s = str(text)
    s = s.replace("```json", "```").replace("```JSON", "```")
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    return s[start : end + 1]


def _json_default(obj: Any) -> Any:
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
    return str(obj)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def parse_gpt_advice(text: str) -> GPTAdvice:
    """Parse model output into a structured GPTAdvice."""
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
    ce_pe_bias: str
    recommended_strategy: str = ""
    preset_request: str = ""
    preset_reason: str = ""
    directional_strike_offset_steps: Optional[int] = None
    greeks_interpretation: str = ""
    option_chain_analysis: str = ""
    reason: str = ""
    confidence: Optional[float] = None
    strategy_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_text: str = ""


def parse_gpt_market_analysis(text: str) -> GPTMarketAnalysis:
    """Parse model output for market snapshot analysis."""
    raw = str(text or "")
    blob = _extract_first_json_object(raw)
    if not blob:
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason="No JSON found in response", raw_text=raw)
    try:
        obj = json.loads(blob)
    except Exception as exc:
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason=f"Invalid JSON in response: {exc}", raw_text=raw)
    bias = str(obj.get("ce_pe_bias") or obj.get("bias") or obj.get("direction") or obj.get("cepe_bias") or "UNKNOWN").strip().upper()
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
    preset_raw = str(obj.get("preset_request") or obj.get("preset") or obj.get("suggested_preset") or "").strip().lower()
    if preset_raw in {"aggressive", "aggresive"}:
        preset_raw = "aggressive"
    elif preset_raw in {"conservative", "safe", "defensive"}:
        preset_raw = "conservative"
    else:
        preset_raw = ""
    preset_reason = str(obj.get("preset_reason") or obj.get("preset_change_reason") or obj.get("preset_rationale") or "").strip()
    raw_steps = obj.get("directional_strike_offset_steps") or obj.get("strike_offset_steps") or obj.get("directional_otm_steps") or obj.get("otm_steps")
    directional_steps: Optional[int]
    try:
        if raw_steps is None or (isinstance(raw_steps, str) and not str(raw_steps).strip()):
            directional_steps = None
        else:
            directional_steps = int(float(raw_steps))
    except Exception:
        directional_steps = None
    if directional_steps is not None:
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
    strategy_parameters: Dict[str, Dict[str, float]] = {}
    try:
        raw_params = obj.get("strategy_parameters") or obj.get("strategy_params") or obj.get("parameter_overrides") or {}
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


def _is_reasoning_model(model: str) -> bool:
    """Return True if the model is a reasoning model that outputs chain-of-thought."""
    m = str(model or "").lower()
    reasoning_patterns = ["r1", "deepseek-r", "hy3", "qwq", "reasoning", "think", "o1", "o3", "o4"]
    return any(p in m for p in reasoning_patterns)


def _analyze_market_raw(
    *,
    snapshot: Dict[str, Any],
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    timeout_sec: float = 45.0,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    http_post: Optional["HttpPost"] = None,
) -> GPTMarketAnalysis:
    """Call OpenAI Chat Completions API for market snapshot analysis."""
    if not api_key:
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason="Missing API key")
    effective_max_tokens = int(max_tokens)
    if _is_reasoning_model(model):
        # Reasoning models need more tokens because they output chain-of-thought before the final answer
        # tencent/hy3-preview and similar models can require 8000+ tokens
        effective_max_tokens = max(effective_max_tokens, 8192)
        if _bool_env("MSTOCK_GPT_DEBUG", False):
            print(f"[GPT] Reasoning model detected, using max_tokens={effective_max_tokens}")
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
        'Output schema: {"ce_pe_bias":string,"recommended_strategy":string,"preset_request":string,"preset_reason":string,"directional_strike_offset_steps":number,"strategy_parameters":object,"greeks_interpretation":string,"option_chain_analysis":string,"reason":string,"confidence":number}'
    )
    user = {"ts": int(time.time()), "snapshot": snapshot}
    model_name = _effective_model_name(model, url)
    token_key = "max_completion_tokens" if _use_max_completion_tokens(model_name) else "max_tokens"
    payload: Dict[str, Any] = {
        "model": model_name,
        token_key: int(effective_max_tokens),
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
        print(f"[GPT] Using model: {model_name}")
        print(f"[GPT] Payload messages count: {len(payload.get('messages', []))}")
        print(f"[GPT] System prompt length: {len(system)}")
        print(f"[GPT] User snapshot keys: {list(snapshot.keys()) if isinstance(snapshot, dict) else 'N/A'}")
    if status != 200:
        safe_text = _redact_secrets(resp_text or "")
        hint = ""
        if status in {401, 403}:
            hint = " (verify API key permissions and base URL)"
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason=f"HTTP {status}: {safe_text[:200]}{hint}", raw_text=safe_text)
    try:
        data = json.loads(resp_text)
    except Exception as exc:
        return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason=f"Bad JSON from API: {exc}", raw_text=resp_text)
    content = None
    finish_reason = None
    reasoning_text = None
    try:
        choices = data.get("choices") or []
        if choices:
            first_choice = choices[0] or {}
            msg = first_choice.get("message") or {}
            content = msg.get("content")
            reasoning_text = msg.get("reasoning") or msg.get("reasoning_content")
            finish_reason = first_choice.get("finish_reason")
    except Exception:
        content = None
        reasoning_text = None
        finish_reason = None
    if not content and reasoning_text:
        import re as _re
        json_match = _re.search(r'\{[^{}]*"ce_pe_bias"[^{}]*\}', str(reasoning_text))
        if json_match:
            content = json_match.group(0)
            if _bool_env("MSTOCK_GPT_DEBUG", False):
                print("[GPT] Extracted content from reasoning field")
        elif _bool_env("MSTOCK_GPT_DEBUG", False):
            print(f"[GPT] No JSON found in reasoning text (len={len(reasoning_text or '')})")
    if not content:
        if isinstance(data, dict):
            content = data.get("content") or data.get("text") or data.get("output")
        if not content:
            debug_info = f"Empty model response (status={status})"
            if finish_reason:
                debug_info += f", finish_reason={finish_reason}"
                if finish_reason == "length":
                    debug_info += " (max tokens reached - consider increasing max_tokens)"
                elif finish_reason == "content_filter":
                    debug_info += " (content filtered by API)"
            if _bool_env("MSTOCK_GPT_DEBUG", False):
                print(f"[GPT] Empty response. Full API response: {resp_text[:1000]}")
                print(f"[GPT] Response type: {type(resp_text)}, len: {len(resp_text) if resp_text else 0}")
                # Try to parse as JSON to see the structure
                try:
                    data_debug = json.loads(resp_text)
                    print(f"[GPT] Response JSON keys: {list(data_debug.keys()) if isinstance(data_debug, dict) else 'not dict'}")
                    if isinstance(data_debug, dict) and 'choices' in data_debug:
                        choices = data_debug.get('choices', [])
                        print(f"[GPT] Choices count: {len(choices)}")
                        if choices:
                            first = choices[0]
                            print(f"[GPT] First choice keys: {list(first.keys())}")
                            msg = first.get('message', {})
                            print(f"[GPT] Message keys: {list(msg.keys()) if isinstance(msg, dict) else 'not dict'}")
                            print(f"[GPT] Finish reason: {first.get('finish_reason')}")
                except Exception as e:
                    print(f"[GPT] Could not parse response as JSON: {e}")
            else:
                print(f"[GPT] Empty response. Response snippet: {resp_text[:200]}")
            return GPTMarketAnalysis(ce_pe_bias="UNKNOWN", reason=debug_info, raw_text=resp_text)
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


def _advise_trade_raw(
    *,
    proposal: Dict[str, Any],
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    timeout_sec: float = 45.0,
    temperature: float = 0.0,
    max_tokens: int = 500,
    http_post: Optional[HttpPost] = None,
) -> GPTAdvice:
    """Call OpenAI Chat Completions API to approve/deny a proposed trade."""
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
            'Output schema: {"decision":"TAKE"|"SKIP","reason":string,"confidence":number}. '
            "When relevant, include optional keys like profitability=profitable|not_profitable|unknown and expected_value_sign=-1|0|1. "
            "You MAY include extra JSON keys when helpful (e.g. side/qty/target)."
        )
    )
    user = {"ts": int(time.time()), "proposal": proposal}
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
        try:
            _notify_ui("gpt_advise_http_error", {"status": status, "detail": safe_text})
        except Exception:
            pass
        return GPTAdvice(decision="UNKNOWN", reason=f"HTTP {status}: {safe_text[:200]}{hint}", raw_text=safe_text)
    try:
        data = json.loads(resp_text)
    except Exception as exc:
        return GPTAdvice(decision="UNKNOWN", reason=f"Bad JSON from API: {exc}", raw_text=resp_text)
    content = None
    try:
        choices = data.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content")
    except Exception:
        content = None
    if not content:
        advice = GPTAdvice(decision="UNKNOWN", reason="Empty model response", raw_text=resp_text)
        try:
            _notify_ui("gpt_advise", asdict(advice))
        except Exception:
            pass
        return advice
    parsed = parse_gpt_advice(str(content))
    try:
        _notify_ui("gpt_advise", asdict(parsed))
    except Exception:
        pass
    return parsed


class GPTCircuitBreaker:
    def __init__(self, max_failures: int = 3, cooldown_seconds: float = 600.0):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.disabled_until = 0.0
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            self.failure_count = 0

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            if self.failure_count >= self.max_failures:
                self.disabled_until = time.time() + self.cooldown_seconds
                print(f"[GPTCircuitBreaker] Circuit open! Disabling GPT for 10 minutes.")

    def is_available(self) -> bool:
        with self._lock:
            now = time.time()
            if now < self.disabled_until:
                return False
            if self.disabled_until > 0.0:
                self.disabled_until = 0.0
                self.failure_count = 0
            return True


circuit_breaker = GPTCircuitBreaker()

_cached_market_analysis: Optional[GPTMarketAnalysis] = None
_cached_trade_advice: Optional[GPTAdvice] = None
_gpt_session_disabled = False
_gpt_session_disable_reason = ""
_gpt_session_lock = threading.Lock()


def disable_gpt_for_session(reason: str) -> None:
    global _gpt_session_disabled, _gpt_session_disable_reason
    with _gpt_session_lock:
        _gpt_session_disabled = True
        _gpt_session_disable_reason = str(reason or "GPT disabled for session")
    try:
        print(f"[GPT] Disabled for session: {_gpt_session_disable_reason}")
    except Exception:
        pass
    try:
        _notify_ui("gpt_disabled", {"reason": _gpt_session_disable_reason})
    except Exception:
        pass


def is_gpt_disabled_for_session() -> bool:
    with _gpt_session_lock:
        return bool(_gpt_session_disabled)


def gpt_disabled_reason() -> str:
    with _gpt_session_lock:
        return str(_gpt_session_disable_reason or "GPT disabled for session")


def _update_gpt_latency_metric(latency_sec: float) -> None:
    try:
        import performance_monitor

        performance_monitor.update_gpt_latency(float(latency_sec))
    except Exception:
        pass


def _execute_with_retries(
    *,
    call_fn: Callable[[], Any],
    is_success_fn: Callable[[Any], bool],
    fallback_fn: Callable[[str], Any],
    cached_result: Optional[Any],
    timeout_sec: float,
    max_retries: int = 2,
) -> Any:
    effective_timeout = max(0.5, min(float(timeout_sec or 5.0), 5.0))
    if is_gpt_disabled_for_session():
        return cached_result or fallback_fn(gpt_disabled_reason())
    if not circuit_breaker.is_available():
        print("[GPTCircuitBreaker] Circuit is open. Returning cached fallback.")
        return cached_result or fallback_fn("GPT disabled due to circuit breaker")

    result_box: Dict[str, Any] = {"result": None, "reason": "Request timed out"}

    def worker() -> None:
        last_reason = "Request timed out"
        for attempt in range(max(0, int(max_retries)) + 1):
            try:
                result = call_fn()
                if is_success_fn(result):
                    result_box["result"] = result
                    result_box["reason"] = ""
                    return
                last_reason = str(getattr(result, "reason", "") or "Unknown GPT response")
                if last_reason.lower().startswith("http 402:"):
                    result_box["result"] = result
                    result_box["reason"] = last_reason
                    return
            except Exception as exc:
                last_reason = str(exc)
            if attempt < max_retries:
                time.sleep(0.5)
        result_box["reason"] = last_reason or "Request timed out"

    started = time.perf_counter()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=effective_timeout + 0.5)
    _update_gpt_latency_metric(time.perf_counter() - started)

    result = result_box.get("result")
    if result is not None and is_success_fn(result):
        circuit_breaker.record_success()
        return result

    failure_reason = str(result_box.get("reason") or "Request timed out")
    if failure_reason.lower().startswith("http 402:"):
        disable_gpt_for_session("HTTP 402 from GPT provider")
        return cached_result or fallback_fn(gpt_disabled_reason())

    circuit_breaker.record_failure()
    return cached_result or fallback_fn(failure_reason)


def analyze_market(
    *,
    snapshot: Dict[str, Any],
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    timeout_sec: float = 5.0,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    http_post: Optional["HttpPost"] = None,
) -> GPTMarketAnalysis:
    global _cached_market_analysis

    result = _execute_with_retries(
        call_fn=lambda: _analyze_market_raw(
            snapshot=snapshot,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_sec=min(float(timeout_sec or 5.0), 5.0),
            temperature=temperature,
            max_tokens=max_tokens,
            http_post=http_post,
        ),
        is_success_fn=lambda res: bool(res) and getattr(res, "ce_pe_bias", "UNKNOWN") != "UNKNOWN" and "HTTP 504" not in str(getattr(res, "reason", "") or "") and "timed out" not in str(getattr(res, "reason", "") or "").lower(),
        fallback_fn=lambda reason: GPTMarketAnalysis(
            ce_pe_bias="NEUTRAL",
            reason=reason,
            confidence=0.0,
            raw_text='{"ce_pe_bias":"NEUTRAL","reason":"fallback"}',
        ),
        cached_result=_cached_market_analysis,
        timeout_sec=timeout_sec,
        max_retries=2,
    )
    if getattr(result, "ce_pe_bias", "UNKNOWN") != "UNKNOWN" and "disabled due to circuit breaker" not in str(getattr(result, "reason", "") or "").lower():
        _cached_market_analysis = result
    return result


def advise_trade(
    *,
    proposal: Dict[str, Any],
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    timeout_sec: float = 5.0,
    temperature: float = 0.0,
    max_tokens: int = 500,
    http_post: Optional[HttpPost] = None,
) -> GPTAdvice:
    global _cached_trade_advice

    result = _execute_with_retries(
        call_fn=lambda: _advise_trade_raw(
            proposal=proposal,
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            timeout_sec=min(float(timeout_sec or 5.0), 5.0),
            temperature=temperature,
            max_tokens=max_tokens,
            http_post=http_post,
        ),
        is_success_fn=lambda res: bool(res) and getattr(res, "decision", "UNKNOWN") != "UNKNOWN" and "HTTP 504" not in str(getattr(res, "reason", "") or "") and "timed out" not in str(getattr(res, "reason", "") or "").lower(),
        fallback_fn=lambda reason: GPTAdvice(
            decision="SKIP",
            reason=reason,
            confidence=0.0,
            raw_text='{"decision":"SKIP","reason":"fallback"}',
        ),
        cached_result=_cached_trade_advice,
        timeout_sec=timeout_sec,
        max_retries=2,
    )
    if getattr(result, "decision", "UNKNOWN") != "UNKNOWN" and "disabled due to circuit breaker" not in str(getattr(result, "reason", "") or "").lower():
        _cached_trade_advice = result
    return result
