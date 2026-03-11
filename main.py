"""
mewhisk.py — Metaculus forecasting bot
======================================

MODEL ROLE ASSIGNMENTS
----------------------
Role                  | Model                                    | Key
----------------------|------------------------------------------|------------------
forecaster_primary    | gemini/gemini-2.5-pro-preview-06-05      | GEMINI_API_KEY
forecaster_checker    | gemini/gemini-2.0-flash                  | GEMINI_API_KEY
parser                | openrouter/mistralai/mistral-7b-instruct:free | OPENROUTER_API_KEY
summarizer            | openrouter/mistralai/mistral-7b-instruct:free | OPENROUTER_API_KEY
researcher            | openrouter/mistralai/mistral-7b-instruct:free | OPENROUTER_API_KEY
default               | openrouter/mistralai/mistral-7b-instruct:free | OPENROUTER_API_KEY

DESIGN
------
- Gemini quota is spent ONLY on the two forecasting calls per question.
- OpenRouter free-tier Mistral handles all utility work: parsing structured
  outputs, summarising research, format retries. High RPM, zero Gemini cost.
- Separate rate-limit buckets per model tier with exponential backoff on 429s.

SEARCH
------
  Exa + Linkup run in parallel. Each falls back to the other on failure.
  AskNews optional for live news context.

AGGREGATION
-----------
  Weighted blend: 0.65 × primary + 0.35 × checker
  Binary extremize:  logit-scale k=1.15  (conservative ~2pp push at p=0.70)
  MC extremize:      power-transform k=1.10 (mild sharpening of leading option)
  Numeric:           NO extremization (preserves distributional integrity)

REQUIRED ENV VARS
-----------------
  GEMINI_API_KEY      aistudio.google.com/app/apikey
  OPENROUTER_API_KEY  openrouter.ai/keys  (free tier sufficient)
  EXA_API_KEY         exa.ai
  LINKUP_API_KEY      linkup.so

OPTIONAL
--------
  ASKNEWS_CLIENT_ID / ASKNEWS_CLIENT_SECRET
"""

import argparse
import asyncio
import logging
import math
import os
import re
import textwrap
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# ─────────────────────────────────────────────────────────────
# Optional search integrations
# ─────────────────────────────────────────────────────────────
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False

try:
    from linkup import LinkupClient
    LINKUP_AVAILABLE = True
except ImportError:
    LINKUP_AVAILABLE = False

try:
    from asknews_sdk import AskNewsSDK
    ASKNEWS_SDK_AVAILABLE = True
except ImportError:
    ASKNEWS_SDK_AVAILABLE = False
    import requests  # type: ignore

from forecasting_tools import (
    BinaryQuestion,
    BinaryPrediction,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mewhisk")

# ─────────────────────────────────────────────────────────────
# API keys
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY    = os.getenv("OPENROUTER_API_KEY")
EXA_API_KEY           = os.getenv("EXA_API_KEY")
LINKUP_API_KEY        = os.getenv("LINKUP_API_KEY")
ASKNEWS_CLIENT_ID     = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_CLIENT_SECRET = os.getenv("ASKNEWS_CLIENT_SECRET")

EXA_ENABLED     = EXA_AVAILABLE    and bool(EXA_API_KEY)
LINKUP_ENABLED  = LINKUP_AVAILABLE and bool(LINKUP_API_KEY)
ASKNEWS_ENABLED = bool(ASKNEWS_CLIENT_ID and ASKNEWS_CLIENT_SECRET)

for _label, _ok, _hint in [
    ("GEMINI_API_KEY",     bool(GEMINI_API_KEY),     "aistudio.google.com/app/apikey"),
    ("OPENROUTER_API_KEY", bool(OPENROUTER_API_KEY), "openrouter.ai/keys (free tier ok)"),
    ("Exa",                EXA_ENABLED,              "pip install exa-py + EXA_API_KEY"),
    ("Linkup",             LINKUP_ENABLED,           "pip install linkup-sdk + LINKUP_API_KEY"),
    ("AskNews",            ASKNEWS_ENABLED,          "optional: ASKNEWS_CLIENT_ID + _SECRET"),
]:
    if not _ok:
        logger.warning(f"⚠️  {_label} not configured — {_hint}")

# ─────────────────────────────────────────────────────────────
# Model strings
# ─────────────────────────────────────────────────────────────

# FORECASTERS — Gemini direct via Google AI API
# LiteLLM routes "gemini/..." using GEMINI_API_KEY automatically.
MODEL_FORECASTER_PRIMARY = "gemini/gemini-2.5-pro-preview-06-05"
MODEL_FORECASTER_CHECKER = "gemini/gemini-2.0-flash"

# UTILITY ROLES — OpenRouter free tier
# LiteLLM routes "openrouter/..." using OPENROUTER_API_KEY automatically.
# Mistral-7B-Instruct is reliable for structured output parsing.
# ":free" suffix = OpenRouter free model quota (no billing needed).
_OR_FREE         = "openrouter/mistralai/mistral-7b-instruct:free"
MODEL_PARSER     = _OR_FREE
MODEL_SUMMARIZER = _OR_FREE
MODEL_RESEARCHER = _OR_FREE
MODEL_DEFAULT    = _OR_FREE

# ─────────────────────────────────────────────────────────────
# Rate-limit buckets
# Gemini free tier: ~2 RPM Pro, ~15 RPM Flash
# OpenRouter free:  ~20 RPM typical (generous)
# ─────────────────────────────────────────────────────────────
_RL_DELAY: Dict[str, float] = {
    "primary": 32.0,   # Gemini 2.5 Pro  — 2 RPM → 30s + 2s buffer
    "checker":  5.0,   # Gemini 2.0 Flash — 15 RPM → 4s + 1s buffer
    "utility":  3.0,   # OpenRouter free  — comfortable pace
}
RATE_LIMIT_MAX_RETRIES  = 6
RATE_LIMIT_BACKOFF_BASE = 2.0

_last_call: Dict[str, float]         = {}
_rl_locks:  Dict[str, asyncio.Lock]  = {}


def _get_rl_lock(bucket: str) -> asyncio.Lock:
    if bucket not in _rl_locks:
        _rl_locks[bucket] = asyncio.Lock()
    return _rl_locks[bucket]


async def _rate_limited_delay(bucket: str) -> None:
    delay = _RL_DELAY.get(bucket, 3.0)
    async with _get_rl_lock(bucket):
        elapsed = time.monotonic() - _last_call.get(bucket, 0.0)
        wait = delay - elapsed
        if wait > 0:
            logger.debug(f"[rl:{bucket}] waiting {wait:.1f}s")
            await asyncio.sleep(wait)
        _last_call[bucket] = time.monotonic()


# ─────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def median(lst: List[Union[float, int]]) -> float:
    vals = sorted(float(x) for x in lst if _is_num(x))
    if not vals:
        raise ValueError("median(): no numeric values")
    n, mid = len(vals), len(vals) // 2
    return (vals[mid - 1] + vals[mid]) / 2.0 if n % 2 == 0 else vals[mid]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci90(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 1.0
    m, s = mean(xs), stdev(xs)
    se = s / math.sqrt(len(xs))
    return max(0.0, m - 1.645 * se), min(1.0, m + 1.645 * se)


def entropy_nats(probs: Dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in probs.values() if p > 0)


def safe_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if x is None:
            return default
        s = str(x).strip().replace(",", "").replace("%", "")
        return float(s) if s else default
    except Exception:
        return default


def normalize_percentile(p: Any) -> float:
    v = safe_float(p, default=0.5) or 0.5
    if v > 1.0:
        v /= 100.0
    return max(0.0, min(1.0, v))


def clamp(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, float(p)))


# ─────────────────────────────────────────────────────────────
# Conservative extremization
# ─────────────────────────────────────────────────────────────
EXTREMIZE_K_BINARY_DEFAULT = 1.15   # ~2pp push at p=0.70, ~5pp at p=0.85
EXTREMIZE_K_MC_DEFAULT     = 1.10   # mild sharpening toward leading option


def _logit(p: float) -> float:
    p = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 \
        else math.exp(x) / (1.0 + math.exp(x))


def extremize_binary(p: float, k: float) -> float:
    if not _is_num(k) or k <= 0 or abs(k - 1.0) < 1e-9:
        return float(p)
    return clamp(_sigmoid(_logit(float(p)) * k))


def extremize_mc(probs: Dict[str, float], k: float) -> Dict[str, float]:
    if not probs or not _is_num(k) or k <= 0 or abs(k - 1.0) < 1e-9:
        s = sum(max(0.0, v) for v in probs.values())
        return ({a: max(0.0, v) / s for a, v in probs.items()} if s > 0
                else {a: 1 / len(probs) for a in probs})
    powered = {a: max(0.0, float(v)) ** k for a, v in probs.items()}
    s = sum(powered.values())
    return ({a: v / s for a, v in powered.items()} if s > 0
            else {a: 1 / len(probs) for a in probs})


# ─────────────────────────────────────────────────────────────
# Regex parsers (fallback when structured output fails)
# ─────────────────────────────────────────────────────────────
_PERCENT_RE = re.compile(r"(?i)\bprob(?:ability)?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%")
_DEC_RE     = re.compile(r"(?i)\bdecimal\s*:\s*([0-9]*\.?[0-9]+)\b")
_SOLO_PCT   = re.compile(r"(?<!\d)([0-9]{1,3}(?:\.[0-9]+)?)\s*%")


def extract_binary_prob(text: str) -> Optional[float]:
    if not text:
        return None
    for pat in [_PERCENT_RE, _DEC_RE]:
        m = pat.search(text)
        if m:
            v = safe_float(m.group(1))
            if v is not None:
                return clamp(v / 100.0 if v > 1.0 else v, 0.0, 1.0)
    m = _SOLO_PCT.search(text)
    if m:
        v = safe_float(m.group(1))
        if v is not None:
            return clamp(v / 100.0, 0.0, 1.0)
    return None


def build_indexed_options(options: List[str]) -> List[str]:
    return [f"{i + 1}) {opt}" for i, opt in enumerate(options)]


def extract_indexed_mc_probs(text: str, n: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for pat in [
        re.compile(r"(?i)\b(?:option\s*)?(\d{1,2})\s*[:\)\-]\s*([0-9]+(?:\.[0-9]+)?)\s*%"),
        re.compile(r"(?i)\b(\d{1,2})\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%"),
    ]:
        for m in pat.finditer(text):
            idx = int(m.group(1))
            if 1 <= idx <= n:
                v = safe_float(m.group(2))
                if v is not None:
                    out[idx] = v / 100.0
    return out


def extract_numeric_percentiles(text: str, targets: List[float]) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for pt in targets:
        pi = int(round(pt * 100))
        for pat in [
            re.compile(rf"(?i)\bpercentile\s*{pi}\s*:\s*([-+]?[0-9,]*\.?[0-9]+)"),
            re.compile(rf"(?i)\bp\s*{pi}\s*:\s*([-+]?[0-9,]*\.?[0-9]+)"),
            re.compile(rf"(?i)\bp{pi}\s*=\s*([-+]?[0-9,]*\.?[0-9]+)"),
        ]:
            m = pat.search(text)
            if m:
                v = safe_float(m.group(1))
                if v is not None:
                    out[pt] = float(v)
                    break
    return out


# ─────────────────────────────────────────────────────────────
# Search query builder
# ─────────────────────────────────────────────────────────────
def build_search_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
    q  = re.sub(r"http\S+|\s+", " ", question.question_text  or "").strip()
    bg = re.sub(r"http\S+|\s+", " ", question.background_info or "").strip()
    if len(q) <= max_chars:
        cand = f"{q} — {bg}" if bg else q
        if len(cand) <= max_chars:
            return cand
        space = max_chars - len(q) - 3
        return f"{q} — {textwrap.shorten(bg, width=space, placeholder='…')}" if space > 10 else q
    first = q.split(".")[0].strip()
    if len(first) > max_chars:
        return textwrap.shorten(first, width=max_chars, placeholder="…")
    rem = max_chars - len(first) - 3
    if rem > 10 and bg:
        combo = f"{first} — {textwrap.shorten(bg, width=rem, placeholder='…')}"
        if len(combo) <= max_chars:
            return combo
    return textwrap.shorten(q, width=max_chars, placeholder="…")


# ─────────────────────────────────────────────────────────────
# Exa search
# ─────────────────────────────────────────────────────────────
def _sync_exa_search(query: str, max_results: int = 5) -> List[str]:
    if not EXA_ENABLED:
        return []
    try:
        client  = Exa(api_key=EXA_API_KEY)
        results = client.search_and_contents(
            query, type="auto", num_results=max_results,
            highlights={"max_characters": 4000},
        )
        out = []
        for i, r in enumerate(results.results[:max_results]):
            title   = getattr(r, "title",  "Untitled") or "Untitled"
            url     = getattr(r, "url",    "")         or ""
            hl      = getattr(r, "highlights", None)
            content = (" ".join(hl) if isinstance(hl, list)
                       else (getattr(r, "text", "") or ""))[:600]
            out.append(
                f"[E{i+1}] {title}: {textwrap.shorten(content, width=260, placeholder='…')}"
                + (f" ({url})" if url else "")
            )
        return out
    except Exception as e:
        logger.error(f"Exa search error: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Linkup search
# ─────────────────────────────────────────────────────────────
def _sync_linkup_search(query: str, max_results: int = 5) -> List[str]:
    if not LINKUP_ENABLED:
        return []
    try:
        client   = LinkupClient(api_key=LINKUP_API_KEY)
        response = client.search(
            query=query, depth="standard", output_type="searchResults"
        )
        results = getattr(response, "results", []) or []
        out = []
        for i, r in enumerate(results[:max_results]):
            title   = getattr(r, "name",    "Untitled") or "Untitled"
            url     = getattr(r, "url",     "")         or ""
            content = (getattr(r, "content","")         or "")[:600]
            out.append(
                f"[L{i+1}] {title}: {textwrap.shorten(content, width=260, placeholder='…')}"
                + (f" ({url})" if url else "")
            )
        return out
    except Exception as e:
        logger.error(f"Linkup search error: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Unified web search — Exa + Linkup parallel, mutual fallback
# ─────────────────────────────────────────────────────────────
async def _unified_web_search(
    query: str, loop: asyncio.AbstractEventLoop
) -> Tuple[List[str], str]:
    exa_snips:    List[str] = []
    linkup_snips: List[str] = []
    exa_ok = linkup_ok = False

    async def _run_exa():
        nonlocal exa_ok
        if not EXA_ENABLED:
            return
        try:
            exa_snips.extend(
                await loop.run_in_executor(None, _sync_exa_search, query)
            )
            exa_ok = True
        except Exception as e:
            logger.error(f"Exa failed: {e}")

    async def _run_linkup():
        nonlocal linkup_ok
        if not LINKUP_ENABLED:
            return
        try:
            linkup_snips.extend(
                await loop.run_in_executor(None, _sync_linkup_search, query)
            )
            linkup_ok = True
        except Exception as e:
            logger.error(f"Linkup failed: {e}")

    await asyncio.gather(_run_exa(), _run_linkup())

    merged: List[str] = []
    for i in range(max(len(exa_snips), len(linkup_snips))):
        if i < len(exa_snips):
            merged.append(exa_snips[i])
        if i < len(linkup_snips):
            merged.append(linkup_snips[i])

    label = (
        "EXA + LINKUP"             if exa_ok and linkup_ok else
        "EXA (Linkup unavailable)" if exa_ok               else
        "LINKUP (Exa unavailable)" if linkup_ok             else
        "WEB SEARCH UNAVAILABLE"
    )
    return merged, label


# ─────────────────────────────────────────────────────────────
# AskNews search
# ─────────────────────────────────────────────────────────────
def _sync_asknews_search(client: Any, query: str) -> List[Any]:
    if client is None:
        return []
    if ASKNEWS_SDK_AVAILABLE:
        try:
            fn = (getattr(client.news, "search_news",    None)
                  or getattr(client.news, "search_stories", None))
            if fn is None:
                raise AttributeError("No search method on AskNews client")
            resp = fn(query=query, n_articles=5, return_type="news",
                      method="kw", return_story_text=True)
            if hasattr(resp, "news"):
                return resp.news
            data = getattr(resp, "data", resp if isinstance(resp, dict) else {})
            return data.get("news", []) if isinstance(data, dict) else []
        except Exception as e:
            logger.error(f"AskNews SDK error: {e}")
            return []
    else:
        try:
            headers = {"Authorization": f"Bearer {client['token']}"}
            r = requests.get(
                "https://api.asknews.app/v1/news",
                headers=headers,
                params={"q": query, "n_articles": 5, "sort": "relevance",
                        "return_type": "news", "return_story_text": "true"},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json().get("data", r.json())
            return data.get("news", [])
        except Exception as e:
            logger.error(f"AskNews HTTP error: {e}")
            return []


# ─────────────────────────────────────────────────────────────
# MEWHISK BOT
# ─────────────────────────────────────────────────────────────
class mewhisk(ForecastBot):
    """
    mewhisk — search-grounded Gemini forecasting bot.

    Role → Model routing
    --------------------
    forecaster_primary  →  Gemini 2.5 Pro   (deep reasoning, Bayesian forecast)
    forecaster_checker  →  Gemini 2.0 Flash  (adversarial second opinion)
    parser              →  Mistral-7B :free  (structure LLM output into typed objects)
    summarizer          →  Mistral-7B :free  (condense long research into bullets)
    researcher          →  Mistral-7B :free  (question decomposition / gap-fill)
    default             →  Mistral-7B :free  (fallback for any unspecified role)

    Gemini quota is spent ONLY on the two forecasting calls per question.
    All parsing, summarising, and format retries use OpenRouter free tier.
    """

    _max_concurrent_questions            = 1
    _concurrency_limiter                 = asyncio.Semaphore(1)
    _structure_output_validation_samples = 1   # minimise OpenRouter utility calls

    # ----------------------------------------------------------
    # Role → model string (consumed by ForecastBot base class)
    # ----------------------------------------------------------
    def _llm_config_defaults(self) -> Dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            # ── Gemini forecasting roles ──────────────────────────
            "forecaster_primary": MODEL_FORECASTER_PRIMARY,
            "forecaster_checker": MODEL_FORECASTER_CHECKER,
            # ── OpenRouter utility roles ──────────────────────────
            "parser":             MODEL_PARSER,
            "summarizer":         MODEL_SUMMARIZER,
            "researcher":         MODEL_RESEARCHER,
            "default":            MODEL_DEFAULT,
        })
        return defaults

    def __init__(
        self,
        *args,
        extremize_enabled:  bool  = True,
        extremize_k_binary: float = EXTREMIZE_K_BINARY_DEFAULT,
        extremize_k_mc:     float = EXTREMIZE_K_MC_DEFAULT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.extremize_enabled  = extremize_enabled
        self.extremize_k_binary = extremize_k_binary
        self.extremize_k_mc     = extremize_k_mc
        self._asknews_client    = None
        self._drop:          Dict[str, int]            = {}
        self._drop_by_model: Dict[str, Dict[str, int]] = {}

        logger.info(
            f"🐱 mewhisk ready\n"
            f"   primary  : {MODEL_FORECASTER_PRIMARY}\n"
            f"   checker  : {MODEL_FORECASTER_CHECKER}\n"
            f"   utility  : {MODEL_DEFAULT} (parser/summarizer/researcher/default)\n"
            f"   search   : Exa={EXA_ENABLED} Linkup={LINKUP_ENABLED} "
            f"AskNews={ASKNEWS_ENABLED}\n"
            f"   extremize: {extremize_enabled} "
            f"k_bin={extremize_k_binary:.2f} k_mc={extremize_k_mc:.2f}"
        )

    def _inc_drop(self, tag: str, reason: str) -> None:
        self._drop[reason] = self._drop.get(reason, 0) + 1
        d = self._drop_by_model.setdefault(tag, {})
        d[reason] = d.get(reason, 0) + 1

    # ----------------------------------------------------------
    # AskNews client (lazy init)
    # ----------------------------------------------------------
    def _get_asknews_client(self) -> Any:
        if self._asknews_client is not None:
            return self._asknews_client
        if not ASKNEWS_ENABLED:
            return None
        if ASKNEWS_SDK_AVAILABLE:
            self._asknews_client = AskNewsSDK(
                client_id=ASKNEWS_CLIENT_ID,
                client_secret=ASKNEWS_CLIENT_SECRET,
                scopes=["news"],
            )
        else:
            try:
                r = requests.post(
                    "https://api.asknews.app/v1/oauth/token",
                    data={"grant_type": "client_credentials",
                          "client_id":     ASKNEWS_CLIENT_ID,
                          "client_secret": ASKNEWS_CLIENT_SECRET,
                          "scope":         "news"},
                    timeout=10,
                )
                r.raise_for_status()
                self._asknews_client = {"token": r.json()["access_token"]}
            except Exception as e:
                logger.error(f"AskNews auth failed: {e}")
                self._asknews_client = None
        return self._asknews_client

    # ----------------------------------------------------------
    # Research: Exa + Linkup + AskNews
    # Summarizer role (OpenRouter :free) compresses if too long.
    # ----------------------------------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today = datetime.now().strftime("%Y-%m-%d")
            query = build_search_query(question)
            loop  = asyncio.get_running_loop()

            # AskNews (optional)
            asknews_out = "[AskNews disabled]"
            if ASKNEWS_ENABLED:
                try:
                    client  = self._get_asknews_client()
                    stories = await loop.run_in_executor(
                        None, _sync_asknews_search, client, query
                    )
                    if stories:
                        lines = []
                        for i, s in enumerate(stories[:5]):
                            t = ((s.get("title") if isinstance(s, dict)
                                  else getattr(s, "title", "Untitled")) or "Untitled")
                            x = (((s.get("text") if isinstance(s, dict)
                                   else getattr(s, "text", "")) or ""))[:700]
                            lines.append(
                                f"[A{i+1}] {t}: "
                                f"{textwrap.shorten(x, width=260, placeholder='…')}"
                            )
                        asknews_out = "\n".join(lines)
                    else:
                        asknews_out = "[AskNews: no stories found]"
                except Exception as e:
                    asknews_out = f"[AskNews error: {e}]"

            # Exa + Linkup
            web_out = "[Web search unavailable]"
            try:
                snippets, label = await _unified_web_search(query, loop)
                web_out = ("\n".join(snippets) if snippets
                           else f"[{label}: no results]")
                logger.info(f"🔍 {label} — {len(snippets)} snippets")
            except Exception as e:
                web_out = f"[Web search error: {e}]"

            raw = (
                f"=== RESEARCH (as of {today}) ===\n\n"
                f"--- NEWS (AskNews) ---\n{asknews_out}\n\n"
                f"--- WEB SNIPPETS (Exa + Linkup) ---\n{web_out}\n"
            )

            # Summarizer role: compress if research is very long
            # (saves Gemini forecaster token budget; uses free OpenRouter call)
            if len(raw) > 6000:
                try:
                    await _rate_limited_delay("utility")
                    summarizer = self.get_llm("summarizer", "llm")
                    summary = await summarizer.invoke(
                        clean_indents(f"""
                            Summarise the research below into 10 concise bullet points
                            relevant to this question: {question.question_text}

                            Preserve ALL specific numbers, dates, names, and percentages.
                            Output ONLY the bullets — no preamble.

                            RESEARCH:
                            {raw}
                        """)
                    )
                    return (
                        f"=== SUMMARISED RESEARCH (as of {today}) ===\n"
                        f"[Compressed {len(raw)} chars → bullets by summarizer role]\n\n"
                        f"{summary}\n\n"
                        f"--- RAW SNIPPETS (truncated) ---\n{raw[:1500]}…\n"
                    )
                except Exception as e:
                    logger.warning(f"Summarizer failed, using raw research: {e}")

            return raw

    # ----------------------------------------------------------
    # Rate-limited LLM call with exponential backoff on 429
    # role_name  : key into _llm_config_defaults (e.g. "forecaster_primary")
    # rl_bucket  : "primary" | "checker" | "utility"
    # ----------------------------------------------------------
    async def _invoke_safe(
        self, role_name: str, prompt: str, rl_bucket: str = "utility"
    ) -> str:
        llm = self.get_llm(role_name, "llm")
        for attempt in range(RATE_LIMIT_MAX_RETRIES):
            await _rate_limited_delay(rl_bucket)
            try:
                return await llm.invoke(prompt)
            except Exception as e:
                err = str(e).lower()
                if any(tok in err for tok in
                       ("429", "rate", "quota", "resource_exhausted", "too many")):
                    wait = RATE_LIMIT_BACKOFF_BASE ** (attempt + 2) * 10
                    logger.warning(
                        f"[rl:{rl_bucket}] quota hit on '{role_name}' — "
                        f"retry {attempt+1}/{RATE_LIMIT_MAX_RETRIES} in {wait:.0f}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError(
            f"'{role_name}' exhausted {RATE_LIMIT_MAX_RETRIES} rate-limit retries"
        )

    # ----------------------------------------------------------
    # Parsers (use "parser" role = OpenRouter :free Mistral)
    # ----------------------------------------------------------
    async def _parse_binary(self, raw: str, tag: str) -> Optional[float]:
        try:
            pred: BinaryPrediction = await structure_output(
                text_to_structure=raw,
                output_type=BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            v = safe_float(getattr(pred, "prediction_in_decimal", None))
            if v is not None:
                return clamp(float(v))
        except Exception:
            self._inc_drop(tag, "parse_structured_binary")
        v2 = extract_binary_prob(raw)
        if v2 is None:
            self._inc_drop(tag, "parse_fallback_binary")
        return v2

    async def _parse_mc(
        self, raw: str, question: MultipleChoiceQuestion, tag: str
    ) -> Optional[Dict[str, float]]:
        options = list(question.options)
        try:
            pred: PredictedOptionList = await structure_output(
                text_to_structure=raw,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=f"Valid options: {options}",
                num_validation_samples=self._structure_output_validation_samples,
            )
            pd: Dict[str, float] = {}
            for po in pred.predicted_options:
                if _is_num(po.probability):
                    pd[str(po.option_name).strip()] = float(po.probability)
            out: Dict[str, float] = {}
            for opt in options:
                if opt in pd:
                    out[opt] = pd[opt]
                else:
                    for k, v in pd.items():
                        if k.casefold() == opt.casefold():
                            out[opt] = v
                            break
            if out:
                return out
        except Exception:
            self._inc_drop(tag, "parse_structured_mc")
        idx = extract_indexed_mc_probs(raw, len(options))
        if not idx:
            self._inc_drop(tag, "parse_fallback_mc")
            return None
        return ({options[i - 1]: idx[i]
                 for i in range(1, len(options) + 1) if i in idx} or None)

    async def _parse_numeric(
        self, raw: str, question: NumericQuestion, tag: str
    ) -> Optional[NumericDistribution]:
        targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        try:
            pts_raw: List[Percentile] = await structure_output(
                text_to_structure=raw,
                output_type=list[Percentile],
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            pts = []
            for p in pts_raw:
                v = safe_float(getattr(p, "value", None))
                if v is None:
                    continue
                pts.append(Percentile(
                    value=float(v),
                    percentile=normalize_percentile(getattr(p, "percentile", 0.5)),
                ))
            if pts:
                pts.sort(key=lambda x: x.percentile)
                for i in range(1, len(pts)):
                    if pts[i].value < pts[i - 1].value:
                        pts[i].value = pts[i - 1].value
                return NumericDistribution.from_question(pts, question)
        except Exception:
            self._inc_drop(tag, "parse_structured_numeric")
        extracted = extract_numeric_percentiles(raw, targets)
        if not extracted:
            self._inc_drop(tag, "parse_fallback_numeric")
            return None
        pts2 = sorted(
            [Percentile(percentile=pt, value=float(extracted[pt]))
             for pt in targets if pt in extracted],
            key=lambda x: x.percentile,
        )
        for i in range(1, len(pts2)):
            if pts2[i].value < pts2[i - 1].value:
                pts2[i].value = pts2[i - 1].value
        return NumericDistribution.from_question(pts2, question)

    # ----------------------------------------------------------
    # Bounds helper
    # ----------------------------------------------------------
    def _bound_msgs(self, q: NumericQuestion) -> Tuple[str, str]:
        lo = q.nominal_lower_bound if q.nominal_lower_bound is not None else q.lower_bound
        hi = q.nominal_upper_bound if q.nominal_upper_bound is not None else q.upper_bound
        return (
            f"Cannot be lower than {lo}." if not q.open_lower_bound
            else f"Unlikely below {lo}.",
            f"Cannot be higher than {hi}." if not q.open_upper_bound
            else f"Unlikely above {hi}.",
        )

    # ----------------------------------------------------------
    # Prompts — search-grounded, statistical framing
    # ----------------------------------------------------------
    def _prompt_binary(self, q: BinaryQuestion, research: str, role: str) -> str:
        return clean_indents(f"""
            You are a calibrated Bayesian forecaster optimising log-score on Metaculus.
            Role: {role}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            QUESTION  : {q.question_text}
            BACKGROUND: {q.background_info}
            RESOLUTION: {q.resolution_criteria}
            FINE PRINT: {q.fine_print}

            EVIDENCE — derive your probability estimate FROM these search results:
            {research}

            INSTRUCTIONS
            1. Identify the outside-view base rate from evidence for this class of event.
            2. Anchor on that base rate. Adjust only with specific inside-view evidence above.
            3. State the strongest counter-argument and reduce confidence if compelling.
            4. Be conservative — do not assign <5% or >95% without overwhelming evidence.

            Write 6-8 reasoning bullets, then output EXACTLY:
            Probability: ZZ%
            Decimal: 0.ZZ
        """)

    def _prompt_mc(
        self, q: MultipleChoiceQuestion, research: str, role: str
    ) -> str:
        indexed = build_indexed_options(list(q.options))
        return clean_indents(f"""
            You are a calibrated Bayesian forecaster optimising log-score on Metaculus.
            Role: {role}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            QUESTION  : {q.question_text}
            BACKGROUND: {q.background_info}
            RESOLUTION: {q.resolution_criteria}
            FINE PRINT: {q.fine_print}

            OPTIONS (use these exact numbers in output):
            {chr(10).join(indexed)}

            EVIDENCE — derive your probability estimate FROM these search results:
            {research}

            INSTRUCTIONS
            1. Extract base rates for each option from the evidence.
            2. Apply reference-class reasoning — how often do similar cases resolve each way?
            3. Adjust only with inside-view signals found in the evidence.
            4. Probabilities MUST sum to exactly 100%.

            Write 6-8 reasoning bullets, then output EXACTLY:
            1: XX%
            2: XX%
            ...
            {len(q.options)}: XX%
        """)

    def _prompt_numeric(
        self, q: NumericQuestion, research: str, role: str
    ) -> str:
        lo_msg, hi_msg = self._bound_msgs(q)
        unit = getattr(q, "unit_of_measure", "inferred units")
        return clean_indents(f"""
            You are a calibrated Bayesian forecaster optimising log-score on Metaculus.
            Role: {role}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            QUESTION  : {q.question_text}
            BACKGROUND: {q.background_info}
            RESOLUTION: {q.resolution_criteria}
            FINE PRINT: {q.fine_print}
            UNITS     : {unit}
            BOUNDS    : {lo_msg} {hi_msg}

            EVIDENCE — derive your distribution FROM these search results:
            {research}

            INSTRUCTIONS
            1. Extract numeric anchors from evidence (past values, rates, comparators).
            2. Estimate central tendency first, then model uncertainty around it.
            3. Percentile values MUST be monotonically increasing.
            4. Calibrate tails conservatively — avoid implausibly wide or narrow ranges.

            Write 6-8 reasoning bullets, then output EXACTLY:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)

    # ----------------------------------------------------------
    # Role runners — Gemini calls via _invoke_safe
    # ----------------------------------------------------------
    async def _run_binary(
        self, q: BinaryQuestion, research: str, tag: str, role: str
    ) -> ReasonedPrediction[float]:
        prompt = self._prompt_binary(q, research, role)
        try:
            raw = await self._invoke_safe(f"forecaster_{tag}", prompt, rl_bucket=tag)
        except Exception as e:
            self._inc_drop(tag, "llm_error_binary")
            raise
        val = await self._parse_binary(raw, tag)
        if val is None:
            # Format retry via parser role (OpenRouter, not Gemini)
            try:
                raw2 = await self._invoke_safe(
                    "parser",
                    "Output ONLY these two lines:\nProbability: ZZ%\nDecimal: 0.ZZ",
                    rl_bucket="utility",
                )
                val = await self._parse_binary(raw2, tag)
                raw += "\n\n[FORMAT_RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_binary")
        if val is None:
            self._inc_drop(tag, "fallback_0.5_binary")
            val = 0.5
        return ReasonedPrediction(prediction_value=clamp(val), reasoning=raw)

    async def _run_mc(
        self, q: MultipleChoiceQuestion, research: str, tag: str, role: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = self._prompt_mc(q, research, role)
        try:
            raw = await self._invoke_safe(f"forecaster_{tag}", prompt, rl_bucket=tag)
        except Exception as e:
            self._inc_drop(tag, "llm_error_mc")
            raise
        probs = await self._parse_mc(raw, q, tag)
        if probs is None:
            try:
                raw2 = await self._invoke_safe(
                    "parser",
                    "Output ONLY option probabilities summing to 100%:\n1: XX%\n2: XX%\n...",
                    rl_bucket="utility",
                )
                probs = await self._parse_mc(raw2, q, tag)
                raw += "\n\n[FORMAT_RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_mc")
        if probs is None:
            self._inc_drop(tag, "fallback_uniform_mc")
            u = 1.0 / max(1, len(q.options))
            probs = {opt: u for opt in q.options}
        return ReasonedPrediction(
            prediction_value=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=opt, probability=p)
                    for opt, p in probs.items()
                ]
            ),
            reasoning=raw,
        )

    async def _run_numeric(
        self, q: NumericQuestion, research: str, tag: str, role: str
    ) -> ReasonedPrediction[NumericDistribution]:
        prompt = self._prompt_numeric(q, research, role)
        try:
            raw = await self._invoke_safe(f"forecaster_{tag}", prompt, rl_bucket=tag)
        except Exception as e:
            self._inc_drop(tag, "llm_error_numeric")
            raise
        dist = await self._parse_numeric(raw, q, tag)
        if dist is None:
            try:
                raw2 = await self._invoke_safe(
                    "parser",
                    ("Output ONLY:\nPercentile 10: X\nPercentile 20: X\n"
                     "Percentile 40: X\nPercentile 60: X\n"
                     "Percentile 80: X\nPercentile 90: X"),
                    rl_bucket="utility",
                )
                dist = await self._parse_numeric(raw2, q, tag)
                raw += "\n\n[FORMAT_RETRY]\n" + raw2
            except Exception:
                self._inc_drop(tag, "retry_failed_numeric")
        if dist is None:
            self._inc_drop(tag, "fallback_midpoint_numeric")
            try:
                lo = float(q.lower_bound or 0.0)
                hi = float(q.upper_bound or 100.0)
            except Exception:
                lo, hi = 0.0, 100.0
            dist = NumericDistribution.from_question(
                [Percentile(value=(lo + hi) / 2, percentile=0.5)], q
            )
        return ReasonedPrediction(prediction_value=dist, reasoning=raw)

    # ----------------------------------------------------------
    # ForecastBot abstract method shims
    # ----------------------------------------------------------
    async def _run_forecast_on_binary(
        self, q: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        return await self._run_binary(q, research, "primary", "PRIMARY")

    async def _run_forecast_on_multiple_choice(
        self, q: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        return await self._run_mc(q, research, "primary", "PRIMARY")

    async def _run_forecast_on_numeric(
        self, q: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._run_numeric(q, research, "primary", "PRIMARY")

    # ----------------------------------------------------------
    # Core aggregation + conservative extremization
    # ----------------------------------------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        async with self._concurrency_limiter:
            preds:      List[Any] = []
            reasonings: List[str] = []

            W_PRIMARY = 0.65
            W_CHECKER  = 0.35

            # ── Primary: Gemini 2.5 Pro ───────────────────────────
            try:
                if isinstance(question, BinaryQuestion):
                    p = await self._run_binary(
                        question, research, "primary", "PRIMARY_FORECASTER"
                    )
                elif isinstance(question, MultipleChoiceQuestion):
                    p = await self._run_mc(
                        question, research, "primary", "PRIMARY_FORECASTER"
                    )
                elif isinstance(question, NumericQuestion):
                    p = await self._run_numeric(
                        question, research, "primary", "PRIMARY_FORECASTER"
                    )
                else:
                    raise ValueError(f"Unknown question type: {type(question)}")
                preds.append(p.prediction_value)
                reasonings.append("[GEMINI 2.5 PRO — PRIMARY]\n" + p.reasoning)
            except Exception as e:
                logger.error(f"Primary forecaster failed: {e}")

            # ── Checker: Gemini 2.0 Flash ─────────────────────────
            try:
                if isinstance(question, BinaryQuestion):
                    c = await self._run_binary(
                        question, research, "checker", "ADVERSARIAL_CHECKER"
                    )
                elif isinstance(question, MultipleChoiceQuestion):
                    c = await self._run_mc(
                        question, research, "checker", "ADVERSARIAL_CHECKER"
                    )
                elif isinstance(question, NumericQuestion):
                    c = await self._run_numeric(
                        question, research, "checker", "ADVERSARIAL_CHECKER"
                    )
                else:
                    raise ValueError(f"Unknown question type: {type(question)}")
                preds.append(c.prediction_value)
                reasonings.append("[GEMINI 2.0 FLASH — CHECKER]\n" + c.reasoning)
            except Exception as e:
                logger.error(f"Checker forecaster failed: {e}")

            if not preds:
                raise RuntimeError("All forecasters failed.")

            joined = "\n\n---\n\n".join(reasonings)

            # ── BINARY ───────────────────────────────────────────
            if isinstance(question, BinaryQuestion):
                g = float(preds[0]) if len(preds) >= 1 and _is_num(preds[0]) else None
                c = float(preds[1]) if len(preds) >= 2 and _is_num(preds[1]) else None

                if g is not None and c is not None:
                    blend = clamp(W_PRIMARY * g + W_CHECKER * c)
                    vals  = [g, c]
                elif g is not None:
                    blend = clamp(g); vals = [g]
                elif c is not None:
                    blend = clamp(c); vals = [c]
                else:
                    blend = 0.5; vals = [0.5]

                final = (
                    extremize_binary(blend, self.extremize_k_binary)
                    if self.extremize_enabled else blend
                )
                med        = median(vals)
                m_val      = mean(vals)
                sd         = stdev(vals)
                lo_ci, hi_ci = ci90(vals)
                stats = (
                    f"[stats] n={len(vals)} mean={m_val:.3f} median={med:.3f} "
                    f"sd={sd:.3f} ci90=({lo_ci:.3f},{hi_ci:.3f}) "
                    f"blend={blend:.3f}→final={final:.3f} "
                    f"extremize="
                    + (f"k={self.extremize_k_binary}" if self.extremize_enabled else "OFF")
                )
                return ReasonedPrediction(
                    prediction_value=final,
                    reasoning=stats + "\n\n" + joined,
                )

            # ── MULTIPLE CHOICE ───────────────────────────────────
            if isinstance(question, MultipleChoiceQuestion):
                options = list(question.options)

                def pol2dict(pol: Any) -> Dict[str, float]:
                    if isinstance(pol, PredictedOptionList):
                        return {
                            str(po.option_name).strip(): float(po.probability)
                            for po in pol.predicted_options
                            if _is_num(po.probability)
                        }
                    return {}

                gd = pol2dict(preds[0]) if len(preds) >= 1 else {}
                cd = pol2dict(preds[1]) if len(preds) >= 2 else {}

                blended: Dict[str, float] = {}
                for opt in options:
                    gv, cv = gd.get(opt), cd.get(opt)
                    if gv is not None and cv is not None:
                        blended[opt] = W_PRIMARY * gv + W_CHECKER * cv
                    elif gv is not None:
                        blended[opt] = float(gv)
                    elif cv is not None:
                        blended[opt] = float(cv)
                    else:
                        blended[opt] = 1e-6

                total = sum(blended.values())
                blended = (
                    {k: v / total for k, v in blended.items()} if total > 0
                    else {k: 1 / len(options) for k in options}
                )

                if self.extremize_enabled:
                    blended = extremize_mc(blended, self.extremize_k_mc)

                ent   = entropy_nats(blended)
                top_k = max(blended, key=blended.get)
                stats = (
                    f"[stats] n_models={len(preds)} entropy={ent:.3f} "
                    f"top={top_k}={blended[top_k]:.3f} "
                    f"extremize="
                    + (f"k={self.extremize_k_mc}" if self.extremize_enabled else "OFF")
                )
                return ReasonedPrediction(
                    prediction_value=PredictedOptionList(
                        predicted_options=[
                            PredictedOption(option_name=opt, probability=float(p))
                            for opt, p in blended.items()
                        ]
                    ),
                    reasoning=stats + "\n\n" + joined,
                )

            # ── NUMERIC ───────────────────────────────────────────
            if isinstance(question, NumericQuestion):
                targets = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

                def dist2map(d: Any) -> Dict[float, float]:
                    if not isinstance(d, NumericDistribution):
                        return {}
                    return {
                        normalize_percentile(getattr(item, "percentile", None)):
                        float(safe_float(getattr(item, "value", None), 0.0))
                        for item in d.declared_percentiles
                        if safe_float(getattr(item, "value", None)) is not None
                    }

                gm = dist2map(preds[0]) if len(preds) >= 1 else {}
                cm = dist2map(preds[1]) if len(preds) >= 2 else {}

                blended_pts: List[Percentile] = []
                for pt in targets:
                    gv = (min(gm.items(), key=lambda kv: abs(kv[0] - pt))[1]
                          if gm else None)
                    cv = (min(cm.items(), key=lambda kv: abs(kv[0] - pt))[1]
                          if cm else None)
                    if gv is not None and cv is not None:
                        v = W_PRIMARY * gv + W_CHECKER * cv
                    elif gv is not None:
                        v = gv
                    elif cv is not None:
                        v = cv
                    else:
                        try:
                            lb = float(question.lower_bound or 0.0)
                            ub = float(question.upper_bound or 100.0)
                        except Exception:
                            lb, ub = 0.0, 100.0
                        v = (lb + ub) / 2.0
                    blended_pts.append(Percentile(percentile=pt, value=float(v)))

                blended_pts.sort(key=lambda x: x.percentile)
                for i in range(1, len(blended_pts)):
                    if blended_pts[i].value < blended_pts[i - 1].value:
                        blended_pts[i].value = blended_pts[i - 1].value

                p10 = next(
                    (p.value for p in blended_pts if abs(p.percentile - 0.1) < 1e-9),
                    None,
                )
                p90 = next(
                    (p.value for p in blended_pts if abs(p.percentile - 0.9) < 1e-9),
                    None,
                )
                spread = (
                    (p90 - p10) if (p10 is not None and p90 is not None)
                    else float("nan")
                )
                stats = (
                    f"[stats] n_models={len(preds)} "
                    f"p10={p10:.3f} p90={p90:.3f} spread={spread:.3f} "
                    f"extremize=OFF (numeric — distributional integrity preserved)"
                )
                return ReasonedPrediction(
                    prediction_value=NumericDistribution.from_question(
                        blended_pts, question
                    ),
                    reasoning=stats + "\n\n" + joined,
                )

            return ReasonedPrediction(prediction_value=preds[0], reasoning=joined)

    def log_internal_drop_stats(self) -> None:
        if not self._drop:
            return
        logger.info(f"[mewhisk drops] totals={self._drop}")
        logger.info(f"[mewhisk drops] by_model={self._drop_by_model}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("litellm").propagate = False

    cli = argparse.ArgumentParser(
        description="mewhisk — Gemini forecaster + OpenRouter free utility roles"
    )
    cli.add_argument(
        "--tournament-ids", nargs="+", type=str,
        default=["minibench", "32916", "market-pulse-26q1", "ACX2026"],
    )
    cli.add_argument(
        "--no-extremize", action="store_true",
        help="Disable extremization (default: ON, conservative)",
    )
    cli.add_argument(
        "--extremize-k-binary", type=float, default=EXTREMIZE_K_BINARY_DEFAULT,
        help=f"Binary logit-scale factor (default {EXTREMIZE_K_BINARY_DEFAULT})",
    )
    cli.add_argument(
        "--extremize-k-mc", type=float, default=EXTREMIZE_K_MC_DEFAULT,
        help=f"MC power factor (default {EXTREMIZE_K_MC_DEFAULT})",
    )
    args = cli.parse_args()

    # Guard rails
    missing = []
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY  →  aistudio.google.com/app/apikey")
    if not os.getenv("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY  →  openrouter.ai/keys (free tier is fine)")
    if not EXA_ENABLED and not LINKUP_ENABLED:
        missing.append("EXA_API_KEY and/or LINKUP_API_KEY")
    if missing:
        for m in missing:
            logger.error(f"❌ Missing: {m}")
        raise SystemExit(1)

    bot = mewhisk(
        research_reports_per_question=1,
        predictions_per_research_report=2,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        extremize_enabled=not args.no_extremize,
        extremize_k_binary=args.extremize_k_binary,
        extremize_k_mc=args.extremize_k_mc,
    )

    async def run_all():
        reports = []
        for tid in args.tournament_ids:
            logger.info(f"▶️  mewhisk → tournament: {tid}")
            reports.extend(
                await bot.forecast_on_tournament(tid, return_exceptions=True)
            )
        return reports

    try:
        reports = asyncio.run(run_all())
        bot.log_report_summary(reports)
        bot.log_internal_drop_stats()
        logger.info("✅ mewhisk run complete.")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise SystemExit(1)
