"""
graph/cache.py
==============
PERSISTENT TTL CACHE — saves Groq/Tavily quota on the free tier.

WHY IT EXISTS:
  The pipeline makes ~18 paid API calls per topic (1 orchestrator + 5 Tavily
  + 5 analyst + 1 writer + 1 claim-extract + up to 10 verifications).
  Free-tier limits (30 RPM / 1K RPD / OTPM 1000) make repeat topics expensive
  and slow. Every LLM call here is deterministic (temperature 0, except the
  writer at 0.2), so identical inputs produce identical outputs — safe to cache.

STRATEGY:
  - Disk-backed JSON files in `.cache/` (survives server reloads, unlike
    MemorySaver). Git-ignored.
  - Per-namespace TTLs: slow-changing data (subtasks, insights, verdicts)
    lives 7 days; web results (Tavily) live 24h for freshness.
  - Keys are SHA-256 hashes of normalized inputs (topics lowercased/stripped,
    sources hashed by content) — key size stays constant.
  - Thread-safe via RLock (Researcher and Fact-Checker run threads).
  - Kill-switch: CACHE_ENABLED=0 disables all caching (e.g. for evals).

WHAT IS CACHED WHERE:
  orchestrator  → subtasks by topic                    (7 days)
  researcher    → Tavily sources by subtask query       (24 hours)
  analyst       → insights by subtask + sources content (7 days)
  writer        → draft by topic + insights + feedback  (7 days)
  fact_checker  → claims by report hash / verdicts by
                  claim + sources hash                 (7 days)
"""

import hashlib
import json
import os
import threading
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── Config (overridable via env) ──────────────────────────────────────────────
CACHE_DIR = os.getenv(
    "CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache"),
)
ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"

DAY = 24 * 3600
TTL_SUBTASKS = int(os.getenv("CACHE_TTL_SUBTASKS", str(7 * DAY)))
TTL_TAVILY = int(os.getenv("CACHE_TTL_TAVILY", str(1 * DAY)))
TTL_INSIGHTS = int(os.getenv("CACHE_TTL_INSIGHTS", str(7 * DAY)))
TTL_DRAFT = int(os.getenv("CACHE_TTL_DRAFT", str(7 * DAY)))
TTL_FACTCHECK = int(os.getenv("CACHE_TTL_FACTCHECK", str(7 * DAY)))

_lock = threading.RLock()
_stats = {"hits": 0, "misses": 0}


def _path(namespace: str, payload: dict) -> str:
    """Deterministic file path for a namespace + JSON-serializable payload."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    safe_ns = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
    return os.path.join(CACHE_DIR, f"{safe_ns}_{digest}.json")


def cache_get(namespace: str, payload: dict, ttl_seconds: int):
    """
    Returns (True, value) on fresh hit, (False, None) on miss/expiry/disabled.
    Expired entries are deleted lazily on read.
    """
    if not ENABLED:
        return False, None
    path = _path(namespace, payload)
    with _lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except (OSError, ValueError):
            _stats["misses"] += 1
            return False, None
        if time.time() - entry.get("ts", 0) > ttl_seconds:
            try:
                os.remove(path)
            except OSError:
                pass
            _stats["misses"] += 1
            return False, None
        _stats["hits"] += 1
        return True, entry.get("value")


def cache_set(namespace: str, payload: dict, value) -> None:
    """Persist a value. Failures never break the pipeline (cache is best-effort)."""
    if not ENABLED:
        return
    path = _path(namespace, payload)
    with _lock:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"ts": time.time(), "value": value}, f, ensure_ascii=True)
            os.replace(tmp, path)
        except (OSError, TypeError, ValueError):
            pass


def cache_stats() -> dict:
    """Snapshot of hit/miss counters (for logs / test_run output)."""
    with _lock:
        return dict(_stats)


def normalize_topic(topic: str) -> str:
    """Canonical cache key for topics: case/whitespace-insensitive."""
    return " ".join((topic or "").strip().lower().split())
