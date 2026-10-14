# Architecture

> Current as of the Qwen 3.8 migration. All 4 LLM agents use
> `qwen/qwen3.8-27b` on Groq. No emojis anywhere in the app.

## Tech Stack

| Layer                  | Choice                                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| Orchestration          | LangGraph `StateGraph` + `MemorySaver` checkpointing                                         |
| LLM (all 4 LLM agents) | `qwen/qwen3.8-27b` via `langchain_groq.ChatGroq`                                             |
| Web search             | Tavily (`advanced` depth, 3 results per subtask)                                             |
| UI                     | Streamlit (`app.py`, 3 phases: input / review / complete)                                    |
| API                    | FastAPI (`api/main.py`: `/research/start`, `/research/approve`, `/research/{id}`, `/health`) |
| Cache                  | Disk-backed TTL cache (`graph/cache.py`, `.cache/`)                                          |
| Secrets                | `.env` via `python-dotenv`, loaded in `graph/config.py` (plus Streamlit secrets fallback)    |

Per-agent token caps (Groq free-tier OTPM is 1000): orchestrator 600,
analyst 500, writer 900, fact-checker 600.

## Graph Structure

```python
graph = StateGraph(AppState)

graph.add_node("orchestrator",  orchestrator_agent)
graph.add_node("researcher",    researcher_agent)
graph.add_node("analyst",       analyst_agent)
graph.add_node("human_review",  human_review_node)
graph.add_node("writer",        writer_agent)
graph.add_node("fact_checker",  fact_checker_agent)

graph.set_entry_point("orchestrator")

graph.add_edge("orchestrator", "researcher")
graph.add_edge("researcher",   "analyst")
graph.add_edge("analyst",      "human_review")      # PAUSES before this

graph.add_conditional_edges(
    "human_review",
    _should_continue_after_review,
    {"approved": "writer", "rejected": END}
)

graph.add_edge("writer",       "fact_checker")
graph.add_edge("fact_checker", END)

graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_review"],
)
```

---

## How the HITL Interrupt Works

```
graph.stream({"topic": "..."}, config)
  │
  Orchestrator runs [done]   → 5 subtasks (cached by topic, 7-day TTL)
  Researcher runs   [done]   → 5 parallel Tavily searches (cached per query, 24h TTL)
  Analyst runs      [done]   → 5 sequential insight extractions, 1.2s stagger
  │
  interrupt_before=["human_review"] fires
  │
  stream() RETURNS — pipeline is paused
  Full AppState is saved by MemorySaver (IN-MEMORY: lost on server reload)

[ Streamlit shows insights to user ]
[ User clicks Approve ]

research_graph.update_state(config, {"human_approved": True, ...}, as_node="human_review")
research_graph.stream(None, config)   ← same config/thread_id, resumes from checkpoint
  │
  human_review runs [done]   → pass-through of approval flags
  Conditional edge: approved → writer
  Writer runs       [done]   → draft report (cached by topic+insights+feedback)
  Fact Checker runs [done]   → claims extracted + verified in parallel
  │
  END
```

The key: using the SAME `thread_id` in config both times.
LangGraph finds the saved checkpoint and resumes from it.

Resume pattern note: the app resumes with `update_state(..., as_node="human_review")`
followed by `stream(None, config)` — NOT `invoke({"human_approved": True})`.
`app.py`, `api/main.py`, and `test_run.py` all use this pattern.

Session-expiry rule: because `MemorySaver` is in-memory, any server reload
(`uvicorn --reload` on file save, Streamlit rerun from code edits) wipes all
checkpoints. Resuming afterwards fails cleanly instead of crashing:
`writer_agent` raises `ValueError: Session state is missing 'topic'...`,
`app.py` shows "Session expired... start over", and
`POST /research/approve` returns 404/400. Fix is always: start a new session.

---

## Resilience: Retries for Groq Rate Limits

All LLM calls go through `invoke_with_retry` (`graph/config.py`):
up to 3 attempts with exponential backoff on 429/RateLimit errors,
honoring the `retry-after`-style "try again in Xs" hint when present.
This is what keeps the pipeline alive on the free tier
(30 RPM / 1K RPD / 8K TPM / OTPM 1000, org-level).

---

## Parallel Research (ThreadPoolExecutor)

```python
with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_subtask = {
        executor.submit(_search_one_subtask, subtask): subtask
        for subtask in subtasks
    }
    for future in as_completed(future_to_subtask):
        results.append(future.result())
```

Each subtask search runs in its own thread simultaneously.
as_completed() yields futures as they finish — no blocking.
Results are re-sorted into original subtask order afterwards.

---

## Parallel Fact-Checking (ThreadPoolExecutor)

Claim verification uses the same pattern with `max_workers=3`
(a lower count to stay under the OTPM 1000 free-tier cap):

```python
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(
        lambda claim: _verify_one_claim(claim, all_sources),
        claims,
    ))
```

`executor.map` preserves input order, so `results[i]` still matches
`claims[i]` — counting and the summary table are unaffected.
Per-call 429s are absorbed by `invoke_with_retry` inside each worker.

---

## Fact-Checker Flow

```
draft_report
    │
    ▼
Extract claims  (1 LLM call, first 3000 chars, max 10 claims)
    │  → cached by report hash (failures NOT cached, so they retry)
    ▼
For each claim, in parallel:
    Verify against sources  (1 LLM call per claim, 5000-char source slice)
    → VERIFIED / UNVERIFIED / DISPUTED
    → each verdict cached by claim + sources hash
    │
    ▼
Append fact-check table to report ([PASS]/[UNKNOWN]/[FAIL] rows)
Compute fact-check score = verified / total × 100
```

Limits: claims are checked only against the pipeline's own Tavily sources
(not the open web), only the first 3000 report chars are mined, max 10 claims.

---

## Cache Strategy (`graph/cache.py`)

Disk-backed JSON cache in `.cache/` (git-ignored, survives reloads).
Thread-safe via `RLock`. Best-effort: failures never break the pipeline.
Disabled with `CACHE_ENABLED=0`.

| Namespace  | Key                            | TTL      | Why safe                                     |
| ---------- | ------------------------------ | -------- | -------------------------------------------- |
| `subtasks` | normalized topic               | 7 days   | planning is deterministic (temp 0)           |
| `tavily`   | normalized query               | 24 hours | web results need freshness                   |
| `insights` | subtask + full source contents | 7 days   | fresh sources hash differently → auto-bypass |
| `draft`    | topic + insights + feedback    | 7 days   | identical inputs → identical draft           |
| `claims`   | report slice                   | 7 days   | same report → same claims                    |
| `verdict`  | claim + sources slice          | 7 days   | same evidence → same verdict                 |

A repeat topic costs ~0 API calls instead of ~18
(1 orchestrator + 5 Tavily + 5 analyst + 1 writer + 1 extract + up to 10 verifications).
`test_run.py` prints `hits/misses` at the end.

---

## FastAPI Architecture

```
Client
  │
  POST /research/start   →  invoke graph → pauses at HITL → return session_id + insights
  │
  POST /research/approve →  update_state + stream(None) → return final report
  │                         404 if session unknown, 400 if checkpoint incomplete
  GET  /research/{id}    →  read current state → return status + outputs
```

All sessions are isolated by `thread_id` in LangGraph MemorySaver.
