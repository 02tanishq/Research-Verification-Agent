# Autonomous Research & Verification Agent

> **5 AI Agents · LangGraph · Human-in-the-Loop · Fact-Checking · FastAPI · Groq Qwen 3.8 27B**

**GitHub:**(https://github.com/02tanishq/Research-Verification-Agent)-

---

## What it does

Enter any research topic -> 5 specialised agents collaborate to deliver a structured,
fact-checked report with a human approval gate before the report is written.
Download the final report as a **styled PDF**.

```
Topic Input
  |
  v
Orchestrator    ->  Breaks topic into 5 focused research subtasks
  |
  v
Researcher      ->  Parallel Tavily web search (5 queries simultaneously)
  |
  v
Analyst         ->  Extracts 3-5 specific insights per subtask
  |
  v
HUMAN REVIEW    ->  You review insights before the report is written
  |
  v (approved)
Writer          ->  Compiles professional structured markdown report
  |
  v
Fact Checker    ->  Validates every claim: VERIFIED / UNVERIFIED / DISPUTED
  |
  v
Final Report + PDF Download + Fact-Check Score
```

---

## Key Features

| Feature | Why it matters |
|---------|----------------|
| Human-in-the-Loop gate | Pipeline pauses; full state preserved by LangGraph MemorySaver |
| Parallel research | All 5 Tavily searches run simultaneously via ThreadPoolExecutor |
| Parallel fact-checking | Up to 10 claim verifications run concurrently (3 workers, order-preserving) |
| Fact-Checker agent | Validates every claim - addresses LLM hallucination directly |
| API cache | Disk-backed TTL cache (`graph/cache.py`): repeat topics cost ~0 calls instead of ~18 |
| Rate-limit retries | Exponential backoff on Groq 429s; per-agent token caps for the free tier |
| PDF export | Styled PDF download with professional typography |
| FastAPI REST endpoint | Full deployable microservice: `/research/start`, `/research/approve`, `/research/{id}`, `/health` |
| Typed shared state | AppState TypedDict - every agent I/O contract is explicit |

---

## Tech Stack

- **LangGraph** - Agent orchestration + HITL interrupt + MemorySaver checkpointing
- **Groq Qwen 3.8 27B** - LLM inference for all 4 LLM agents (via `langchain-groq`)
- **Tavily** - Real-time web search API (`advanced` depth, 3 results per subtask)
- **FastAPI** - Production REST API layer
- **Streamlit** - Interactive UI with 3-phase workflow
- **fpdf2** - Markdown to styled PDF export

---

## Quick Start

```bash
git clone https://github.com/02tanishq/Research-Verification-Agent.git
cd Autonomous-Research-Verification-Agent-
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY + TAVILY_API_KEY
streamlit run app.py        # launch Streamlit UI
uvicorn api.main:app --reload  # launch FastAPI (separate terminal)
```

---

## Notes

- **Cache:** repeat topics are served from `.cache/` (git-ignored). Disable with
  `CACHE_ENABLED=0`; tune TTLs with `CACHE_TTL_TAVILY`, `CACHE_TTL_SUBTASKS`,
  `CACHE_TTL_INSIGHTS`, `CACHE_TTL_DRAFT`, `CACHE_TTL_FACTCHECK`.
- **Sessions are in-memory:** `MemorySaver` checkpoints vanish on server reload,
  so don't edit code between Start and Approve. If a session expires, just
  start a new research session.
- **Free-tier limits (Groq):** 30 RPM / 1K requests/day / OTPM 1000 per org.
  The pipeline retries 429s automatically, but heavy use may still throttle.
- **Docs:** `docs/ARCHITECTURE.md` (workflow, caching, resume pattern),
  `docs/SETUP.md` (setup guide). Terminal test: `python test_run.py`.
