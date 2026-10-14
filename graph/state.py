"""
graph/state.py
==============
THE SHARED STATE — The backbone of the entire system.

Every agent in this pipeline reads from and writes to this one object.
LangGraph passes it between agents automatically — no manual wiring needed.

Think of it like a shared research folder:
  - Orchestrator    adds the research plan (subtasks)
  - Researcher      adds raw web search results
  - Analyst         adds structured insights
  - [HUMAN REVIEW]  approves or rejects the insights
  - Writer          adds the draft report
  - Fact-Checker    adds verified claims + final report

How to explain to HR:
  "All 5 agents share a single typed state object called AppState.
   It acts as a contract — every agent knows exactly which fields
   it reads and which it writes. LangGraph manages state transitions
   automatically. This is the core pattern behind production
   agentic systems at companies like Anthropic and Google."
"""

from typing import TypedDict, List, Optional


class SubtaskResult(TypedDict):
    """One research subtask + its raw search results."""
    subtask:  str          # e.g. "What are the latest LLM benchmarks?"
    sources:  List[dict]   # list of {url, title, content} from Tavily


class FactCheckResult(TypedDict):
    """One claim + whether the fact-checker verified it."""
    claim:    str          # e.g. "GPT-4 scored 90% on MMLU"
    verdict:  str          # "VERIFIED", "UNVERIFIED", or "DISPUTED"
    source:   str          # URL that supports/disputes the claim


class AppState(TypedDict):
    # ── INPUT ──────────────────────────────────────────────────────────────
    topic: str                          # user's research topic

    # ── ORCHESTRATOR OUTPUT ────────────────────────────────────────────────
    subtasks: List[str]                 # 4-6 focused research subtasks

    # ── RESEARCHER OUTPUT ──────────────────────────────────────────────────
    raw_research: List[SubtaskResult]   # web search results per subtask

    # ── ANALYST OUTPUT ────────────────────────────────────────────────────
    structured_insights: List[str]      # key insights extracted per subtask

    # ── HUMAN-IN-THE-LOOP ─────────────────────────────────────────────────
    # The pipeline PAUSES here — human reviews insights before report is written
    human_approved:  bool               # True = proceed, False = revise
    human_feedback:  Optional[str]      # optional notes from the human reviewer

    # ── WRITER OUTPUT ─────────────────────────────────────────────────────
    draft_report: str                   # structured markdown report

    # ── FACT-CHECKER OUTPUT ───────────────────────────────────────────────
    fact_check_results: List[FactCheckResult]  # per-claim verification
    final_report:       str             # report after fact-checking
    facts_verified:     int             # count of VERIFIED claims
    facts_disputed:     int             # count of DISPUTED claims
