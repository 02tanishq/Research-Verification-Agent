"""
agents/analyst.py
=================
AGENT 3 — The Analyst

WHAT IT DOES:
  Takes the raw web search results from the Researcher and extracts
  clean, structured insights for each subtask. Removes fluff,
  pulls out facts, numbers, and key findings.

WHY IT EXISTS:
  Raw search results are messy — HTML fragments, repeated text,
  irrelevant content. The Analyst acts as a filter and synthesiser.
  It gives the Writer clean, structured material to work with.

HOW TO EXPLAIN TO HR:
  "The Analyst takes raw scraped content — which is messy and
   unstructured — and extracts 3 to 5 clean, specific insights
   per subtask. It's the difference between handing the Writer a
   stack of unread articles versus a highlighted summary. This
   separation of concerns also means I can swap the Analyst's
   prompt independently without touching any other agent."

Input  → state["raw_research"]
Output → state["structured_insights"]
"""

import os
import time
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AppState
from graph.config import get_secret, invoke_with_retry
from graph.cache import cache_get, cache_set, TTL_INSIGHTS

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=get_secret("GROQ_API_KEY"),
    max_tokens=500,
)


def _combine_sources(sources: list[dict]) -> str:
    """
    Combine multiple source contents into one text block for the LLM.
    Limits each source to 600 chars to stay within context limits.
    """
    parts = []
    for s in sources:
        parts.append(f"Source: {s['url']}\n{s['content'][:600]}")
    return "\n\n---\n\n".join(parts)


def _extract_insights_for_subtask(subtask: str, sources: list[dict]) -> str:
    """
    Ask the LLM to extract key insights from the sources for one subtask.
    Returns a formatted string of bullet-point insights.
    """

    # Cache: same subtask + same source contents → same insights.
    # Key hashes the full source payloads, so fresh search results
    # automatically bypass stale insights.
    cache_payload = {"subtask": subtask, "sources": sources}
    hit, cached_insight = cache_get("insights", cache_payload, TTL_INSIGHTS)
    if hit:
        return cached_insight

    combined_content = _combine_sources(sources)

    response = invoke_with_retry(llm, [
        SystemMessage(content="""You are a research analyst.
Extract exactly 3-5 specific, factual insights from the sources below.

Rules:
- Use real numbers and statistics when available
- Be specific — no vague statements like "many companies"
- Each insight must be a single clear sentence
- Return as a numbered list

No preamble. Just the numbered insights."""),

        HumanMessage(content=f"""
Research Question: {subtask}

Sources:
{combined_content}

Extract 3-5 key insights:""")
    ])

    insight = f"### {subtask}\n{response.content}"
    cache_set("insights", cache_payload, insight)
    return insight


def analyst_agent(state: AppState) -> dict:
    """
    MAIN FUNCTION — called by LangGraph as a node.

    For each subtask, extracts structured insights from raw sources.

    Reads:  state["raw_research"]
    Writes: state["structured_insights"]
    """

    insights = []

    for idx, research_item in enumerate(state["raw_research"]):
        # small stagger to avoid burst OTPM 1000 on free tier (5 parallel would exceed)
        if idx > 0:
            time.sleep(1.2)
        insight = _extract_insights_for_subtask(
            subtask = research_item["subtask"],
            sources = research_item["sources"],
        )
        insights.append(insight)

    return {"structured_insights": insights}
