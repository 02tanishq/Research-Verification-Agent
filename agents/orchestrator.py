"""
agents/orchestrator.py
======================
AGENT 1 — The Orchestrator

WHAT IT DOES:
  Takes the user's research topic and breaks it into 4-6 focused
  subtasks. Every other agent works from these subtasks.

WHY IT EXISTS:
  A single search query like "AI in healthcare" returns generic results.
  But breaking it into focused questions like:
    - "What are the latest FDA-approved AI diagnostic tools?"
    - "How is AI reducing hospital readmission rates?"
  ...gives specific, deep, cross-referenced research.

  This is exactly what a research team lead does before assigning work.

HOW TO EXPLAIN TO HR:
  "The Orchestrator is the team lead. It reads the topic and creates
   a research plan — 4 to 6 specific subtasks. Every downstream agent
   works from this plan. This mirrors how real research projects work:
   you don't research everything at once, you decompose the problem first."

Input  → state["topic"]
Output → state["subtasks"]  (list of 4-6 research questions)
"""

import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AppState
from graph.config import get_secret, invoke_with_retry
from graph.cache import cache_get, cache_set, TTL_SUBTASKS, normalize_topic

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,                    # deterministic — same topic = same plan
    api_key=get_secret("GROQ_API_KEY"),
    max_tokens=600,
)


def _build_prompt(topic: str) -> str:
    """Build the prompt asking the LLM to decompose the topic."""
    return f"""Research Topic: {topic}

Break this topic into exactly 5 focused research subtasks.
Each subtask should be a specific question that, when answered,
contributes to a complete understanding of the main topic.

Return ONLY a valid JSON array of 5 strings.
No explanation. No markdown. Just the JSON array.

Example format:
["subtask 1?", "subtask 2?", "subtask 3?", "subtask 4?", "subtask 5?"]"""


def _parse_subtasks(raw_response: str) -> list[str]:
    """
    Safely parse the LLM response into a list of subtasks.
    Strips markdown fences if the LLM added them.
    """
    cleaned = raw_response.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    subtasks = json.loads(cleaned)
    return subtasks[:6]   # max 6 subtasks


def orchestrator_agent(state: AppState) -> dict:
    """
    MAIN FUNCTION — called by LangGraph as a node.

    Reads:  state["topic"]
    Writes: state["subtasks"]
    """

    # Cache: deterministic planning (temperature 0) → same topic = same plan.
    topic = state.get("topic", "") or ""
    cache_payload = {"topic": normalize_topic(topic)}
    hit, cached_subtasks = cache_get("subtasks", cache_payload, TTL_SUBTASKS)
    if hit:
        return {"subtasks": cached_subtasks}

    # Step 1: ask the LLM to break the topic into subtasks
    response = invoke_with_retry(llm, [
        SystemMessage(content="You are a research director. Decompose topics into focused subtasks."),
        HumanMessage(content=_build_prompt(topic)),
    ])

    # Step 2: parse the response
    subtasks = _parse_subtasks(response.content)

    cache_set("subtasks", cache_payload, subtasks)
    return {"subtasks": subtasks}
