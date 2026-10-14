"""
agents/researcher.py
====================
AGENT 2 — The Researcher

WHAT IT DOES:
  Takes the 4-6 subtasks from the Orchestrator and searches the web
  for each one using Tavily. All searches run in PARALLEL — so 5
  searches take the same time as 1.

IMPRESSIVE FEATURE — PARALLEL EXECUTION:
  Uses Python's ThreadPoolExecutor to run all Tavily searches
  at the same time. This is a key performance pattern in
  production AI systems.

  Sequential  → 5 searches × 3s each = 15 seconds
  Parallel    → 5 searches run together = ~3 seconds

HOW TO EXPLAIN TO HR:
  "The Researcher searches the web for each subtask in parallel
   using ThreadPoolExecutor. This cuts research time from 15 seconds
   to 3 seconds. In production systems, I/O-bound tasks like API calls
   should always run concurrently — this is a standard engineering
   pattern that shows I think about performance, not just correctness."

Input  → state["subtasks"]
Output → state["raw_research"]  (list of {subtask, sources})
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tavily import TavilyClient
from graph.state import AppState, SubtaskResult
from graph.config import get_secret
from graph.cache import cache_get, cache_set, TTL_TAVILY, normalize_topic

# ── Tavily client setup ────────────────────────────────────────────────────────
tavily = TavilyClient(api_key=get_secret("TAVILY_API_KEY"))


def _search_one_subtask(subtask: str) -> SubtaskResult:
    """
    Search the web for a single subtask.

    Returns a SubtaskResult containing:
    - The subtask question
    - Up to 3 sources with url, title, content
    """

    # Cache: same subtask question → same web results (24h TTL for freshness).
    # Key is normalized; the ORIGINAL subtask string is kept in the result so
    # downstream ordering (by original subtask) is unaffected.
    cache_payload = {"q": normalize_topic(subtask), "n": 3}
    hit, cached_sources = cache_get("tavily", cache_payload, TTL_TAVILY)
    if hit:
        return SubtaskResult(subtask=subtask, sources=cached_sources)

    # Tavily search — advanced mode fetches full page content
    response = tavily.search(
        query=subtask,
        max_results=3,
        search_depth="advanced",
    )

    # Extract only the fields we need
    sources = [
        {
            "url":     result.get("url", ""),
            "title":   result.get("title", ""),
            "content": result.get("content", "")[:800],  # first 800 chars
        }
        for result in response.get("results", [])
    ]

    cache_set("tavily", cache_payload, sources)
    return SubtaskResult(subtask=subtask, sources=sources)


def _search_all_subtasks_in_parallel(subtasks: list[str]) -> list[SubtaskResult]:
    """
    Run all subtask searches at the same time using threads.

    ThreadPoolExecutor creates a pool of worker threads.
    Each thread handles one subtask search independently.
    Results are collected as they complete.
    """

    results = []

    # max_workers=5 means up to 5 searches run simultaneously
    with ThreadPoolExecutor(max_workers=5) as executor:

        # Submit all searches to the thread pool
        future_to_subtask = {
            executor.submit(_search_one_subtask, subtask): subtask
            for subtask in subtasks
        }

        # Collect results as each search completes
        for future in as_completed(future_to_subtask):
            result = future.result()
            results.append(result)

    # Sort by original subtask order for consistency
    subtask_order = {s: i for i, s in enumerate(subtasks)}
    results.sort(key=lambda r: subtask_order.get(r["subtask"], 0))

    return results


def researcher_agent(state: AppState) -> dict:
    """
    MAIN FUNCTION — called by LangGraph as a node.

    Reads:  state["subtasks"]
    Writes: state["raw_research"]
    """

    # Run all searches in parallel
    raw_research = _search_all_subtasks_in_parallel(state["subtasks"])

    return {"raw_research": raw_research}
