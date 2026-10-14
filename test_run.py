"""
test_run.py
===========
Terminal test — confirms the full pipeline works end-to-end.
Auto-approves the HITL gate so you don't need to click anything.

Usage:  python test_run.py
"""

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from graph.workflow import research_graph
from graph.cache import cache_stats

TOPIC = "Impact of Generative AI on software engineering jobs in India 2025"

print("\n" + "="*65)
print("  Multi-Agent Research & Report System — Test Run")
print(f"  Topic: {TOPIC}")
print("="*65 + "\n")

config = {"configurable": {"thread_id": "test-session-001"}}

# ── Phase 1: Run until HITL gate ──────────────────────────────────────────────
print("PHASE 1 — Running until Human Review gate...\n")

for event in research_graph.stream({"topic": TOPIC}, config=config):
    node = list(event.keys())[0]
    print(f"  [OK]  {node.upper()}")

# Pipeline paused at HITL gate
state = research_graph.get_state(config).values

print(f"\n  Subtasks generated: {len(state.get('subtasks', []))}")
print(f"  Insights ready:     {len(state.get('structured_insights', []))}")

print("\n" + "-"*65)
print("  [AUTO-APPROVING for test — in UI the human reviews here]")
print("-"*65 + "\n")

# ── Phase 2: Resume with approval ─────────────────────────────────────────────
print("PHASE 2 — Resuming with approval...\n")

research_graph.update_state(config, {"human_approved": True, "human_feedback": ""}, as_node="human_review")
for event in research_graph.stream(None, config=config):
    node = list(event.keys())[0]
    print(f"  [OK]  {node.upper()}")

# ── Final results ─────────────────────────────────────────────────────────────
final = research_graph.get_state(config).values

print("\n" + "="*65)
print("  RESULTS")
print("="*65)
print(f"  Facts verified : {final.get('facts_verified', 0)}")
print(f"  Facts disputed : {final.get('facts_disputed', 0)}")
print(f"  Total claims   : {len(final.get('fact_check_results', []))}")

print("\n  REPORT PREVIEW (first 600 chars):")
print("-"*65)
print(final.get("final_report", "")[:600])
print("\n... full report in Streamlit UI\n")

stats = cache_stats()
print(f"  Cache: {stats.get('hits', 0)} hits / {stats.get('misses', 0)} misses (.cache/)\n")
