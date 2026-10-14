"""
graph/workflow.py
=================
THE LANGGRAPH WORKFLOW — Wires all 5 agents + HITL gate

FULL PIPELINE FLOW:
  START
    │
    ▼
  Orchestrator      ← breaks topic into 5 subtasks
    │
    ▼
  Researcher        ← parallel Tavily searches (one per subtask)
    │
    ▼
  Analyst           ← extracts structured insights from raw results
    │
    ▼
  INTERRUPT ←──── PIPELINE PAUSES HERE
  Human Review      ← human approves/rejects insights in Streamlit UI
    │
    ▼ (if approved)
  Writer            ← compiles professional report from insights
    │
    ▼
  Fact Checker      ← validates every claim against sources
    │
    ▼
  END

THE HUMAN-IN-THE-LOOP PATTERN:
  LangGraph supports HITL via interrupt_before=[node_name].
  When the graph hits that node, execution PAUSES.
  The full state is saved by MemorySaver.
  The Streamlit UI shows the insights to the user.
  When the user clicks Approve, the app updates state
  (human_approved=True) and calls graph.invoke() again.
  LangGraph resumes from exactly where it paused.

HOW TO EXPLAIN TO HR:
  "The pipeline uses LangGraph's interrupt_before mechanism to pause
   execution after the Analyst runs. The state is fully preserved —
   nothing is lost. When the human approves, the graph resumes from
   the exact checkpoint. This is the same pattern used in Anthropic's
   Claude and other production AI systems that require human oversight
   at critical decision points."
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AppState
from agents.orchestrator  import orchestrator_agent
from agents.researcher    import researcher_agent
from agents.analyst       import analyst_agent
from agents.human_review  import human_review_node
from agents.writer        import writer_agent
from agents.fact_checker  import fact_checker_agent


def _should_continue_after_review(state: AppState) -> str:
    """
    Conditional edge after human review.

    If human approved → continue to Writer
    If not approved   → end the pipeline
    """
    if state.get("human_approved", False):
        return "approved"
    return "rejected"


def build_graph():
    """
    Builds and compiles the research report workflow.

    Key design decisions:
    1. interrupt_before=["human_review"] — pauses before human review node
    2. Conditional edge after review — approved or rejected
    3. MemorySaver — preserves state across the interrupt
    """

    graph = StateGraph(AppState)

    # ── Register all nodes ────────────────────────────────────────────────
    graph.add_node("orchestrator",  orchestrator_agent)
    graph.add_node("researcher",    researcher_agent)
    graph.add_node("analyst",       analyst_agent)
    graph.add_node("human_review",  human_review_node)
    graph.add_node("writer",        writer_agent)
    graph.add_node("fact_checker",  fact_checker_agent)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("orchestrator")

    # ── Linear edges (no branching) ───────────────────────────────────────
    graph.add_edge("orchestrator", "researcher")
    graph.add_edge("researcher",   "analyst")
    graph.add_edge("analyst",      "human_review")   # ← PAUSES before this

    # ── Conditional edge after human review ───────────────────────────────
    graph.add_conditional_edges(
        "human_review",
        _should_continue_after_review,
        {
            "approved": "writer",    # human said yes → write the report
            "rejected": END,         # human said no  → stop here
        }
    )

    # ── Final linear edges ────────────────────────────────────────────────
    graph.add_edge("writer",       "fact_checker")
    graph.add_edge("fact_checker", END)

    # ── Memory: preserves state across the HITL interrupt ─────────────────
    # Without MemorySaver, the graph would lose all state when it pauses.
    # MemorySaver serialises the full AppState so we can resume later.
    memory = MemorySaver()

    return graph.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],  # ← THE HITL GATE
    )


# Single instance — reused by app.py, api/main.py, test_run.py
research_graph = build_graph()
