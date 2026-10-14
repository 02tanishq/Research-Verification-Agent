"""
agents/human_review.py
======================
HUMAN-IN-THE-LOOP (HITL) GATE

WHAT IT DOES:
  This is NOT a traditional LLM agent — it's a special node that
  PAUSES the pipeline and waits for a human to review the Analyst's
  insights before the Writer and Fact-Checker run.

  The human can:
    APPROVE  → pipeline continues to Writer
    REJECT   → pipeline stops (user can restart with a new topic)
    FEEDBACK → add optional notes the Writer will incorporate

WHY THIS IS IMPRESSIVE FOR 10-12 LPA:
  Human-in-the-loop is a PRODUCTION AI pattern. It's how real AI
  systems at Google, Anthropic, and enterprises are built — not
  fully automated, but with human oversight at critical checkpoints.

  Most tutorial projects are fully automated.
  Adding HITL shows you understand:
  - AI systems need human oversight
  - Not every decision should be automated
  - How to pause and resume stateful workflows

HOW TO EXPLAIN TO HR:
  "After the Analyst extracts insights, the pipeline pauses and shows
   them to the user for review. Using LangGraph's interrupt mechanism,
   the graph state is preserved — nothing is lost. When the user
   approves, the graph resumes exactly where it paused. This is the
   Human-in-the-Loop pattern used in production AI pipelines at
   Anthropic and other AI companies."

TECHNICAL DETAIL:
  LangGraph supports HITL via interrupt_before=[node_name].
  When the graph hits that node, it pauses and returns control.
  The calling code then resumes the graph by invoking it again
  with updated state (human_approved=True/False, human_feedback=...).

Input  → state["structured_insights"]
Output → state["human_approved"], state["human_feedback"]
"""

from graph.state import AppState


def human_review_node(state: AppState) -> dict:
    """
    MAIN FUNCTION — called by LangGraph as a node.

    In practice, this node is reached AFTER the graph is interrupted.
    The Streamlit UI handles showing insights to the user and
    collecting their approval/feedback, then resumes the graph
    with human_approved set to True or False.

    This function just passes the approval state through.
    The real HITL logic happens in the graph's interrupt mechanism.

    Reads:  state["human_approved"], state["human_feedback"]
    Writes: nothing new — just validates the state is set
    """

    # This node acts as a pass-through after the human has reviewed.
    # The interrupt happens BEFORE this node runs (configured in workflow.py).
    # By the time this function executes, human_approved is already set.

    return {
        "human_approved": state.get("human_approved", False),
        "human_feedback": state.get("human_feedback", ""),
    }
