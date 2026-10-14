"""
api/main.py
===========
FASTAPI REST API — Exposes the research pipeline as a deployable service.

IMPRESSIVE FEATURE FOR 10-12 LPA:
  Not just a Streamlit demo — a proper REST API that any frontend,
  mobile app, or third-party service can call.

  Three endpoints:
    POST /research/start      ← start a new research session
    POST /research/approve    ← human approves insights (HITL gate)
    GET  /research/{id}       ← get current state of a session

HOW TO EXPLAIN TO HR:
  "I wrapped the entire agent pipeline in a FastAPI service with
   three endpoints. /start kicks off the pipeline and returns a
   session ID when it pauses at the HITL gate. /approve resumes
   the pipeline with the human's decision. /status lets you poll
   the session state. This means the system is not just a demo —
   it's a deployable microservice any product can integrate with."

USAGE:
  uvicorn api.main:app --reload
  Then open: http://localhost:8000/docs
"""

import uuid

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from graph.workflow import research_graph

app = FastAPI(
    title="Multi-Agent Research & Report System",
    description="4 AI agents that research any topic and deliver a fact-checked report.",
    version="1.0.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class StartRequest(BaseModel):
    topic: str                    # research topic

class ApproveRequest(BaseModel):
    session_id: str               # from /start response
    approved:   bool              # True = continue, False = reject
    feedback:   Optional[str] = ""  # optional reviewer notes

class SessionResponse(BaseModel):
    session_id:  str
    status:      str              # "researching", "awaiting_review", "complete", "rejected"
    subtasks:    list = []
    insights:    list = []
    final_report: str = ""
    facts_verified: int = 0
    facts_disputed: int = 0


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_config(session_id: str) -> dict:
    """Build the LangGraph config dict for a session."""
    return {"configurable": {"thread_id": session_id}}

def _get_state(session_id: str) -> dict:
    """Get current state values for a session."""
    config = _get_config(session_id)
    return research_graph.get_state(config).values


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/research/start", response_model=SessionResponse)
async def start_research(req: StartRequest):
    """
    Start a new research session.

    The pipeline runs until the HITL gate (after Analyst).
    Returns the session_id and structured insights for human review.
    """

    session_id = str(uuid.uuid4())
    config     = _get_config(session_id)

    # Run the graph — it will pause at interrupt_before=["human_review"]
    research_graph.invoke(
        {"topic": req.topic},
        config=config,
    )

    state = _get_state(session_id)

    return SessionResponse(
        session_id = session_id,
        status     = "awaiting_review",
        subtasks   = state.get("subtasks", []),
        insights   = state.get("structured_insights", []),
    )


@app.post("/research/approve", response_model=SessionResponse)
async def approve_research(req: ApproveRequest):
    """
    Resume the pipeline after human review.

    If approved=True, the Writer and Fact-Checker run and produce the report.
    If approved=False, the pipeline ends without a report.
    """

    config = _get_config(req.session_id)

    # Check session exists (MemorySaver is in-memory: a server reload wipes checkpoints)
    current = research_graph.get_state(config)
    if not current.values:
        raise HTTPException(status_code=404, detail="Session not found (server may have reloaded). Start a new session via POST /research/start.")
    if not current.values.get("topic") or not current.values.get("structured_insights"):
        raise HTTPException(status_code=400, detail="Session checkpoint is incomplete (missing topic/insights after reload). Start a new session via POST /research/start.")

    # Resume the graph — update state then stream from checkpoint (matches Streamlit app.py pattern)
    try:
        research_graph.update_state(
            config,
            {"human_approved": req.approved, "human_feedback": req.feedback or ""},
            as_node="human_review",
        )
        for _ in research_graph.stream(None, config=config):
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    state = _get_state(req.session_id)

    if state.get("final_report"):
        status = "complete"
    elif not req.approved:
        status = "rejected"
    else:
        status = "awaiting_review" if state.get("structured_insights") else "rejected"

    return SessionResponse(
        session_id     = req.session_id,
        status         = status,
        subtasks       = state.get("subtasks", []),
        insights       = state.get("structured_insights", []),
        final_report   = state.get("final_report", ""),
        facts_verified = state.get("facts_verified", 0),
        facts_disputed = state.get("facts_disputed", 0),
    )


@app.get("/research/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get the current state of any research session by ID."""

    state = _get_state(session_id)

    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")

    if state.get("final_report"):
        status = "complete"
    elif state.get("structured_insights"):
        status = "awaiting_review"
    elif state.get("subtasks"):
        status = "researching"
    else:
        status = "unknown"

    return SessionResponse(
        session_id     = session_id,
        status         = status,
        subtasks       = state.get("subtasks", []),
        insights       = state.get("structured_insights", []),
        final_report   = state.get("final_report", ""),
        facts_verified = state.get("facts_verified", 0),
        facts_disputed = state.get("facts_disputed", 0),
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "system": "Multi-Agent Research & Report System"}
