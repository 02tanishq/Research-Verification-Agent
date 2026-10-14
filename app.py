"""
app.py
======
Streamlit UI for the Multi-Agent Research & Report System

IMPORTANT — TWO-PHASE UI:
  Phase 1: User enters topic → pipeline runs → pauses at HITL gate
           UI shows the structured insights for review

  Phase 2: User clicks Approve or Reject
           If approved → Writer + Fact-Checker run → final report shown
           If rejected → session ends
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import uuid
import re
import streamlit as st
import plotly.graph_objects as go

from graph.workflow import research_graph


# ── PDF generator (fpdf2) ────────────────────────────────────────────────────────────
def _generate_pdf(markdown_text: str) -> bytes:
    """
    Convert a markdown research report to a styled PDF using fpdf2.
    Parses headings, bullets, numbered lists, tables, and body text.
    """
    from fpdf import FPDF

    DARK_BLUE  = (15,  52,  96)
    MID_BLUE   = (22,  33,  62)
    ACCENT_RED = (233, 69,  96)
    BODY_COLOR = (30,  30,  50)

    class ReportPDF(FPDF):
        def header(self):
            old_x, old_y = self.x, self.y
            self.set_fill_color(*DARK_BLUE)
            self.rect(0, 0, 210, 12, "F")
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(255, 255, 255)
            self.set_xy(10, 2)
            self.cell(0, 8, "Multi-Agent Research & Report System")
            self.set_xy(old_x, old_y)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10,
                f"Page {self.page_no()} | Powered by LangGraph & Groq Qwen 3.8 27B",
                align="C")

    def _strip_md(text: str) -> str:
        """Remove common inline markdown tokens and sanitise for fpdf2 latin-1 fonts."""
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*",     r"\1", text)
        text = re.sub(r"`(.+?)`",        r"\1", text)
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
        # Replace common unicode chars that latin-1 can't encode
        text = text.replace("\u2014", "--").replace("\u2013", "-")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2026", "...").replace("\u00a0", " ")
        # Drop any remaining non-latin-1 chars
        text = text.encode("latin-1", errors="ignore").decode("latin-1")
        return text

    pdf = ReportPDF()
    pdf.set_margins(15, 20, 15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    first_table_row = True  # track first row of a table for header styling

    for line in markdown_text.split("\n"):
        s = line.rstrip()

        # ── H1
        if re.match(r"^# [^#]", s):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(*DARK_BLUE)
            pdf.multi_cell(0, 10, _strip_md(s[2:].strip()))
            y = pdf.get_y()
            pdf.set_draw_color(*ACCENT_RED)
            pdf.set_line_width(0.8)
            pdf.line(15, y, 195, y)
            pdf.ln(3)
            first_table_row = True

        # ── H2
        elif re.match(r"^## [^#]", s):
            pdf.ln(5)
            y = pdf.get_y()
            pdf.set_fill_color(*ACCENT_RED)
            pdf.rect(15, y, 3, 8, "F")
            pdf.set_xy(21, y)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(*MID_BLUE)
            pdf.multi_cell(0, 8, _strip_md(s[3:].strip()))
            pdf.ln(2)
            first_table_row = True

        # ── H3
        elif re.match(r"^### ", s):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*DARK_BLUE)
            pdf.multi_cell(0, 6, _strip_md(s[4:].strip()))
            pdf.ln(1)
            first_table_row = True

        # ── Horizontal rule
        elif re.match(r"^-{3,}$", s) or re.match(r"^={3,}$", s):
            pdf.ln(2)
            pdf.set_draw_color(180, 180, 200)
            pdf.set_line_width(0.3)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(3)
            first_table_row = True

        # ── Bullet
        elif re.match(r"^[-*] ", s):
            text = _strip_md(s[2:].strip())
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*BODY_COLOR)
            pdf.set_x(15)
            pdf.multi_cell(0, 6, f"  -  {text}")

        # ── Numbered list
        elif re.match(r"^\d+\. ", s):
            num   = re.match(r"^(\d+)\.", s).group(1)
            text  = _strip_md(re.sub(r"^\d+\.\s", "", s).strip())
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*BODY_COLOR)
            pdf.set_x(15)
            pdf.multi_cell(0, 6, f"  {num}.  {text}")

        # ── Table row
        elif s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            # Skip separator rows like |---|---|
            if all(re.match(r"^[-:| ]+$", c) for c in cells):
                first_table_row = False
                continue
            n_cols = len(cells)
            col_w  = min(50, int(175 / max(n_cols, 1)))
            if first_table_row:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(255, 255, 255)
                pdf.set_fill_color(*DARK_BLUE)
                fill = True
                first_table_row = False
            else:
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(*BODY_COLOR)
                pdf.set_fill_color(244, 246, 250)
                fill = True
            for cell in cells:
                pdf.cell(col_w, 7, _strip_md(cell)[:35], border=1, fill=fill)
            pdf.ln()

        # ── Blank line
        elif s == "":
            pdf.ln(2)
            first_table_row = True

        # ── Regular paragraph
        else:
            text = _strip_md(s)
            if text.strip():
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(*BODY_COLOR)
                pdf.multi_cell(0, 6, text)

    return bytes(pdf.output())




# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research System",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id"  not in st.session_state: st.session_state.session_id  = str(uuid.uuid4())
if "phase"       not in st.session_state: st.session_state.phase       = "input"   # input | review | complete
if "graph_state" not in st.session_state: st.session_state.graph_state = None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Multi-Agent Research & Report System")
st.caption("**LangGraph** · **5 AI Agents** · **Human-in-the-Loop** · **Fact-Checking** · **FastAPI** · **Groq Qwen 3.8 27B**")

# ── Pipeline diagram ──────────────────────────────────────────────────────────
with st.expander("Agent Pipeline Architecture", expanded=False):
    st.markdown("""
```
Topic Input
    │
    ▼
Orchestrator    →  Breaks topic into 5 focused research subtasks
    │
    ▼
Researcher      →  Parallel Tavily searches (5 searches simultaneously)
    │
    ▼
Analyst         →  Extracts 3-5 specific insights per subtask
    │
    ▼
HUMAN REVIEW   →  YOU review insights — Approve or Reject
    │
    ▼ (approved)
Writer          →  Compiles professional structured report
    │
    ▼
Fact Checker   →  Validates every claim → VERIFIED / UNVERIFIED / DISPUTED
    │
    ▼
Final Report
```
**All agents share one AppState object — LangGraph manages state seamlessly.**
    """)

st.divider()

config = {"configurable": {"thread_id": st.session_state.session_id}}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — INPUT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "input":

    st.subheader("Enter Research Topic")
    topic = st.text_input(
        "What do you want to research?",
        placeholder="e.g.  Impact of Generative AI on software engineering jobs in India 2025",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run = st.button("Start Research", type="primary", disabled=not topic.strip())
    with col2:
        st.caption("The pipeline runs Orchestrator → Researcher → Analyst, then pauses for your review.")

    if run and topic.strip():
        node_labels = {
            "orchestrator": ("Orchestrator",  "Breaking topic into research subtasks..."),
            "researcher":   ("Researcher",    "Searching web in parallel (5 queries)..."),
            "analyst":      ("Analyst",       "Extracting structured insights..."),
        }

        # Status cards
        cols = st.columns(3)
        slots = {}
        for i, (k, (label, _)) in enumerate(node_labels.items()):
            slots[k] = cols[i].empty()
            slots[k].info(f"**{label}**\n\nWaiting...")

        try:
            with st.status("Running research pipeline...", expanded=True) as status:
                for event in research_graph.stream(
                    {"topic": topic.strip()},
                    config=config,
                ):
                    node = list(event.keys())[0]
                    if node in node_labels:
                        label, desc = node_labels[node]
                        st.write(f"**{label}** — {desc}")
                        slots[node].success(f"**{label}**\n\nDone")

                status.update(
                    label="Pipeline paused — waiting for your review",
                    state="complete"
                )

            st.session_state.graph_state = research_graph.get_state(config).values
            st.session_state.phase = "review"
            st.rerun()
        except Exception as e:
            st.error(f"Research pipeline failed: {e}")
            msg = str(e).lower()
            if "api_key" in msg or "401" in msg or "auth" in msg:
                st.info("Check your API keys in Streamlit secrets or .env (GROQ_API_KEY, TAVILY_API_KEY).")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — HUMAN REVIEW (HITL GATE)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "review":

    s = st.session_state.graph_state or {}

    # Session may expire if the server reloaded (in-memory checkpoint is wiped).
    if not s.get("subtasks") or not s.get("structured_insights"):
        st.warning("Session expired (server reloaded and checkpoint was lost). Please start a new research session.")
        if st.button("Start Over"):
            st.session_state.phase = "input"
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.graph_state = None
            st.rerun()
        st.stop()

    st.subheader("Human Review Gate")
    st.info("The pipeline has paused. Review the research insights below before the report is written.")

    # Show research plan
    with st.expander("Research Plan (Orchestrator output)", expanded=False):
        for i, task in enumerate(s.get("subtasks", []), 1):
            st.markdown(f"**{i}.** {task}")

    # Show insights
    st.subheader("Analyst Insights — Review Before Approving")
    insights = s.get("structured_insights", [])
    for insight in insights:
        st.markdown(insight)
        st.divider()

    # Optional feedback
    feedback = st.text_area(
        "Optional feedback for the Writer (leave blank to proceed as-is)",
        placeholder="e.g. Focus more on India-specific data. Add a section on salary trends.",
        height=80,
    )

    # Approve / Reject buttons
    st.markdown("**Your decision:**")
    col_approve, col_reject, _ = st.columns([1, 1, 3])

    with col_approve:
        if st.button("Approve — Generate Report", type="primary"):
            # Guard: checkpoint may be gone after a server reload (MemorySaver is in-memory).
            checkpoint = research_graph.get_state(config).values or {}
            if not checkpoint.get("topic") or not checkpoint.get("structured_insights"):
                st.warning("Session expired (server reloaded and checkpoint was lost). Please start a new research session.")
                st.session_state.phase = "input"
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.graph_state = None
                st.rerun()
            try:
                with st.status("Generating report + fact-checking...", expanded=True) as status:

                    node_labels = {
                        "human_review": ("Human Review",  "Processing approval..."),
                        "writer":       ("Writer",        "Writing structured report..."),
                        "fact_checker": ("Fact Checker",  "Validating every claim..."),
                    }
                    cols2 = st.columns(3)
                    slots2 = {}
                    for i, (k, (label, _)) in enumerate(node_labels.items()):
                        slots2[k] = cols2[i].empty()
                        slots2[k].info(f"**{label}**\n\nWaiting...")

                    # Update graph state with approval details before resuming
                    research_graph.update_state(
                        config,
                        {
                            "human_approved": True,
                            "human_feedback": feedback.strip(),
                        },
                        as_node="human_review"
                    )

                    for event in research_graph.stream(
                        None,
                        config=config,
                    ):
                        node = list(event.keys())[0]
                        if node in node_labels:
                            label, desc = node_labels[node]
                            slots2[node].success(f"**{label}**\n\n{desc} Done")

                    status.update(label="Report complete!", state="complete")

                st.session_state.graph_state = research_graph.get_state(config).values
                st.session_state.phase = "complete"
                st.rerun()
            except KeyError as e:
                st.error(f"Report generation failed: session state is missing {e} (server likely reloaded). Please click Start Over and run the research again.")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                msg = str(e).lower()
                if "api_key" in msg or "401" in msg or "auth" in msg:
                    st.info("Check your API keys in Streamlit secrets or .env (GROQ_API_KEY, TAVILY_API_KEY).")

    with col_reject:
        if st.button("Reject — Start Over"):
            st.session_state.phase       = "input"
            st.session_state.session_id  = str(uuid.uuid4())
            st.session_state.graph_state = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — COMPLETE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "complete":

    s = st.session_state.graph_state or {}

    if not s.get("final_report"):
        st.warning("Graph state was reset — please start a new research session.")
        if st.button("Start Over"):
            st.session_state.phase = "input"
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.graph_state = None
            st.rerun()
    else:
        verified = s.get("facts_verified", 0)
        disputed = s.get("facts_disputed", 0)
        total    = len(s.get("fact_check_results", []))
        score    = round((verified / total) * 100) if total > 0 else 0

        # Top metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Research Subtasks",  len(s.get("subtasks", [])))
        c2.metric("Sources Consulted",  len(s.get("subtasks", [])) * 3)
        c3.metric("Facts Verified",     f"{verified} / {total}")
        c4.metric("Fact-Check Score",   f"{score}%",
                  delta="Verified" if score >= 70 else "Review needed")

        st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Final Report", "Fact-Check Details", "Research Plan"])

    with tab1:
        st.markdown(s.get("final_report", "") or "")

        col_dl1, col_dl2 = st.columns([1, 5])
        with col_dl1:
            try:
                pdf_bytes = _generate_pdf(s.get("final_report", ""))
                st.download_button(
                    label     = "Download PDF",
                    data      = pdf_bytes,
                    file_name = "research_report.pdf",
                    mime      = "application/pdf",
                    type      = "primary",
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")
                st.download_button(
                    label     = "Download (.md)",
                    data      = s.get("final_report", ""),
                    file_name = "research_report.md",
                    mime      = "text/markdown",
                )

    with tab2:
        results = s.get("fact_check_results", [])
        if results:
            for r in results:
                icon = "[PASS]" if r["verdict"] == "VERIFIED" else "[UNKNOWN]" if r["verdict"] == "UNVERIFIED" else "[FAIL]"
                with st.expander(f"{icon} {r['verdict']} — {r['claim'][:80]}"):
                    st.caption(f"Source: {r['source']}")

            # Pie chart
            counts = {
                "VERIFIED":   sum(1 for r in results if r["verdict"] == "VERIFIED"),
                "UNVERIFIED": sum(1 for r in results if r["verdict"] == "UNVERIFIED"),
                "DISPUTED":   sum(1 for r in results if r["verdict"] == "DISPUTED"),
            }
            fig = go.Figure(go.Pie(
                labels = list(counts.keys()),
                values = list(counts.values()),
                marker_colors = ["#10B981", "#F59E0B", "#EF4444"],
            ))
            fig.update_layout(title="Claim Verification Breakdown", height=350,
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        for i, task in enumerate(s.get("subtasks", []), 1):
            st.markdown(f"**{i}.** {task}")

    st.divider()
    if st.button("New Research"):
        st.session_state.phase       = "input"
        st.session_state.session_id  = str(uuid.uuid4())
        st.session_state.graph_state = None
        st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
**5 Agents:**
1. Orchestrator — research plan
2. Researcher — parallel web search
3. Analyst — structured insights
4. Writer — professional report
5. Fact Checker — claim validation

**Key Features:**
- Human-in-the-Loop gate
- Parallel search (5x faster)
- LLM hallucination detection
- FastAPI REST endpoint
- LangGraph state machine
    """)
    phase_labels = {"input": "Awaiting input", "review": "Awaiting review", "complete": "Complete"}
    st.info(f"**Status:** {phase_labels.get(st.session_state.phase, '—')}")
