"""
agents/fact_checker.py
======================
AGENT 5 — The Fact-Checker

WHAT IT DOES:
  Reads the Writer's draft report, extracts every factual claim,
  and validates each one against the original research sources.

  Each claim gets one of three verdicts:
    VERIFIED   — claim is supported by a source we searched
    UNVERIFIED — claim can't be confirmed from our sources
    DISPUTED   — claim contradicts what our sources say

  The final report adds a fact-check summary and marks any
  disputed claims clearly so the reader knows what to verify.

WHY THIS IS IMPRESSIVE FOR 10-12 LPA:
  Hallucination prevention is one of the biggest challenges in
  production LLM systems. Adding a fact-checker shows you
  understand this problem and know how to address it.

  Most tutorial chatbots output whatever the LLM says.
  A Fact-Checker shows production thinking.

HOW TO EXPLAIN TO HR:
  "The Fact-Checker is the last agent before output. It extracts
   every factual claim from the draft report and checks it against
   the original Tavily sources. This addresses LLM hallucination —
   the tendency to generate plausible-sounding but false statements.
   Each claim is labelled VERIFIED, UNVERIFIED, or DISPUTED.
   The final report includes a fact-check score so readers know
   how much they can trust the content."

Input  → state["draft_report"], state["raw_research"]
Output → state["fact_check_results"], state["final_report"],
         state["facts_verified"], state["facts_disputed"]
"""

import os
import json
import re
from concurrent.futures import ThreadPoolExecutor
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AppState, FactCheckResult
from graph.config import get_secret, invoke_with_retry
from graph.cache import cache_get, cache_set, TTL_FACTCHECK

# ── LLM setup ─────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,            # deterministic — important for fact-checking
    api_key=get_secret("GROQ_API_KEY"),
    max_tokens=600,
)


def _extract_claims_from_report(report: str) -> list[str]:
    """
    Ask the LLM to extract all factual claims from the report.
    Returns a list of claim strings.
    """

    # Cache: same report slice → same claim list.
    report_slice = report[:3000]
    cache_payload = {"report": report_slice}
    hit, cached_claims = cache_get("claims", cache_payload, TTL_FACTCHECK)
    if hit:
        return cached_claims

    response = invoke_with_retry(llm, [
        SystemMessage(content="""Extract all factual claims from this report.
A factual claim is a specific, verifiable statement (includes numbers, names, percentages).
Return ONLY a JSON array of claim strings, like: ["claim 1", "claim 2"]. No explanation, no markdown."""),

        HumanMessage(content=f"Report:\n{report[:3000]}")
    ])

    raw = response.content.strip()

    # Strip markdown code fences if present
    raw_clean = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Try direct parse first
    try:
        claims = json.loads(raw_clean)
        if isinstance(claims, list):
            claims = claims[:10]
            cache_set("claims", cache_payload, claims)
            return claims
    except Exception:
        pass

    # Fallback: extract the first JSON array found anywhere in the response
    match = re.search(r"(\[.*?\])", raw_clean, re.DOTALL)
    if match:
        try:
            claims = json.loads(match.group(1))
            if isinstance(claims, list):
                print(f"[fact_checker] Extracted {len(claims)} claims via regex fallback.")
                claims = claims[:10]
                cache_set("claims", cache_payload, claims)
                return claims
        except Exception as e:
            print("\n=== CLAIM EXTRACTION ERROR ===")
            print("Raw LLM Response:", raw)
            print("Error:", e)
            print("==============================\n")

    print("\n=== CLAIM EXTRACTION FAILED — no JSON array found ===")
    print("Raw response:", raw[:500])
    print("======================================================\n")
    return []


def _verify_one_claim(
    claim:        str,
    all_sources:  str,
) -> FactCheckResult:
    """
    Verify a single claim against the research sources.
    Returns a FactCheckResult with verdict and source URL.
    """

    # Cache: same claim + same sources slice → same verdict.
    sources_slice = all_sources[:5000]
    cache_payload = {"claim": claim, "sources": sources_slice}
    hit, cached = cache_get("verdict", cache_payload, TTL_FACTCHECK)
    if hit:
        return FactCheckResult(
            claim   = claim,
            verdict = cached.get("verdict", "UNVERIFIED"),
            source  = cached.get("source", "Not found in sources"),
        )

    response = invoke_with_retry(llm, [
        SystemMessage(content="""You are a fact-checker. Check if the claim is supported by the sources.

Respond with ONLY valid JSON (no markdown, no explanation):
{"verdict": "VERIFIED" | "UNVERIFIED" | "DISPUTED", "source": "<URL or 'Not found'>"}

Rules:
- VERIFIED   = the claim is directly stated or strongly implied by a source
- UNVERIFIED = the sources don't mention this topic at all
- DISPUTED   = a source explicitly contradicts the claim

Be generous with VERIFIED — if a source discusses the same topic and the numbers/facts are plausible, mark as VERIFIED."""),

        HumanMessage(content=f"""
Claim to verify: {claim}

Sources:
{sources_slice}""")
    ])

    raw = response.content.strip()
    raw_clean = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Try direct parse
    data = {}
    try:
        data = json.loads(raw_clean)
    except Exception:
        # Fallback: find first JSON object in response
        match = re.search(r"(\{.*?\})", raw_clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except Exception:
                pass

    result = FactCheckResult(
        claim   = claim,
        verdict = data.get("verdict", "UNVERIFIED"),
        source  = data.get("source",  "Not found in sources"),
    )
    cache_set("verdict", cache_payload, {"verdict": result["verdict"], "source": result["source"]})
    return result


def _build_sources_text(raw_research: list) -> str:
    """Combine all research sources into one text block for fact-checking."""
    parts = []
    for item in raw_research:
        for src in item.get("sources", []):
            parts.append(f"URL: {src['url']}\n{src['content'][:700]}")
    return "\n\n---\n\n".join(parts)


def _add_fact_check_summary(
    report:   str,
    results:  list[FactCheckResult],
    verified: int,
    disputed: int,
) -> str:
    """Append a fact-check summary section to the report."""

    total = len(results)
    score = round((verified / total) * 100) if total > 0 else 0

    # Build the claims table
    rows = []
    for r in results:
        icon = "[PASS]" if r["verdict"] == "VERIFIED" else "[UNKNOWN]" if r["verdict"] == "UNVERIFIED" else "[FAIL]"
        rows.append(f"| {icon} {r['verdict']} | {r['claim'][:80]} |")

    table = "\n".join(rows)

    summary = f"""

---

## Fact-Check Report

**Fact-Check Score: {score}% ({verified}/{total} claims verified)**

| Verdict | Claim |
|---------|-------|
{table}

*Fact-checking performed by AI — always verify critical claims independently.*"""

    return report + summary


def fact_checker_agent(state: AppState) -> dict:
    """
    MAIN FUNCTION — called by LangGraph as a node.

    Reads:  state["draft_report"], state["raw_research"]
    Writes: state["fact_check_results"], state["final_report"],
            state["facts_verified"], state["facts_disputed"]
    """

    draft = state.get("draft_report", "") or ""

    print("\n" + "="*60)
    print("FACT CHECKER AGENT STARTED")
    print(f"  draft_report length : {len(draft)}")
    print(f"  raw_research items  : {len(state.get('raw_research', []))}")
    print(f"  draft snippet       : {repr(draft[:200])}")
    print("="*60 + "\n")

    # Guard: nothing to fact-check if report is empty
    if not draft.strip():
        print("[fact_checker] ERROR: draft_report is empty — skipping fact check.")
        return {
            "fact_check_results": [],
            "final_report":       draft,
            "facts_verified":     0,
            "facts_disputed":     0,
        }

    # Step 1: Extract claims from the draft report
    claims = _extract_claims_from_report(draft)

    print(f"[fact_checker] Claims extracted: {len(claims)}")
    for i, c in enumerate(claims, 1):
        print(f"  {i}. {c[:100]}")

    if not claims:
        print("[fact_checker] No claims extracted — returning 0/0.")
        return {
            "fact_check_results": [],
            "final_report":       draft,
            "facts_verified":     0,
            "facts_disputed":     0,
        }

    # Step 2: Combine all sources into one text block
    all_sources = _build_sources_text(state.get("raw_research", []))
    print(f"[fact_checker] Source text length: {len(all_sources)} chars")

    # Step 3: Verify each claim in PARALLEL (same logic as the Researcher's
    # parallel Tavily searches). executor.map preserves input order, so
    # results[i] still corresponds to claims[i]. Per-call 429s are absorbed
    # by invoke_with_retry's exponential backoff inside _verify_one_claim.
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(
            lambda claim: _verify_one_claim(claim, all_sources),
            claims,
        ))

    # Step 4: Count verdicts
    verified = sum(1 for r in results if r["verdict"] == "VERIFIED")
    disputed = sum(1 for r in results if r["verdict"] == "DISPUTED")

    # Step 5: Add fact-check summary to the report
    final_report = _add_fact_check_summary(
        report   = state["draft_report"],
        results  = results,
        verified = verified,
        disputed = disputed,
    )

    return {
        "fact_check_results": results,
        "final_report":       final_report,
        "facts_verified":     verified,
        "facts_disputed":     disputed,
    }
