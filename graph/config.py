"""
graph/config.py
================
Centralised configuration — reads from os.environ first,
then falls back to streamlit secrets for Streamlit Cloud.
"""

import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def invoke_with_retry(llm, messages, max_retries: int = 3, base_delay: float = 2.0):
    """Invoke Groq LLM with exponential backoff on 429/RateLimit."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RateLimit" in type(e).__name__ or "rate_limit" in msg.lower()
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                # try to honor retry-after header if present
                if "try again in" in msg.lower():
                    try:
                        import re
                        m = re.search(r"try again in (\d+\.?\d*)s", msg)
                        if m:
                            delay = float(m.group(1)) + 0.5
                    except Exception:
                        pass
                print(f"[retry] Groq rate limit hit (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("Max retries exceeded for LLM invoke")


def get_secret(key: str, default: str | None = None) -> str | None:
    val = os.environ.get(key)
    if val:
        return val
    try:
        # pyrefly: ignore [missing-import]
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default
