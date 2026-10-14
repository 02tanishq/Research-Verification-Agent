# Setup Guide

## Prerequisites
- Python 3.10+
- Groq API key → https://console.groq.com/keys (free)
- Tavily API key → https://tavily.com (1000 free searches/month)

---

## Local Setup

```bash
git clone https://github.com/yourusername/research-report-system
cd research-report-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add GROQ_API_KEY and TAVILY_API_KEY
```

## Run terminal test
```bash
python test_run.py
```

## Run Streamlit UI
```bash
streamlit run app.py
```

## Run FastAPI
```bash
uvicorn api.main:app --reload
# Open http://localhost:8000/docs for interactive API docs
```

---

## Deploy Streamlit to Streamlit Cloud
1. Push to public GitHub repo
2. share.streamlit.io → New app → select repo → main file: `app.py`
3. Add secrets: `GROQ_API_KEY` and `TAVILY_API_KEY`
4. Deploy

## Deploy FastAPI to Render (free)
1. Create a new Web Service on render.com
2. Connect GitHub repo
3. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add env vars: `GROQ_API_KEY`, `TAVILY_API_KEY`

---

## Common Errors

```
ModuleNotFoundError: No module named 'langchain_groq'
→ pip install langchain-groq

ImportError: cannot import name 'MemorySaver'
→ pip install --upgrade langgraph

TavilyError: Invalid API key
→ Check your .env file for typos
```
