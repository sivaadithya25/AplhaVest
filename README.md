# AlphaVest Capital — AI Investment & Financial Research Assistant

## Features

- Streamlit conversational UI
- LangChain agents
- Financial news search with DuckDuckGo
- Company research with DuckDuckGo + Wikipedia
- PDF RAG with ChromaDB
- Multi-company research with RunnableParallel
- Conditional request routing
- Pydantic structured investment reports
- CAGR, growth, ROI and comparison tools
- SQLite conversation and investor memory
- TXT/Markdown report download
- Gmail SMTP report delivery
- Optional Google Drive extension point

## Setup

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:

```text
GROQ_API_KEY=...
```

Run:

```powershell
streamlit run app.py
```

## Example prompts

```text
Research NVIDIA.
What is the latest NVIDIA news?
Compare Microsoft, Google, Amazon and Meta.
Calculate CAGR from 100 to 150 over 3 years.
Calculate ROI from 1000 to 1300.
Summarize the risk factors in my annual report.
Generate an investment report for NVIDIA.
Remember that I prefer low-risk technology investments.
```

## Gmail

Use a Gmail App Password rather than your normal Gmail password. Set:

```text
GMAIL_ADDRESS=...
GMAIL_APP_PASSWORD=...
CLIENT_EMAIL=...
```

## Important

This application is a research assistant, not a guarantee of investment returns or a substitute for a licensed financial adviser.
