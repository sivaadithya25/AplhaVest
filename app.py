import os
import re
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from agents.agents import make_news_agent, make_research_agent, make_financial_agent
from tools.financial_tools import FINANCIAL_TOOLS
from tools.search_tools import web_search, wiki
from rag.pdf_rag import build_knowledge_base, retrieve_context, get_retriever
from memory.sqlite_memory import (
    init_db, save_message, recent_messages, set_profile, get_profile, clear_memory
)
from workflows.parallel import build_parallel_research
from models.report import InvestmentReport
from report_generator import report_to_markdown
from email_agent import send_gmail_report

load_dotenv()
init_db()

st.set_page_config(page_title="AlphaVest", page_icon="📈", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "report" not in st.session_state:
    st.session_state.report = None
if "last_context" not in st.session_state:
    st.session_state.last_context = ""
if "last_research" not in st.session_state:
    st.session_state.last_research = {}

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

news_agent = make_news_agent(llm, [web_search])
research_agent = make_research_agent(llm, [web_search, wiki])
financial_agent = make_financial_agent(llm, FINANCIAL_TOOLS + [web_search])

st.title("📈 AlphaVest Capital")
st.caption("AI Investment & Financial Research Assistant")

with st.sidebar:
    st.header("Research Workspace")

    uploads = st.file_uploader(
        "Upload annual / quarterly reports",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Build Knowledge Base", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one PDF.")
        else:
            os.makedirs("data/uploads", exist_ok=True)
            paths = []
            for f in uploads:
                path = os.path.join("data/uploads", f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(path)
            with st.spinner("Reading PDFs and building ChromaDB..."):
                pages, chunks = build_knowledge_base(paths)
            st.success(f"Knowledge base ready: {pages} pages, {chunks} chunks.")

    st.divider()
    st.subheader("Investor Memory")

    profile = get_profile()
    name = st.text_input("Client name", profile.get("client_name", ""))
    risk = st.selectbox(
        "Risk profile",
        ["Not set", "Low", "Medium", "High"],
        index=["Not set", "Low", "Medium", "High"].index(profile.get("risk_profile", "Not set"))
    )
    interests = st.text_input(
        "Investment interests",
        profile.get("investment_interests", "")
    )

    if st.button("Save Investor Profile", use_container_width=True):
        set_profile("client_name", name)
        set_profile("risk_profile", risk)
        set_profile("investment_interests", interests)
        st.success("Investor profile saved.")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_context = ""
        st.rerun()

    if st.button("Clear All Memory", use_container_width=True):
        clear_memory()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Previous Conversations")
    old = recent_messages(st.session_state.session_id, 8)
    for role, content in old:
        st.caption(f"{role}: {content[:120]}")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def extract_companies(text):
    known = [
        "Microsoft", "Google", "Alphabet", "Amazon", "Meta", "NVIDIA",
        "Tesla", "Apple", "AMD", "Intel", "Netflix"
    ]
    return [c for c in known if c.lower() in text.lower()]

def ask_news(question):
    return news_agent.invoke({"messages": [{"role": "user", "content": question}]})["messages"][-1].content

def ask_research(question):
    return research_agent.invoke({"messages": [{"role": "user", "content": question}]})["messages"][-1].content

def ask_financial(question):
    return financial_agent.invoke({"messages": [{"role": "user", "content": question}]})["messages"][-1].content

def ask_pdf(question):
    docs, context = retrieve_context(question)
    st.session_state.last_context = context
    if not docs:
        return "No PDF knowledge base is available. Upload a report and click Build Knowledge Base."
    prompt = f"""You are the AlphaVest PDF Research Agent.
Answer ONLY from the uploaded document context below.
If the answer is not present, say so.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""
    return llm.invoke(prompt).content

def compare_companies(question):
    companies = extract_companies(question)
    if len(companies) < 2:
        return "Please name at least two companies to compare."
    research = build_parallel_research(web_search, companies[:6])
    combined = "\n\n".join(
        f"### {k}\n{v['research']}" for k, v in research.items()
    )
    prompt = f"""Compare these companies using the research below.
Create a concise table and explain strengths, weaknesses, business model,
financial/revenue information available from the research, growth opportunities,
risks and a neutral conclusion.

RESEARCH:
{combined}

USER REQUEST:
{question}
"""
    answer = llm.invoke(prompt).content
    st.session_state.last_research = research
    return answer

def generate_report(company):
    research = ask_research(f"Research {company} with business overview, industry, business model, products, competitors, revenue sources and recent announcements.")
    news = ask_news(f"Latest financial news and major announcements for {company}.")
    _, pdf_context = retrieve_context(f"{company} financial highlights risks revenue growth future plans")
    if not pdf_context or pdf_context.startswith("No PDF"):
        pdf_context = "No uploaded report context available."

    structured = llm.with_structured_output(InvestmentReport)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Create a structured investment research report.
Use the supplied research, news and PDF context. Do not invent facts.
If information is missing, say so in the relevant field.
The output must be valid structured data."""),
        ("human", """Company: {company}

COMPANY RESEARCH:
{research}

LATEST NEWS:
{news}

PDF CONTEXT:
{pdf_context}""")
    ])
    chain = prompt | structured
    return chain.invoke({
        "company": company,
        "research": research,
        "news": news,
        "pdf_context": pdf_context
    })

def classify(question):
    q = question.lower()
    if any(x in q for x in ["latest news", "recent news", "earnings", "announcement", "stock news"]):
        return "news"
    if any(x in q for x in ["annual report", "quarterly report", "uploaded report", "risk factors", "future plans", "revenue growth"]):
        return "pdf"
    if any(x in q for x in ["compare", "comparison", " versus ", " vs "]):
        return "compare"
    if any(x in q for x in ["research", "company overview", "business model", "competitors"]):
        return "research"
    return "financial"

question = st.chat_input("Ask AlphaVest: research NVIDIA, compare Microsoft and Google, calculate CAGR...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    save_message(st.session_state.session_id, "user", question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                route = classify(question)
                companies = extract_companies(question)

                if route == "news":
                    answer = ask_news(question)
                elif route == "pdf":
                    answer = ask_pdf(question)
                elif route == "compare":
                    answer = compare_companies(question)
                elif route == "research":
                    answer = ask_research(question)
                else:
                    answer = ask_financial(question)

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                save_message(st.session_state.session_id, "assistant", answer)

                # Generate a structured report when explicitly requested.
                if any(x in question.lower() for x in ["generate report", "investment report", "research report"]):
                    company = companies[0] if companies else None
                    if company:
                        report = generate_report(company)
                        st.session_state.report = report
                        st.success("Structured investment report generated.")

            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

# Report area
if st.session_state.report:
    report = st.session_state.report
    markdown = report_to_markdown(report)

    st.divider()
    st.header("📄 Generated Investment Report")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download TXT / Markdown",
            data=markdown,
            file_name="alphavest_investment_report.md",
            mime="text/markdown",
            on_click="ignore"
        )
    with col2:
        if st.button("Email Report to Client"):
            ok, msg = send_gmail_report(
                "AlphaVest Investment Research Report",
                markdown
            )
            (st.success if ok else st.error)(msg)

    st.markdown(markdown)

with st.expander("Retrieved PDF Chunks"):
    st.text(st.session_state.last_context or "No PDF retrieval performed yet.")

with st.expander("Financial Calculations / Research"):
    st.json(st.session_state.last_research or {"info": "No parallel comparison performed yet."})

with st.expander("Investor Profile"):
    st.json(get_profile())
