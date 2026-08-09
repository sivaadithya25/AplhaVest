import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from agents.agents import (
    make_news_agent,
    make_research_agent,
    make_financial_agent,
)
from tools.financial_tools import FINANCIAL_TOOLS
from tools.search_tools import web_search, wiki
from rag.pdf_rag import build_knowledge_base, retrieve_context
from memory.sqlite_memory import (
    init_db,
    save_message,
    recent_messages,
    set_profile,
    get_profile,
    clear_memory,
)
from workflows.parallel import build_parallel_research
from models.report import InvestmentReport
from report_generator import report_to_markdown
from email_agent import send_gmail_report


# --------------------------------------------------
# SETUP
# --------------------------------------------------

load_dotenv()
init_db()

st.set_page_config(
    page_title="AlphaVest",
    page_icon="📈",
    layout="wide",
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

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

if "last_company" not in st.session_state:
    st.session_state.last_company = None


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# --------------------------------------------------
# AGENTS
# --------------------------------------------------

news_agent = make_news_agent(
    llm,
    [web_search],
)

research_agent = make_research_agent(
    llm,
    [web_search, wiki],
)

financial_agent = make_financial_agent(
    llm,
    FINANCIAL_TOOLS + [web_search],
)


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("📈 AlphaVest Capital")
st.caption("AI Investment & Financial Research Assistant")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("Research Workspace")

    uploads = st.file_uploader(
        "Upload annual / quarterly reports",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Build Knowledge Base",
        use_container_width=True,
    ):
        if not uploads:
            st.warning("Upload at least one PDF.")
        else:
            os.makedirs("data/uploads", exist_ok=True)

            paths = []

            for f in uploads:
                path = os.path.join(
                    "data/uploads",
                    f.name,
                )

                with open(path, "wb") as out:
                    out.write(f.getbuffer())

                paths.append(path)

            with st.spinner(
                "Reading PDFs and building ChromaDB..."
            ):
                pages, chunks = build_knowledge_base(paths)

            st.success(
                f"Knowledge base ready: {pages} pages, {chunks} chunks."
            )

    st.divider()

    # --------------------------------------------------
    # INVESTOR MEMORY
    # --------------------------------------------------

    st.subheader("Investor Memory")

    profile = get_profile()

    name = st.text_input(
        "Client name",
        profile.get("client_name", ""),
    )

    risk_options = [
        "Not set",
        "Low",
        "Medium",
        "High",
    ]

    current_risk = profile.get(
        "risk_profile",
        "Not set",
    )

    if current_risk not in risk_options:
        current_risk = "Not set"

    risk = st.selectbox(
        "Risk profile",
        risk_options,
        index=risk_options.index(current_risk),
    )

    interests = st.text_input(
        "Investment interests",
        profile.get(
            "investment_interests",
            "",
        ),
    )

    if st.button(
        "Save Investor Profile",
        use_container_width=True,
    ):

        set_profile(
            "client_name",
            name,
        )

        set_profile(
            "risk_profile",
            risk,
        )

        set_profile(
            "investment_interests",
            interests,
        )

        st.success(
            "Investor profile saved."
        )

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.last_context = ""
        st.rerun()

    if st.button(
        "Clear All Memory",
        use_container_width=True,
    ):
        clear_memory()
        st.session_state.messages = []
        st.session_state.last_company = None
        st.rerun()

    st.divider()

    st.subheader("Previous Conversations")

    old = recent_messages(
        st.session_state.session_id,
        8,
    )

    for role, content in old:
        st.caption(
            f"{role}: {content[:120]}"
        )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def extract_companies(text):

    known = [
        "Microsoft",
        "Google",
        "Alphabet",
        "Amazon",
        "Meta",
        "NVIDIA",
        "Tesla",
        "Apple",
        "AMD",
        "Intel",
        "Netflix",
    ]

    return [
        c
        for c in known
        if c.lower() in text.lower()
    ]


def profile_context():

    profile = get_profile()

    if not profile:
        return "No investor profile has been saved."

    return f"""
Investor profile:

Client name: {profile.get("client_name", "Not set")}
Risk profile: {profile.get("risk_profile", "Not set")}
Investment interests: {profile.get("investment_interests", "Not set")}
"""


def ask_news(question):

    return news_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )["messages"][-1].content


def ask_research(question):

    return research_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )["messages"][-1].content


def ask_financial(question):

    return financial_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )["messages"][-1].content


def ask_pdf(question):

    docs, context = retrieve_context(
        question
    )

    st.session_state.last_context = context

    if not docs:
        return (
            "No PDF knowledge base is available. "
            "Upload a report and click Build Knowledge Base."
        )

    prompt = f"""
You are the AlphaVest PDF Research Agent.

Answer ONLY from the uploaded document context below.

If the answer is not present, say so.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

    return llm.invoke(prompt).content


# --------------------------------------------------
# PARALLEL COMPANY COMPARISON
# --------------------------------------------------

def compare_companies(question):

    companies = extract_companies(question)

    if len(companies) < 2:
        return (
            "Please name at least two companies to compare."
        )

    research = build_parallel_research(
        web_search,
        companies[:6],
    )

    combined = "\n\n".join(
        f"### {k}\n{v['research']}"
        for k, v in research.items()
    )

    prompt = f"""
Compare these companies using the research below.

Create:

1. Comparison table
2. Business models
3. Strengths
4. Weaknesses
5. Financial/revenue information available
6. Growth opportunities
7. Risks
8. Neutral conclusion

RESEARCH:

{combined}

USER REQUEST:

{question}
"""

    answer = llm.invoke(prompt).content

    st.session_state.last_research = research

    return answer


# --------------------------------------------------
# STRUCTURED INVESTMENT REPORT
# --------------------------------------------------

def generate_report(company):

    # --------------------------------------------------
    # DIRECT WEB RESEARCH
    # --------------------------------------------------

    research_queries = [
        f"{company} business overview products competitors revenue",
        f"{company} business model industry financial highlights",
        f"{company} recent announcements",
    ]

    research_results = []

    for query in research_queries:
        try:
            result = web_search.invoke(query)
            research_results.append(str(result))
        except Exception as e:
            research_results.append(
                f"Research unavailable for query '{query}': {e}"
            )

    research = "\n\n".join(research_results)

    # --------------------------------------------------
    # NEWS
    # --------------------------------------------------

    try:
        news = web_search.invoke(
            f"{company} latest financial news earnings announcements"
        )
    except Exception as e:
        news = f"News unavailable: {e}"

    # --------------------------------------------------
    # PDF RAG
    # --------------------------------------------------

    try:
        _, pdf_context = retrieve_context(
            f"{company} financial highlights risks revenue growth future plans"
        )
    except Exception:
        pdf_context = ""

    if not pdf_context:
        pdf_context = "No uploaded report context available."

    # --------------------------------------------------
    # STRUCTURED OUTPUT
    # --------------------------------------------------

    structured_llm = llm.with_structured_output(
    InvestmentReport
)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a professional financial research report generator.

Create a balanced investment research report using ONLY
the information supplied below.

IMPORTANT:
- Do NOT call tools.
- Do NOT search the internet.
- Do NOT invent facts.
- Do NOT invent financial numbers.
- If information is missing, use:
  ["Not available in the provided research."]
- latest_news MUST be a list of strings.
- strengths MUST be a list of strings.
- weaknesses MUST be a list of strings.
- financial_highlights MUST be a list of strings.
- growth_opportunities MUST be a list of strings.
- potential_risks MUST be a list of strings.
- Never return a string where a list is required.
- Do not provide personalized financial advice.

Return valid structured output.
"""
        ),
        (
            "human",
            """
COMPANY:
{company}

COMPANY RESEARCH:
{research}

LATEST NEWS:
{news}

PDF CONTEXT:
{pdf_context}

INVESTOR PROFILE:
{profile}
"""
        )
    ])

    chain = prompt | structured_llm

    return chain.invoke({
        "company": company,
        "research": research,
        "news": news,
        "pdf_context": pdf_context,
        "profile": profile_context(),
    })


# --------------------------------------------------
# ROUTER
# --------------------------------------------------

def classify(question):

    q = question.lower()

    # Email must be checked FIRST.
    if any(
        x in q
        for x in [
            "email",
            "send report",
            "mail report",
            "send it to my client",
        ]
    ):
        return "email"

    # Investment report.
    if any(
        x in q
        for x in [
            "generate report",
            "investment report",
            "research report",
            "complete report",
        ]
    ):
        return "report"

    # Investor memory.
    if any(
        x in q
        for x in [
            "my investor profile",
            "my saved profile",
            "my investment profile",
            "my risk profile",
            "my investment preferences",
            "what do you remember about me",
        ]
    ):
        return "profile"

    # News.
    if any(
        x in q
        for x in [
            "latest news",
            "recent news",
            "earnings",
            "announcement",
            "stock news",
        ]
    ):
        return "news"

    # PDF.
    if any(
        x in q
        for x in [
            "annual report",
            "quarterly report",
            "uploaded report",
            "risk factors",
            "future plans",
            "revenue growth",
        ]
    ):
        return "pdf"

    # Comparison.
    if any(
        x in q
        for x in [
            "compare",
            "comparison",
            " versus ",
            " vs ",
        ]
    ):
        return "compare"

    # Research.
    if any(
        x in q
        for x in [
            "research",
            "company overview",
            "business model",
            "competitors",
        ]
    ):
        return "research"

    return "financial"


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask AlphaVest: research NVIDIA, compare Microsoft and Google, calculate CAGR..."
)


# --------------------------------------------------
# MAIN PROCESSING
# --------------------------------------------------

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    save_message(
        st.session_state.session_id,
        "user",
        question,
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Researching..."):

            try:

                route = classify(question)

                companies = extract_companies(
                    question
                )

                # Remember last researched company.
                if companies:
                    st.session_state.last_company = (
                        companies[0]
                    )

                # --------------------------------
                # ROUTE: NEWS
                # --------------------------------

                if route == "news":

                    answer = ask_news(
                        question
                    )

                # --------------------------------
                # ROUTE: PDF
                # --------------------------------

                elif route == "pdf":

                    answer = ask_pdf(
                        question
                    )

                # --------------------------------
                # ROUTE: COMPARISON
                # --------------------------------

                elif route == "compare":

                    answer = compare_companies(
                        question
                    )

                # --------------------------------
                # ROUTE: RESEARCH
                # --------------------------------

                elif route == "research":

                    answer = ask_research(
                        question
                    )

                # --------------------------------
                # ROUTE: INVESTOR PROFILE
                # --------------------------------

                elif route == "profile":

                    profile = get_profile()

                    if not profile:
                        answer = (
                            "No investor profile has "
                            "been saved yet."
                        )
                    else:
                        answer = f"""
### Saved Investor Profile

**Client Name:** {profile.get("client_name", "Not set")}

**Risk Profile:** {profile.get("risk_profile", "Not set")}

**Investment Interests:** {profile.get("investment_interests", "Not set")}
"""

                # --------------------------------
                # ROUTE: REPORT
                # --------------------------------

                elif route == "report":

                    company = (
                        companies[0]
                        if companies
                        else st.session_state.last_company
                    )

                    if not company:
                        answer = (
                            "Please specify a company, "
                            "for example: Generate a complete "
                            "investment report for NVIDIA."
                        )
                    else:

                        # Generate report.
                        report = generate_report(
                            company
                        )

                        st.session_state.report = (
                            report
                        )

                        answer = (
                            f"Investment report for "
                            f"**{company}** has been generated "
                            "below."
                        )

                # --------------------------------
                # ROUTE: EMAIL
                # --------------------------------

                elif route == "email":

                    company = (
                        companies[0]
                        if companies
                        else st.session_state.last_company
                    )

                    if not company:

                        answer = (
                            "Please specify the company "
                            "whose report should be emailed."
                        )

                    else:

                        # Generate report first.
                        report = generate_report(
                            company
                        )

                        st.session_state.report = (
                            report
                        )

                        markdown = (
                            report_to_markdown(
                                report
                            )
                        )

                        ok, msg = send_gmail_report(
                            "AlphaVest Investment Research Report",
                            markdown,
                        )

                        if ok:

                            answer = (
                                f"Investment report for "
                                f"**{company}** was generated "
                                f"and emailed successfully.\n\n"
                                f"{msg}"
                            )

                        else:

                            answer = (
                                f"The report for "
                                f"**{company}** was generated, "
                                "but the email could not be sent.\n\n"
                                f"Email status: {msg}"
                            )

                # --------------------------------
                # ROUTE: FINANCIAL
                # --------------------------------

                else:

                    answer = ask_financial(
                        question
                    )

                # Display answer.
                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                save_message(
                    st.session_state.session_id,
                    "assistant",
                    answer,
                )

            except Exception as e:

                answer = f"Error: {e}"

                st.error(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )


# --------------------------------------------------
# GENERATED REPORT
# --------------------------------------------------

if st.session_state.report:

    report = st.session_state.report

    markdown = report_to_markdown(
        report
    )

    st.divider()

    st.header(
        "📄 Generated Investment Report"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "Download TXT / Markdown",
            data=markdown,
            file_name="alphavest_investment_report.md",
            mime="text/markdown",
            on_click="ignore",
        )

    with col2:

        if st.button(
            "Email Report to Client"
        ):

            ok, msg = send_gmail_report(
                "AlphaVest Investment Research Report",
                markdown,
            )

            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown(markdown)


# --------------------------------------------------
# DEBUG / INFORMATION PANELS
# --------------------------------------------------

with st.expander(
    "Retrieved PDF Chunks"
):

    st.text(
        st.session_state.last_context
        or "No PDF retrieval performed yet."
    )


with st.expander(
    "Financial Calculations / Research"
):

    st.json(
        st.session_state.last_research
        or {
            "info": (
                "No parallel comparison "
                "performed yet."
            )
        }
    )


with st.expander(
    "Investor Profile"
):

    st.json(
        get_profile()
    )