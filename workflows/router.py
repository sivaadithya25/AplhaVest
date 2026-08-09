from langchain_core.runnables import RunnableBranch, RunnableLambda

def make_router(news_fn, pdf_fn, compare_fn, research_fn, default_fn):
    return RunnableBranch(
        (lambda q: any(w in q.lower() for w in ["latest news", "recent news", "earnings", "announcement", "stock news"]),
         RunnableLambda(news_fn)),
        (lambda q: any(w in q.lower() for w in ["annual report", "quarterly report", "uploaded report", "risk factors", "revenue growth"]),
         RunnableLambda(pdf_fn)),
        (lambda q: any(w in q.lower() for w in ["compare", "comparison", "versus", " vs "]),
         RunnableLambda(compare_fn)),
        (lambda q: any(w in q.lower() for w in ["research", "company overview", "business model", "competitors"]),
         RunnableLambda(research_fn)),
        RunnableLambda(default_fn),
    )
