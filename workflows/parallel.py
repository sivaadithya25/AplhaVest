from langchain_core.runnables import RunnableParallel, RunnableLambda

def research_company(search_tool, company):
    result = search_tool.invoke(
        f"{company} company business overview products competitors revenue latest announcements financial results"
    )
    return {"company": company, "research": result}

def build_parallel_research(search_tool, companies):
    branches = {
        company: RunnableLambda(lambda x, c=company: research_company(search_tool, c))
        for company in companies
    }
    parallel = RunnableParallel(**branches)
    return parallel.invoke({})
