from models.report import InvestmentReport

def report_to_markdown(report: InvestmentReport) -> str:
    def bullets(items):
        return "\n".join(f"- {x}" for x in items) if items else "- None reported"

    return f"""# AlphaVest Investment Research Report

## Company Overview
{report.company_overview}

## Industry
{report.industry}

## Business Model
{report.business_model}

## Latest News
{bullets(report.latest_news)}

## Strengths
{bullets(report.strengths)}

## Weaknesses
{bullets(report.weaknesses)}

## Financial Highlights
{bullets(report.financial_highlights)}

## Growth Opportunities
{bullets(report.growth_opportunities)}

## Potential Risks
{bullets(report.potential_risks)}

## Investment Summary
{report.investment_summary}

> This report is for research purposes and is not a guarantee of investment performance.
"""
