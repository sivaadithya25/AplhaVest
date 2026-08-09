from pydantic import BaseModel, Field
from typing import List


class InvestmentReport(BaseModel):

    company_overview: str = Field(
        description="Overview of the company"
    )

    industry: str = Field(
        description="Industry of the company"
    )

    business_model: str = Field(
        description="How the company makes money"
    )

    latest_news: List[str] = Field(
        description="Important recent news. Return an array."
    )

    strengths: List[str] = Field(
        description="Major company strengths. Return an array."
    )

    weaknesses: List[str] = Field(
        description="Major company weaknesses. Return an array."
    )

    financial_highlights: List[str] = Field(
        description="Important financial information. Return an array."
    )

    growth_opportunities: List[str] = Field(
        description="Potential growth opportunities. Return an array."
    )

    potential_risks: List[str] = Field(
        description="Potential risks. Return an array."
    )

    investment_summary: str = Field(
        description="Balanced investment summary"
    )