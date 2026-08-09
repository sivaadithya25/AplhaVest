from pydantic import BaseModel, Field
from typing import List

class InvestmentReport(BaseModel):
    company_overview: str = ""
    industry: str = ""
    business_model: str = ""
    latest_news: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    financial_highlights: List[str] = Field(default_factory=list)
    growth_opportunities: List[str] = Field(default_factory=list)
    potential_risks: List[str] = Field(default_factory=list)
    investment_summary: str = ""
