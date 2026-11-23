
from pydantic import BaseModel, HttpUrl, Field


class LinkAnalysisRequest(BaseModel):
    url: HttpUrl


class Competitor(BaseModel):
    name: str
    website: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class CompetitorAnalysis(BaseModel):
    brand_name: str
    overview: str
    category: str
    products_services: list[str] = Field(default_factory=list)
    competitors: list[Competitor] = Field(default_factory=list)
    competitive_advantages: list[str] = Field(default_factory=list)
    positioning_opportunities: list[str] = Field(default_factory=list)


class SearchQuery(BaseModel):
    query: str
    intent: str
    suitable_platforms: list[str] = Field(default_factory=list)
    business_relevance: str
    target_audience: str


class SEOAnalysis(BaseModel):
    primary_keywords: list[str] = Field(default_factory=list)
    secondary_keywords: list[str] = Field(default_factory=list)
    long_tail_keywords: list[str] = Field(default_factory=list)
    seo_title: str
    meta_description: str
    content_recommendations: list[str] = Field(default_factory=list)


class Service(BaseModel):
    name: str
    description: str
    target_industry: str
    customer_benefit: str
    is_inferred: bool = False


class ServicesAnalysis(BaseModel):
    services: list[Service] = Field(default_factory=list)


class AudienceProfile(BaseModel):
    audience_name: str
    industries: list[str] = Field(default_factory=list)
    business_types: list[str] = Field(default_factory=list)
    geography: list[str] = Field(default_factory=list)
    needs_and_challenges: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)


class TargetAudienceAnalysis(BaseModel):
    primary_audience: AudienceProfile
    secondary_audiences: list[AudienceProfile] = Field(default_factory=list)
    marketing_segments: list[str] = Field(default_factory=list)


class WebsiteAnalysis(BaseModel):
    """The five analysis sections shared by Gemini and the API response."""

    competitor_analysis: CompetitorAnalysis
    search_queries: list[SearchQuery]
    seo_analysis: SEOAnalysis
    services_analysis: ServicesAnalysis
    target_audience: TargetAudienceAnalysis


class AIAnalysisResult(WebsiteAnalysis):
    pass


class LinkAnalysisResponse(WebsiteAnalysis):
    success: bool
    url: str
