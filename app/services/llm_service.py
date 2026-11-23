
import json

from google import genai
from google.genai import types

from app.config import settings
from app.schemas import AIAnalysisResult


class AIServiceError(Exception):
    """An AI failure that the API should return as an HTTP error."""

    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def get_gemini_client():
    if not settings.gemini_api_key:
        raise AIServiceError(
            503,
            "AI service is not configured.",
        )

    return genai.Client(
        api_key=settings.gemini_api_key
    )


async def analyze_website_with_gemini(
    website_data: dict,
) -> AIAnalysisResult:

    client = get_gemini_client()

    website_text = json.dumps(
        website_data,
        ensure_ascii=False,
    )

    prompt = f"""
    You are an experienced digital marketing and SEO analyst.

    Analyze the website information below.

    WEBSITE INFORMATION:
    {website_text}

    Prepare these five sections:

    1. COMPETITOR ANALYSIS
    Identify the brand, overview, category, products and services.
    Identify at least 3 genuine competitors when reliable information
    is available. Never invent competitor names or websites.
    Include strengths, weaknesses, competitive advantages and
    positioning opportunities.

    2. SEARCH QUERIES
    Provide 5 relevant search queries.
    Include search intent, suitable platforms, business relevance
    and target audience.
    Do not invent search volume or ranking statistics.

    3. SEO ANALYSIS
    Provide primary, secondary and long-tail keywords,
    an SEO title, meta description and content recommendations.

    4. SERVICES ANALYSIS
    Provide 5 relevant services where possible.
    Clearly distinguish services found on the website from
    recommendations inferred by you.

    5. TARGET AUDIENCE
    Identify primary and secondary audiences, industries,
    business types, geography, needs, challenges and interests.

    IMPORTANT:
    - Use evidence from the supplied website information.
    - Do not invent facts, competitors or customer statistics.
    - If information is unavailable, say so clearly.
    - Recommendations must be identified as recommendations.
    """

    async with client.aio as async_client:
        response = await async_client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AIAnalysisResult.model_json_schema(),
                temperature=0.2,
                max_output_tokens=4000,
            ),
        )

    if not response.text:
        raise AIServiceError(
            502,
            "AI service returned an empty response.",
        )

    return AIAnalysisResult.model_validate_json(response.text)
