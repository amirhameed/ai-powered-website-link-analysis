import asyncio

import httpx

from fastapi import FastAPI, HTTPException
from app.services.llm_service import AIServiceError, analyze_website_with_gemini
from app.schemas import LinkAnalysisRequest
from app.schemas import LinkAnalysisResponse
from app.services.website_extractor import extract_website_content
from google.genai import errors as genai_errors
from pydantic import ValidationError
from app.utils.validators import validate_public_url

app = FastAPI(
    title="AI Website Link Analysis API",
    description="Analyze websites using AI and generate business insights.",
    version="1.0.0",
)

@app.get("/")
def home():
    return {
        "message": "AI Website Link Analysis API is running",
        "status": "success",
    }

@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse
)
async def analyze_website(request: LinkAnalysisRequest):
    try:

        is_public = await asyncio.to_thread(
            validate_public_url,
            str(request.url),
        )
        if not is_public:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unsafe URL."
            )

        website_data = await extract_website_content(
            str(request.url)
        )
        ai_analysis = await analyze_website_with_gemini(website_data)

        # Step 3: Return complete structured response
        return LinkAnalysisResponse(
            success=True,
            url=str(request.url),
            **ai_analysis.model_dump()
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Website request timed out",
        )

    except httpx.InvalidURL:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsafe URL.",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Website returned HTTP {error.response.status_code}",
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail="Unable to connect to the website",
        )

    except AIServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
        )

    except genai_errors.APIError as exc:
        try:
            code = int(exc.code)
        except (TypeError, ValueError):
            code = None

        if code == 429:
            raise HTTPException(
                status_code=503,
                detail="AI service usage limit reached. Try again later."
            )

        elif code in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="AI service authentication failed."
            )

        elif code is not None and code >= 500:
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable."
            )

        else:
            raise HTTPException(
                status_code=502,
                detail="AI service could not process the request."
            )

    except ValidationError:
        raise HTTPException(
            status_code=502,
            detail="AI returned data that did not match the expected format."
        )