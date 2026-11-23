
import gzip
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.main import app
from app.schemas import AIAnalysisResult
from app.services.website_extractor import (
    MAX_CONTENT_SIZE,
    extract_website_content,
)
from app.utils.validators import validate_public_url

client = TestClient(app)

_EXAMPLE_IP = "93.184.216.34"
_real_getaddrinfo = socket.getaddrinfo


def _example_addresses(port):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (_EXAMPLE_IP, port)),
    ]


def _fake_getaddrinfo(host, port, *args, **kwargs):
    name = host[:-1] if isinstance(host, str) and host.endswith(".") else host
    if name == "example.com":
        return _example_addresses(port)
    return _real_getaddrinfo(host, port, *args, **kwargs)


async def _fake_async_getaddrinfo(hostname, port):
    return _fake_getaddrinfo(hostname, port)


@pytest.fixture(autouse=True)
def mock_example_com_dns():
    """Answer example.com locally so tests do not use DNS or the internet."""
    with patch(
        "app.utils.validators.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo,
    ), patch(
        "app.utils.validators._async_getaddrinfo",
        side_effect=_fake_async_getaddrinfo,
    ):
        yield


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200


def test_public_url_is_allowed():
    assert validate_public_url("https://example.com") is True


def test_localhost_is_blocked():
    assert validate_public_url("http://localhost") is False


def test_loopback_ip_is_blocked():
    assert validate_public_url("http://127.0.0.1") is False


def test_private_ip_is_blocked():
    assert validate_public_url("http://192.168.1.1") is False


def test_unsafe_url_returns_400():
    response = client.post(
        "/link-analysis",
        json={"url": "http://127.0.0.1"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or unsafe URL."

def test_invalid_url_returns_422():
    response = client.post(
        "/link-analysis",
        json={"url": "not-a-valid-url"},
    )

    assert response.status_code == 422


def test_missing_url_returns_422():
    response = client.post(
        "/link-analysis",
        json={},
    )

    assert response.status_code == 422

@pytest.mark.asyncio
async def test_website_extraction():
    page = (
        b"<html><head><title>Example Domain</title></head>"
        b"<body><h1>Example Domain</h1>"
        b"<p>Example website content</p></body></html>"
    )
    mock_async_client = _mock_html_response(
        {
            "content-type": "text/html; charset=utf-8",
            "content-length": str(len(page)),
        },
        [page],
    )

    with patch(
        "app.services.website_extractor.httpx.AsyncClient",
        return_value=mock_async_client,
    ):
        result = await extract_website_content("https://example.com")

    assert result["title"] == "Example Domain"
    assert "Example website content" in result["content"]
    assert result["headings"] == ["Example Domain"]
    assert result["word_count"] > 0


def _analysis_result():
    return AIAnalysisResult.model_validate({
        "competitor_analysis": {
            "brand_name": "Example",
            "overview": "A documentation site",
            "category": "Reference",
            "products_services": ["Documentation"],
            "competitors": [{
                "name": "Other Docs",
                "website": "https://docs.example.org",
                "strengths": ["Clear writing"],
                "weaknesses": ["Narrow scope"],
            }],
            "competitive_advantages": ["Simple examples"],
            "positioning_opportunities": ["More tutorials"],
        },
        "search_queries": [{
            "query": "example domain",
            "intent": "informational",
            "suitable_platforms": ["Google"],
            "business_relevance": "Brand lookup",
            "target_audience": "Developers",
        }],
        "seo_analysis": {
            "primary_keywords": ["example"],
            "secondary_keywords": ["domain"],
            "long_tail_keywords": ["example domain documentation"],
            "seo_title": "Example Domain",
            "meta_description": "Example documentation site",
            "content_recommendations": ["Add a tutorial"],
        },
        "services_analysis": {
            "services": [{
                "name": "Documentation",
                "description": "Reference pages",
                "target_industry": "Software",
                "customer_benefit": "Fast answers",
                "is_inferred": False,
            }],
        },
        "target_audience": {
            "primary_audience": {
                "audience_name": "Developers",
                "industries": ["Software"],
                "business_types": ["SaaS"],
                "geography": ["Global"],
                "needs_and_challenges": ["Clear examples"],
                "interests": ["Web standards"],
            },
            "secondary_audiences": [],
            "marketing_segments": ["Technical buyers"],
        },
    })


def test_link_analysis_returns_mocked_analysis():
    website = _website_sample()
    analysis = _analysis_result()

    with patch(
        "app.main.extract_website_content",
        new_callable=AsyncMock,
    ) as mock_extract, patch(
        "app.main.analyze_website_with_gemini",
        new_callable=AsyncMock,
    ) as mock_gemini:
        mock_extract.return_value = website
        mock_gemini.return_value = analysis

        response = client.post(
            "/link-analysis",
            json={"url": "https://example.com"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["url"] == "https://example.com/"
    assert body["competitor_analysis"] == (
        analysis.competitor_analysis.model_dump()
    )
    assert body["search_queries"] == [
        item.model_dump() for item in analysis.search_queries
    ]
    assert body["seo_analysis"] == analysis.seo_analysis.model_dump()
    assert body["services_analysis"] == (
        analysis.services_analysis.model_dump()
    )
    assert body["target_audience"] == analysis.target_audience.model_dump()
    mock_extract.assert_awaited_once_with("https://example.com/")
    mock_gemini.assert_awaited_once_with(website)


def test_gemini_quota_error_returns_503():
    with patch(
        "app.main.extract_website_content",
        new_callable=AsyncMock,
    ) as mock_extract, patch(
        "app.main.analyze_website_with_gemini",
        new_callable=AsyncMock,
    ) as mock_gemini:

        mock_extract.return_value = {
            "title": "Example",
            "description": "Test website",
            "headings": ["Example"],
            "content": "Example website content",
            "word_count": 3,
        }

        mock_gemini.side_effect = genai_errors.APIError(
            429,
            {"message": "Quota exceeded"},
        )

        response = client.post(
            "/link-analysis",
            json={"url": "https://example.com"},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == (
            "AI service usage limit reached. Try again later."
        )



@pytest.mark.asyncio
async def test_website_timeout():
    def handler(request):
        raise httpx.TimeoutException("Connection timed out")

    real_async_client = httpx.AsyncClient

    def client_with_timeout(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    with patch(
        "app.services.website_extractor.httpx.AsyncClient",
        side_effect=client_with_timeout,
    ):
        with pytest.raises(httpx.TimeoutException):
            await extract_website_content("https://example.com")


def test_website_timeout_returns_504():
    with patch(
        "app.main.extract_website_content",
        new_callable=AsyncMock,
    ) as mock_extract:
        mock_extract.side_effect = httpx.TimeoutException(
            "Connection timed out"
        )

        response = client.post(
            "/link-analysis",
            json={"url": "https://example.com"},
        )

    assert response.status_code == 504
    assert response.json()["detail"] == "Website request timed out"


def test_gemini_auth_error_returns_502():
    with patch(
        "app.main.extract_website_content",
        new_callable=AsyncMock,
    ) as mock_extract, patch(
        "app.main.analyze_website_with_gemini",
        new_callable=AsyncMock,
    ) as mock_gemini:

        mock_extract.return_value = {
            "title": "Example",
            "description": "Test website",
            "headings": ["Example"],
            "content": "Example website content",
            "word_count": 3,
        }

        mock_gemini.side_effect = genai_errors.APIError(
            401,
            {"message": "Invalid API key"},
        )

        response = client.post(
            "/link-analysis",
            json={"url": "https://example.com"},
        )

        assert response.status_code == 502
        assert response.json()["detail"] == (
            "AI service authentication failed."
        )


def _website_sample():
    return {
        "title": "Example",
        "description": "Test website",
        "headings": ["Example"],
        "content": "Example website content",
        "word_count": 3,
    }


@pytest.mark.parametrize(
    "status_code,http_status,detail",
    [
        (403, 502, "AI service authentication failed."),
        (500, 503, "AI service is temporarily unavailable."),
    ],
)
def test_gemini_api_errors_return_expected_status(
    status_code,
    http_status,
    detail,
):
    with patch(
        "app.main.extract_website_content",
        new_callable=AsyncMock,
    ) as mock_extract, patch(
        "app.main.analyze_website_with_gemini",
        new_callable=AsyncMock,
    ) as mock_gemini:
        mock_extract.return_value = _website_sample()
        mock_gemini.side_effect = genai_errors.APIError(
            status_code,
            {"message": "upstream failure"},
        )

        response = client.post(
            "/link-analysis",
            json={"url": "https://example.com"},
        )

    assert response.status_code == http_status
    assert response.json()["detail"] == detail


def test_gemini_error_without_code_returns_502():
    with patch(
        "app.main.extract_website_content",
        new_callable=AsyncMock,
    ) as mock_extract, patch(
        "app.main.analyze_website_with_gemini",
        new_callable=AsyncMock,
    ) as mock_gemini:
        mock_extract.return_value = _website_sample()
        mock_gemini.side_effect = genai_errors.APIError(
            None,
            {"message": "unknown"},
        )

        response = client.post(
            "/link-analysis",
            json={"url": "https://example.com"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "AI service could not process the request."
    )


def test_missing_gemini_key_returns_503():
    with patch(
        "app.main.extract_website_content",
        new_callable=AsyncMock,
    ) as mock_extract, patch(
        "app.services.llm_service.settings.gemini_api_key",
        None,
    ):
        mock_extract.return_value = _website_sample()

        response = client.post(
            "/link-analysis",
            json={"url": "https://example.com"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "AI service is not configured."
    assert "Traceback" not in response.text


def test_empty_gemini_response_returns_502():
    mock_response = MagicMock()
    mock_response.text = ""
    mock_models = MagicMock()
    mock_models.generate_content = AsyncMock(return_value=mock_response)
    mock_entered = MagicMock()
    mock_entered.models = mock_models
    mock_client = MagicMock()
    mock_client.aio.__aenter__ = AsyncMock(return_value=mock_entered)
    mock_client.aio.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "app.main.extract_website_content",
        new_callable=AsyncMock,
    ) as mock_extract, patch(
        "app.services.llm_service.get_gemini_client",
        return_value=mock_client,
    ):
        mock_extract.return_value = _website_sample()

        response = client.post(
            "/link-analysis",
            json={"url": "https://example.com"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "AI service returned an empty response."
    )


def test_invalid_httpx_url_returns_400():
    with patch(
        "app.main.extract_website_content",
        new_callable=AsyncMock,
    ) as mock_extract:
        mock_extract.side_effect = httpx.InvalidURL("bad url")

        response = client.post(
            "/link-analysis",
            json={"url": "https://example.com"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or unsafe URL."


@pytest.mark.asyncio
async def test_extractor_invalid_httpx_url_is_safe():
    with patch(
        "app.services.website_extractor.httpx.URL",
        side_effect=httpx.InvalidURL("bad url"),
    ):
        with pytest.raises(ValueError, match="Invalid or unsafe URL"):
            await extract_website_content("https://example.com")


@pytest.mark.asyncio
async def test_invalid_content_length_has_safe_message():
    mock_async_client = _mock_html_response(
        {
            "content-type": "text/html",
            "content-length": "not-a-number",
        },
        [b"<html><title>Ok</title></html>"],
    )

    with patch(
        "app.services.website_extractor.httpx.AsyncClient",
        return_value=mock_async_client,
    ):
        with pytest.raises(
            ValueError,
            match="Website content length is invalid.",
        ):
            await extract_website_content("https://example.com")

@pytest.mark.asyncio
async def test_oversized_website_is_rejected():
    mock_response = MagicMock()

    mock_response.is_redirect = False
    mock_response.headers = {
        "content-type": "text/html",
        "content-length": "2000000",
    }

    mock_response.raise_for_status.return_value = None

    # Mock the HTTP streaming context manager
    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(
        return_value=mock_response
    )
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    # Mock the HTTP client
    mock_http_client = MagicMock()
    mock_http_client.stream.return_value = mock_stream

    mock_async_client = MagicMock()
    mock_async_client.__aenter__ = AsyncMock(
        return_value=mock_http_client
    )
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "app.services.website_extractor.httpx.AsyncClient",
        return_value=mock_async_client,
    ):
        with pytest.raises(
            ValueError,
            match="Website content is too large to analyze.",
        ):
            await extract_website_content("https://example.com")


@pytest.mark.asyncio
async def test_streaming_limit_without_content_length():
    mock_response = MagicMock()

    mock_response.is_redirect = False
    mock_response.headers = {
        "content-type": "text/html"
    }
    mock_response.raise_for_status.return_value = None

    # Simulate two chunks of 600 KB each
    async def fake_chunks():
        yield b"a" * 600_000
        yield b"b" * 600_000

    mock_response.aiter_raw = fake_chunks

    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(
        return_value=mock_response
    )
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    mock_http_client = MagicMock()
    mock_http_client.stream.return_value = mock_stream

    mock_async_client = MagicMock()
    mock_async_client.__aenter__ = AsyncMock(
        return_value=mock_http_client
    )
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "app.services.website_extractor.httpx.AsyncClient",
        return_value=mock_async_client,
    ):
        with pytest.raises(
            ValueError,
            match="Website content is too large to analyze.",
        ):
            await extract_website_content("https://example.com")


def _mock_html_response(headers, chunks):
    mock_response = MagicMock()
    mock_response.is_redirect = False
    mock_response.headers = headers
    mock_response.raise_for_status.return_value = None

    async def fake_chunks():
        for chunk in chunks:
            yield chunk

    mock_response.aiter_raw = fake_chunks

    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    mock_http_client = MagicMock()
    mock_http_client.stream.return_value = mock_stream

    mock_async_client = MagicMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)
    return mock_async_client


@pytest.mark.asyncio
async def test_oversized_chunk_is_rejected():
    mock_async_client = _mock_html_response(
        {"content-type": "text/html"},
        [b"x" * (MAX_CONTENT_SIZE + 1)],
    )

    with patch(
        "app.services.website_extractor.httpx.AsyncClient",
        return_value=mock_async_client,
    ):
        with pytest.raises(
            ValueError,
            match="Website content is too large to analyze.",
        ):
            await extract_website_content("https://example.com")


@pytest.mark.asyncio
async def test_gzip_bomb_is_rejected():
    compressed = gzip.compress(b"a" * (MAX_CONTENT_SIZE + 1))
    assert len(compressed) < MAX_CONTENT_SIZE

    mock_async_client = _mock_html_response(
        {
            "content-type": "text/html",
            "content-encoding": "gzip",
            "content-length": str(len(compressed)),
        },
        [compressed],
    )

    with patch(
        "app.services.website_extractor.httpx.AsyncClient",
        return_value=mock_async_client,
    ):
        with pytest.raises(
            ValueError,
            match="Website content is too large to analyze.",
        ):
            await extract_website_content("https://example.com")


@pytest.mark.asyncio
async def test_normal_html_extraction_from_stream():
    page = b"<html><head><title>Ok</title></head><body><p>Hello world</p></body></html>"
    mock_async_client = _mock_html_response(
        {
            "content-type": "text/html; charset=utf-8",
            "content-length": str(len(page)),
        },
        [page],
    )

    with patch(
        "app.services.website_extractor.httpx.AsyncClient",
        return_value=mock_async_client,
    ):
        result = await extract_website_content("https://example.com")

    assert result["title"] == "Ok"
    assert "Hello world" in result["content"]
    assert result["word_count"] > 0


@pytest.mark.asyncio
async def test_unsupported_content_encoding_is_rejected():
    mock_async_client = _mock_html_response(
        {
            "content-type": "text/html",
            "content-encoding": "br",
        },
        [b"<html><title>Ok</title></html>"],
    )

    with patch(
        "app.services.website_extractor.httpx.AsyncClient",
        return_value=mock_async_client,
    ):
        with pytest.raises(ValueError, match="Unsupported content encoding."):
            await extract_website_content("https://example.com")


UNSAFE_URLS = [
    "http://10.0.0.1",
    "http://172.16.0.1",
    "http://192.168.1.1",
    "http://127.0.0.1",
    "http://127.1",
    "http://0.0.0.0",
    "http://[::]",
    "http://[::1]",
    "http://[fc00::1]",
    "http://[fe80::1]",
    "http://169.254.169.254",
    "http://[::ffff:127.0.0.1]",
    "http://[::ffff:169.254.169.254]",
    "http://[::ffff:7f00:1]",
    "http://224.0.0.1",
    "http://[ff02::1]",
    "http://0177.0.0.1",
    "http://0177.0.0.1.",
    "http://2130706433",
    "http://0x7f000001",
    "http://[2002:7f00:1::]",
    "http://[2002:a9fe:a9fe::]",
    "http://[2002:a00:1::]",
]


def test_public_6to4_address_is_allowed():
    assert validate_public_url("http://[2002:808:808::]") is True


@pytest.mark.parametrize("url", UNSAFE_URLS)
def test_unsafe_ip_representations_are_blocked(url):
    assert validate_public_url(url) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("url", UNSAFE_URLS)
async def test_extractor_rejects_unsafe_ip_before_connecting(url):
    with patch(
        "app.services.website_extractor.httpx.AsyncClient"
    ) as mock_client:
        with pytest.raises(ValueError, match="Invalid or unsafe URL"):
            await extract_website_content(url)

    mock_client.assert_not_called()


def test_dns_answer_pointing_at_private_ip_is_blocked():
    addresses = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 80)),
    ]

    with patch(
        "app.utils.validators.socket.getaddrinfo",
        return_value=addresses,
    ):
        assert validate_public_url("http://rebind.example") is False


def test_mixed_public_and_private_dns_answers_are_blocked():
    addresses = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
    ]

    with patch(
        "app.utils.validators.socket.getaddrinfo",
        return_value=addresses,
    ):
        assert validate_public_url("http://rebind.example") is False


@pytest.mark.asyncio
async def test_connection_is_pinned_to_validated_ip():
    addresses = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    mock_response = MagicMock()
    mock_response.is_redirect = False
    mock_response.headers = {
        "content-type": "text/html; charset=utf-8",
        "content-length": "48",
    }
    mock_response.raise_for_status.return_value = None

    async def fake_chunks():
        yield b"<html><title>Ok</title><body>Hello</body></html>"

    mock_response.aiter_raw = fake_chunks

    mock_stream = MagicMock()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream.__aexit__ = AsyncMock(return_value=None)

    mock_http_client = MagicMock()
    mock_http_client.stream.return_value = mock_stream

    mock_async_client = MagicMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "app.utils.validators._async_getaddrinfo",
        new_callable=AsyncMock,
        return_value=addresses,
    ), patch(
        "app.services.website_extractor.httpx.AsyncClient",
        return_value=mock_async_client,
    ) as mock_client_cls:
        result = await extract_website_content(
            "https://example.com/docs?q=1"
        )

    assert result["title"] == "Ok"
    assert mock_client_cls.call_args.kwargs["follow_redirects"] is False
    assert mock_client_cls.call_args.kwargs["verify"] is True

    method, requested_url = mock_http_client.stream.call_args.args
    stream_kwargs = mock_http_client.stream.call_args.kwargs
    requested = httpx.URL(requested_url)

    assert method == "GET"
    assert requested.host == "93.184.216.34"
    assert requested.path == "/docs"
    assert requested.params["q"] == "1"
    assert stream_kwargs["headers"]["Host"] == "example.com"
    assert stream_kwargs["extensions"]["sni_hostname"] == "example.com"
