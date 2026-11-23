
import re
import zlib

import httpx
from bs4 import BeautifulSoup

from app.utils.validators import resolve_pinned_connection_async

MAX_CONTENT_SIZE = 1_000_000
_TOO_LARGE = "Website content is too large to analyze."


def _content_encoding(headers) -> str:
    value = headers.get("content-encoding")
    if value is None:
        return "identity"
    value = value.strip().lower()
    if value in ("", "identity"):
        return "identity"
    if value == "gzip":
        return "gzip"
    raise ValueError("Unsupported content encoding.")


async def _read_limited_body(response) -> bytes:
    """Read at most MAX_CONTENT_SIZE bytes, including after gzip expansion."""
    encoding = _content_encoding(response.headers)
    content = bytearray()
    raw_size = 0
    decoder = None
    if encoding == "gzip":
        decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)

    async for chunk in response.aiter_raw():
        if not chunk:
            continue
        if len(chunk) > MAX_CONTENT_SIZE - raw_size:
            raise ValueError(_TOO_LARGE)
        raw_size += len(chunk)

        if decoder is None:
            content.extend(chunk)
            continue

        remaining = MAX_CONTENT_SIZE - len(content)
        produced = decoder.decompress(chunk, remaining + 1)
        if len(produced) > remaining or decoder.unconsumed_tail:
            raise ValueError(_TOO_LARGE)
        content.extend(produced)

    if decoder is not None and not decoder.eof:
        raise ValueError(_TOO_LARGE)

    return bytes(content)


async def extract_website_content(url: str) -> dict:
    # Resolve and check the address here, then connect to that IP.
    # The original hostname stays in Host and, for HTTPS, in TLS SNI.
    target = await resolve_pinned_connection_async(url)
    try:
        pinned_url = httpx.URL(url).copy_with(host=target.ip)
    except httpx.InvalidURL:
        raise ValueError("Invalid or unsafe URL.")
    extensions = None
    if target.sni_hostname:
        extensions = {"sni_hostname": target.sni_hostname}

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=False,
        verify=True,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; WebsiteAnalyzer/1.0)"
        },
    ) as client:

        async with client.stream(
            "GET",
            pinned_url,
            headers={
                "Host": target.host_header,
                "Accept-Encoding": "gzip",
            },
            extensions=extensions,
        ) as response:
            if response.is_redirect:
                raise ValueError("Website redirects are not currently supported.")

            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()

            if (
                "text/html" not in content_type
                and "application/xhtml+xml" not in content_type
            ):
                raise ValueError("The provided URL does not return an HTML page.")

            content_length = response.headers.get("content-length")

            if content_length:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    raise ValueError("Website content length is invalid.")
                if declared_length > MAX_CONTENT_SIZE:
                    raise ValueError("Website content is too large to analyze.")

            content = await _read_limited_body(response)

    html = content.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")

    for element in soup(
        ["script", "style", "nav", "footer", "header",
        "noscript", "svg", "iframe", "form"]
    ):
        element.decompose()

    title = soup.title.get_text(strip=True) if soup.title else None

    description_tag = soup.find(
        "meta",
        attrs={"name": "description"}
    )

    description = (
        description_tag.get("content")
        if description_tag
        else None
    )

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    headings = []
    for heading in main_content.find_all(["h1", "h2", "h3"]):
        heading_text = heading.get_text(" ", strip=True)

        if heading_text and heading_text not in headings:
            headings.append(heading_text)

    raw_text = main_content.get_text(separator=" ", strip=True)

    clean_text = re.sub(r"\s+", " ", raw_text).strip()

    word_count = len(clean_text.split())

    return {
        "title": title,
        "description": description,
        "headings": headings[:30],
        "content": clean_text[:15000],
        "word_count": word_count,
    }
