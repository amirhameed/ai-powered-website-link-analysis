# AI Website Link Analysis API

FastAPI service that downloads a public web page, extracts its text, and asks Gemini for structured marketing analysis. The response covers competitors, search queries, SEO, services, and target audiences.

## Features

- `POST /link-analysis` accepts a public `http` or `https` URL and returns structured JSON.
- Page text is extracted with HTTPX and BeautifulSoup.
- Gemini returns JSON that is validated with Pydantic before it is sent to the client.
- Private and disguised addresses are rejected before a connection is opened.
- Downloads are limited to 1 MB, including content that expands after gzip decompression.

## Python setup and installation

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the tests, also install pytest:

```bash
pip install pytest pytest-asyncio
```

## Environment configuration

Copy the example file and replace the placeholders with your own keys. Do not commit the real `.env` file.

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes, for analysis | Gemini API key |
| `GEMINI_MODEL` | No | Defaults to `gemini-3.5-flash-lite` |

`POST /link-analysis` uses Gemini. If `GEMINI_API_KEY` is missing, the endpoint returns HTTP 503.

## Run the server

From the project root, with the virtual environment active:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 4000
```

## Swagger documentation

Interactive API docs are available at:

http://127.0.0.1:4000/docs

## Example request

```bash
curl -X POST http://127.0.0.1:4000/link-analysis \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

A successful response has `success`, the requested `url`, and five analysis objects: `competitor_analysis`, `search_queries`, `seo_analysis`, `services_analysis`, and `target_audience`.

## Run the tests

From the project root:

```bash
pytest -v
```

`pytest.ini` adds the project root to the Python path so the `app` package can be imported.

## Security protections

- Only `http` and `https` URLs are accepted. Localhost, `.local` names, and non-canonical IP spellings are rejected before DNS lookup.
- Every DNS answer is checked. Private, loopback, link-local, multicast, reserved, and unspecified IPv4 and IPv6 addresses are rejected, including unsafe IPv4-mapped IPv6 and 6to4 addresses.
- The download connects to the validated IP. The original hostname is kept for the `Host` header, TLS SNI, and certificate verification. TLS verification stays enabled, and redirects are not followed.
- The response body is streamed with a 1,000,000 byte limit. Only identity and gzip encodings are accepted, and gzip output is capped before it is stored.
- Upstream failures return fixed HTTP messages. Invalid content length, an unusable URL, a missing Gemini key, and Gemini API errors do not expose Python exception text.
