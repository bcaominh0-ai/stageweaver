import contextlib
import os, time, asyncio
import logging
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

import colorlog
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from openai import AsyncOpenAI
from crawl4ai import AsyncWebCrawler
from mcp.server.fastmcp import FastMCP
load_dotenv()

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("mcp.crawl_extract")


def _env_nonempty(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        text = value.strip()
        if text and text.upper() not in {"EMPTY", "NONE", "NULL"}:
            return text
    return ""

# --------------------------------------------------------------------------- #
#  Configurable model
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = _env_nonempty("CRAWL_EXTRACT_MODEL", "JUDGE_MODEL", "META_MODEL") or "gpt-4.1"
EXPLICIT_CRAWL_EXTRACT_OVERRIDE = any(
    _env_nonempty(name)
    for name in ("CRAWL_EXTRACT_API_KEY", "CRAWL_EXTRACT_BASE_URL", "CRAWL_EXTRACT_MODEL")
)

# Customizable Parameters
RATE_LIMIT = 10
RATE_INTERVAL = 60

EXTRACTOR_SYSTEM_PROMPT = (
    "You are a careful, concise information extraction assistant. "
    "Given a user query and a webpage in Markdown, do NOT summarize the whole page. "
    "Instead, extract ONLY the content that directly answers or is highly relevant to the query. "
    "If the answer is not present, say so explicitly.\n\n"
    "Output format:\n"
    "1) Direct Answer: 2–4 concise sentences (or 'Not found in page').\n"
    "2) Key Evidence: Bullet list of short quotes from the Markdown (quote verbatim, minimal trimming).\n"
    "3) Entities/Numbers: Bullet list of important names, dates, figures tied to the query.\n"
    "4) Uncertainties: Note any ambiguities or missing info.\n"
    "Stay grounded in the provided Markdown. Avoid fabrication."
)

# --------------------------------------------------------------------------- #
#  Rate Limiting
# --------------------------------------------------------------------------- #
class RateLimiting:
    def __init__(self, max_calls: int, interval: int):
        self.max_calls = max_calls
        self.interval = interval
        self.calls = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.interval]
            if len(self.calls) >= self.max_calls:
                raise RuntimeError(f"Rate limit exceeded. Max calls: {self.max_calls}, Interval: {self.interval} ")
            self.calls.append(now)

rate_limiter = RateLimiting(RATE_LIMIT, RATE_INTERVAL)
# --------------------------------------------------------------------------- #
#  Chat backend
# --------------------------------------------------------------------------- #
class OpenAIBackend:
    def __init__(self):
        if EXPLICIT_CRAWL_EXTRACT_OVERRIDE:
            api_key = _env_nonempty("CRAWL_EXTRACT_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "CRAWL_EXTRACT_* override is active but CRAWL_EXTRACT_API_KEY is missing. "
                    "Fill CRAWL_EXTRACT_API_KEY for the configured extraction endpoint."
                )
            base_url = _env_nonempty("CRAWL_EXTRACT_BASE_URL") or "https://api.openai.com/v1"
        else:
            api_key = _env_nonempty("JUDGE_API_KEY", "OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "Missing crawl extraction API key. Set CRAWL_EXTRACT_API_KEY, JUDGE_API_KEY, or OPENAI_API_KEY."
                )
            base_url = _env_nonempty("JUDGE_BASE_URL", "OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.model = DEFAULT_MODEL
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 30000,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = await self.client.chat.completions.create(**payload)
        msg = resp.choices[0].message
        return {"content": msg.content or ""}

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
async def _extract_for_query(
    backend: OpenAIBackend,
    md: str,
    query: str,
    *,
    max_tokens: int = 30000,
    temperature: float = 0.1,
) -> str:
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"User Query:\n{query}\n\nPage Markdown (verbatim):\n\n{md}"},
    ]
    res = await backend.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)
    return (res["content"] or "").strip()

async def _crawl_markdown(url: str) -> str:
    # MCP stdio uses stdout for JSON-RPC; Crawl4AI progress output must not
    # leak there or the client will fail to parse protocol messages.
    with contextlib.redirect_stdout(sys.stderr):
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        md = (result.markdown or "").strip()
        if not md:
            raise RuntimeError(f"No markdown extracted from {url}")
            return result
        return md

async def _crawl_and_extract(
    url: str,
    query: str,
    *,
    max_tokens: int = 30000,
    temperature: float = 0.1,
) -> str:
    logger.info(f"Crawling: {url}")
    md = await _crawl_markdown(url)
    logger.info(f"Markdown length: {len(md):,} characters")
    backend = OpenAIBackend()  
    return await _extract_for_query(
        backend,
        md,
        query=query,
        max_tokens=max_tokens,
        temperature=temperature,
    )

# --------------------------------------------------------------------------- #
#  FastMCP server
# --------------------------------------------------------------------------- #
mcp = FastMCP("crawl-extract")

@mcp.tool()
async def crawl_extract(
    url: str,
    query: str,
    temperature: float = 0.1,
    max_tokens: int = 30000,
) -> str:
    """
    Crawl a URL to Markdown and extract only the content relevant to the query.

    Parameters
    ----------
    url : str
        Target URL to crawl.
    query : str
        The information need; used to extract only the most relevant snippets from the page Markdown.
    temperature : float, optional (default: 0.1)
        Sampling temperature for the extraction model.
    max_tokens : int, optional (default: 1400)
        Maximum tokens allowed in the extraction model response.

    Returns
    -------
    str
        A compact, four-part extraction (Direct Answer, Key Evidence, Entities/Numbers, Uncertainties).

    Notes
    -----
    - Exposed as the single MCP tool.
    - Set CRAWL_EXTRACT_MODEL to choose the extraction model.
    """
    await rate_limiter.acquire()

    return await _crawl_and_extract(
        url=url,
        query=query,
        temperature=temperature,
        max_tokens=max_tokens,
    )

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    mcp.run(transport="stdio")
