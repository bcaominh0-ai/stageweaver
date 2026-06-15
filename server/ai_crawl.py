import contextlib
import os, time, asyncio
import logging
import math
import re
import sys
from typing import Dict, Any, List, Iterable
from dotenv import load_dotenv

import colorlog
import httpx
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
MAX_EXTRACT_INPUT_CHARS = int(os.getenv("CRAWL_EXTRACT_MAX_INPUT_CHARS", "60000"))
CHUNK_WORDS = int(os.getenv("CRAWL_EXTRACT_CHUNK_WORDS", "900"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CRAWL_EXTRACT_CHUNK_OVERLAP_WORDS", "90"))
TOP_CHUNKS = int(os.getenv("CRAWL_EXTRACT_TOP_CHUNKS", "10"))
ANCHOR_CHARS = int(os.getenv("CRAWL_EXTRACT_ANCHOR_CHARS", "4000"))
CHUNK_EXCERPT_CHARS = int(os.getenv("CRAWL_EXTRACT_CHUNK_EXCERPT_CHARS", "6000"))
RERANK_MODEL = _env_nonempty("CRAWL_RERANK_MODEL") or "Qwen/Qwen3-Reranker-8B"
RERANK_CANDIDATES = int(os.getenv("CRAWL_RERANK_CANDIDATES", "32"))
RERANK_DOC_CHARS = int(os.getenv("CRAWL_RERANK_DOC_CHARS", "2500"))
RERANK_TIMEOUT = float(os.getenv("CRAWL_RERANK_TIMEOUT", "30"))

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
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]{1,}")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
EMPTY_MARKDOWN_LINK_RE = re.compile(r"\[\]\([^)]+\)")
WIKI_FILE_LINK_RE = re.compile(r"\[[^\]]*\]\(https?://[^)]*/wiki/File:[^)]+\)")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
WIKI_SECTION_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$")
WIKI_TRAILING_SECTIONS = {
    "notes",
    "references",
    "bibliography",
    "further reading",
    "external links",
    "sources",
}
WIKI_NAV_SECTIONS = {
    "contents",
    "navigation",
    "tools",
    "personal tools",
    "appearance",
    "languages",
    "in other projects",
    "print/export",
}
WIKI_NOISE_LINES = {
    "appearance",
    "contribute",
    "search",
    "jump to content",
    "log in",
    "main menu",
    "main menu move to sidebar hide",
    "move to sidebar hide",
    "navigation",
    "contents",
    "current events",
    "random article",
    "about wikipedia",
    "contact us",
    "donate",
    "help",
    "learn to edit",
    "community portal",
    "recent changes",
    "upload file",
    "special pages",
    "permanent link",
    "page information",
    "cite this page",
    "get shortened url",
    "download qr code",
    "wikidata item",
}
WIKI_LANGUAGE_LINE_RE = re.compile(r"^\d+\s+languages?$", re.IGNORECASE)
WIKI_REAL_TITLE_RE = re.compile(r"^#\s+(?!contents$|navigation$|tools$|appearance$|languages$|personal tools$)(.+)", re.IGNORECASE)
WIKI_COORD_RE = re.compile(r"^Coordinates?:", re.IGNORECASE)
STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "and", "are",
    "because", "been", "before", "being", "between", "can", "could", "did",
    "does", "doing", "down", "during", "each", "for", "from", "further",
    "had", "has", "have", "having", "here", "how", "into", "itself",
    "more", "most", "only", "other", "over", "same", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "under", "until", "very", "what", "when", "where", "which",
    "while", "who", "why", "will", "was", "were", "with", "would", "your",
    "summarize", "summary", "page",
}


def _normalize_markdown(md: str) -> str:
    """Reduce markup noise while preserving readable page facts."""
    md = MARKDOWN_IMAGE_RE.sub("", md)
    md = WIKI_FILE_LINK_RE.sub("", md)
    md = EMPTY_MARKDOWN_LINK_RE.sub("", md)
    md = MARKDOWN_LINK_RE.sub(r"\1", md)
    raw_lines = md.splitlines()
    first_title = next(
        (idx for idx, raw_line in enumerate(raw_lines) if WIKI_REAL_TITLE_RE.match(raw_line.strip())),
        None,
    )
    if first_title and first_title > 0:
        raw_lines = raw_lines[first_title:]
    if raw_lines and WIKI_REAL_TITLE_RE.match(raw_lines[0].strip()):
        first_content_idx = None
        for idx, raw_line in enumerate(raw_lines[1:], start=1):
            if raw_line.strip():
                first_content_idx = idx
                break

        bullet_lines_after_title = 0
        language_menu_after_title = bool(
            first_content_idx is not None
            and WIKI_LANGUAGE_LINE_RE.match(raw_lines[first_content_idx].strip())
        )
        start_idx = (first_content_idx + 1) if language_menu_after_title else 1
        for raw_line in raw_lines[start_idx:]:
            stripped_candidate = raw_line.strip()
            if not stripped_candidate:
                continue
            if stripped_candidate.startswith(("* ", "  * ")):
                bullet_lines_after_title += 1
                continue
            break

        body_start = None
        if language_menu_after_title or bullet_lines_after_title >= 20:
            for idx, raw_line in enumerate(raw_lines[start_idx:], start=start_idx):
                stripped_candidate = raw_line.strip()
                if (
                    not stripped_candidate
                    or stripped_candidate.startswith(("* ", "  * "))
                    or stripped_candidate.startswith("!")
                    or stripped_candidate.endswith('")')
                ):
                    continue
                if WIKI_COORD_RE.match(stripped_candidate) or stripped_candidate.startswith("|") or len(stripped_candidate.split()) >= 8:
                    body_start = idx
                    break
            if body_start and body_start > 1:
                raw_lines = [raw_lines[0], ""] + raw_lines[body_start:]

    lines = []
    blank_count = 0
    for raw_line in raw_lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        normalized = re.sub(r"\s+", " ", stripped).strip(" -*#").lower()
        if not stripped:
            blank_count += 1
            if blank_count <= 1:
                lines.append("")
            continue
        blank_count = 0

        section_match = WIKI_SECTION_RE.match(stripped)
        if section_match:
            section_title = re.sub(r"\s+", " ", section_match.group(1)).strip().lower()
            if section_title in WIKI_TRAILING_SECTIONS:
                continue
            if section_title in WIKI_NAV_SECTIONS:
                continue

        if normalized in WIKI_NOISE_LINES or WIKI_LANGUAGE_LINE_RE.match(stripped):
            continue
        if normalized.startswith(("from wikipedia", "retrieved from", "categories:", "hidden categories:")):
            continue
        if "wikipedia" in normalized and any(token in normalized for token in ("portal", "sister", "project", "commons", "wikivoyage")):
            continue
        if stripped.startswith(("  * ", "* ")) and any(
            marker in normalized
            for marker in (
                "main page", "contents", "current events", "random article",
                "about wikipedia", "contact us", "donate", "special pages",
                "permanent link", "page information", "cite this page",
                "external links", "wikipedia articles", "wikimedia commons",
            )
        ):
            continue

        # Wikipedia and search result pages often emit dense nav/link lists.
        if stripped.count(" | ") >= 8:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _query_terms(query: str) -> set[str]:
    return {
        term
        for term in (token.lower() for token in WORD_RE.findall(query))
        if len(term) > 2 and term not in STOPWORDS
    }


def _iter_word_chunks(text: str, chunk_words: int, overlap_words: int) -> Iterable[tuple[int, str]]:
    words = text.split()
    if not words:
        return
    step = max(1, chunk_words - max(0, overlap_words))
    for chunk_id, start in enumerate(range(0, len(words), step)):
        chunk = " ".join(words[start : start + chunk_words])
        if chunk:
            yield chunk_id, chunk
        if start + chunk_words >= len(words):
            break


def _score_chunk(chunk: str, terms: set[str]) -> float:
    if not terms:
        return 0.0
    tokens = [token.lower() for token in WORD_RE.findall(chunk)]
    if not tokens:
        return 0.0
    counts = {term: 0 for term in terms}
    for token in tokens:
        if token in counts:
            counts[token] += 1
    matched_terms = sum(1 for value in counts.values() if value)
    total_hits = sum(counts.values())
    if matched_terms == 0:
        return 0.0
    return (matched_terms * 3.0 + total_hits) / math.sqrt(len(tokens))


def _excerpt_chunk(chunk: str, terms: set[str], max_chars: int = CHUNK_EXCERPT_CHARS) -> str:
    if len(chunk) <= max_chars or not terms:
        return chunk
    lower = chunk.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
    if not positions:
        return chunk[:max_chars].rstrip() + "\n...[truncated]"
    center = min(positions)
    start = max(0, center - max_chars // 3)
    end = min(len(chunk), start + max_chars)
    prefix = "...[chunk prefix omitted]\n" if start > 0 else ""
    suffix = "\n...[chunk suffix omitted]" if end < len(chunk) else ""
    return prefix + chunk[start:end].strip() + suffix


def _fit_sections(sections: list[tuple[str, str]], max_chars: int) -> str:
    packed = []
    used = 0
    for label, chunk in sections:
        header = f"\n\n[{label}]\n"
        addition = header + chunk.strip()
        if packed and used + len(addition) > max_chars:
            remaining = max_chars - used - len(header) - len("\n...[truncated]")
            if remaining > 500:
                packed.append(header + chunk.strip()[:remaining].rstrip() + "\n...[truncated]")
                used = max_chars
                break
            continue
        if not packed and len(addition) > max_chars:
            packed.append(addition[:max_chars].rstrip() + "\n...[truncated]")
            break
        packed.append(addition)
        used += len(addition)
    return "".join(packed).strip()


def _pack_ranked_chunks(
    original_md: str,
    cleaned: str,
    chunks: list[tuple[int, str]],
    ranked_ids: list[int],
    terms: set[str],
    max_chars: int,
    label: str,
) -> str:
    anchor_chars = min(ANCHOR_CHARS, max(500, max_chars // 8))
    selected: list[tuple[str, str]] = [("Page head", cleaned[:anchor_chars])]
    chunk_by_id = {chunk_id: chunk for chunk_id, chunk in chunks}
    selected.extend(
        (f"Relevant chunk {chunk_id}", _excerpt_chunk(chunk_by_id[chunk_id], terms))
        for chunk_id in ranked_ids
        if chunk_id in chunk_by_id
    )
    tail = cleaned[-anchor_chars:]
    if tail and tail not in selected[0][1]:
        selected.append(("Page tail", tail))
    packed = _fit_sections(selected, max_chars)
    return (
        f"[Compressed crawl content from {len(original_md):,} to {len(packed):,} characters "
        f"using {label}.]\n{packed}"
    )


def _select_relevant_markdown(md: str, query: str, max_chars: int = MAX_EXTRACT_INPUT_CHARS) -> str:
    cleaned = _normalize_markdown(md)
    if len(cleaned) <= max_chars:
        return cleaned

    chunks = list(_iter_word_chunks(cleaned, CHUNK_WORDS, CHUNK_OVERLAP_WORDS))
    if not chunks:
        return cleaned[:max_chars].rstrip() + "\n...[truncated]"

    terms = _query_terms(query)
    scored = [
        (chunk_id, chunk, _score_chunk(chunk, terms))
        for chunk_id, chunk in chunks
    ]
    ranked_ids = [
        chunk_id
        for chunk_id, _chunk, _score in sorted(scored, key=lambda item: item[2], reverse=True)
    ][:TOP_CHUNKS]

    return _pack_ranked_chunks(
        md,
        cleaned,
        chunks,
        ranked_ids,
        terms,
        max_chars,
        "query-aware lexical chunk selection",
    )


def _rerank_endpoint() -> str:
    base_url = _env_nonempty("CRAWL_RERANK_BASE_URL", "CRAWL_EXTRACT_BASE_URL", "JUDGE_BASE_URL", "OPENAI_BASE_URL")
    if not base_url:
        base_url = "https://api.siliconflow.cn/v1"
    base_url = base_url.rstrip("/")
    if base_url.endswith("/rerank"):
        return base_url
    return f"{base_url}/rerank"


def _rerank_api_key() -> str:
    return _env_nonempty("CRAWL_RERANK_API_KEY", "CRAWL_EXTRACT_API_KEY", "JUDGE_API_KEY", "OPENAI_API_KEY")


def _candidate_chunk_ids(scored: list[tuple[int, str, float]], total_chunks: int) -> list[int]:
    if total_chunks <= 0:
        return []
    ranked = [chunk_id for chunk_id, _chunk, _score in sorted(scored, key=lambda item: item[2], reverse=True)]
    ids: dict[int, None] = {}
    for chunk_id in ranked[: max(1, RERANK_CANDIDATES)]:
        ids[chunk_id] = None
    ids[0] = None
    ids[total_chunks - 1] = None
    sample_count = min(8, total_chunks)
    if sample_count > 2:
        for i in range(1, sample_count - 1):
            ids[round(i * (total_chunks - 1) / (sample_count - 1))] = None
    return list(ids.keys())[: max(1, RERANK_CANDIDATES)]


async def _rerank_chunk_ids(query: str, chunks: list[tuple[int, str]], candidate_ids: list[int], terms: set[str]) -> list[int]:
    api_key = _rerank_api_key()
    if not api_key or not candidate_ids:
        return []

    chunk_by_id = {chunk_id: chunk for chunk_id, chunk in chunks}
    documents = [
        _excerpt_chunk(chunk_by_id[chunk_id], terms, max_chars=RERANK_DOC_CHARS)
        for chunk_id in candidate_ids
        if chunk_id in chunk_by_id
    ]
    if not documents:
        return []

    payload: Dict[str, Any] = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(TOP_CHUNKS, len(documents)),
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=RERANK_TIMEOUT) as client:
        response = await client.post(_rerank_endpoint(), json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    ranked_ids = []
    for result in data.get("results", []):
        try:
            document_index = int(result["index"])
            ranked_ids.append(candidate_ids[document_index])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return ranked_ids


async def _select_relevant_markdown_with_rerank(
    md: str,
    query: str,
    max_chars: int = MAX_EXTRACT_INPUT_CHARS,
) -> str:
    cleaned = _normalize_markdown(md)
    if len(cleaned) <= max_chars:
        return cleaned

    chunks = list(_iter_word_chunks(cleaned, CHUNK_WORDS, CHUNK_OVERLAP_WORDS))
    if not chunks:
        return cleaned[:max_chars].rstrip() + "\n...[truncated]"

    terms = _query_terms(query)
    scored = [
        (chunk_id, chunk, _score_chunk(chunk, terms))
        for chunk_id, chunk in chunks
    ]
    lexical_ids = [
        chunk_id
        for chunk_id, _chunk, _score in sorted(scored, key=lambda item: item[2], reverse=True)
    ][:TOP_CHUNKS]

    try:
        candidate_ids = _candidate_chunk_ids(scored, len(chunks))
        reranked_ids = await _rerank_chunk_ids(query, chunks, candidate_ids, terms)
    except Exception as exc:
        logger.warning(f"Rerank failed; falling back to lexical chunk selection: {exc}")
        reranked_ids = []

    if not reranked_ids:
        return _pack_ranked_chunks(
            md,
            cleaned,
            chunks,
            lexical_ids,
            terms,
            max_chars,
            "query-aware lexical chunk selection",
        )

    return _pack_ranked_chunks(
        md,
        cleaned,
        chunks,
        reranked_ids,
        terms,
        max_chars,
        f"{RERANK_MODEL} rerank after lexical preselection",
    )


async def _extract_for_query(
    backend: OpenAIBackend,
    md: str,
    query: str,
    *,
    max_tokens: int = 30000,
    temperature: float = 0.1,
) -> str:
    selected_md = await _select_relevant_markdown_with_rerank(md, query)
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"User Query:\n{query}\n\nPage Markdown (query-filtered):\n\n{selected_md}"},
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
