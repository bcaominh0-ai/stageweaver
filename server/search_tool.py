"""
search_server.py – FastMCP server exposing a SearxNG search tool
----------------------------------------------------------------
* search(query, num_results?, category?, language?, time_range?,
          safe_search?, host?)
      – Privacy‑respecting web search powered by any SearxNG instance.

Dependencies
    pip install httpx mcp fastmcp
"""

# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# --------------------------------------------------------------------------- #
#  FastMCP server instance
# --------------------------------------------------------------------------- #

mcp = FastMCP("search")

# --------------------------------------------------------------------------- #
#  Constants & simple validators
# --------------------------------------------------------------------------- #

DEFAULT_HOST = os.getenv("SEARXNG_HOST") or os.getenv("SEARXNG_URL") or "http://127.0.0.1:18080"
DEFAULT_CATEGORY = "general"
SAFE_SEARCH_LEVELS = {0, 1, 2}
VALID_TIME_RANGES = {"day", "week", "month", "year"}
SEARCH_TIMEOUT = float(os.getenv("SEARXNG_SEARCH_TIMEOUT_SEC", "20"))
SEARCH_ATTEMPTS = max(1, int(os.getenv("SEARXNG_SEARCH_ATTEMPTS", "3")))


def _check_safe(level: int) -> None:
    if level not in SAFE_SEARCH_LEVELS:
        raise ValueError(f"safe_search must be 0, 1 or 2 (got {level}).")


def _check_time_range(rng: Optional[str]) -> None:
    if rng is not None and rng not in VALID_TIME_RANGES:
        raise ValueError(
            f"time_range must be one of {sorted(VALID_TIME_RANGES)} (got {rng})."
        )


# --------------------------------------------------------------------------- #
#  Tool
# --------------------------------------------------------------------------- #


@mcp.tool()
async def search(
    query: str,
    num_results: int = 10,
    category: str | None = None,
    language: str = "en",
    time_range: str | None = None,
    safe_search: int = 1,
    host: str = DEFAULT_HOST,
) -> List[Dict[str, str]]:
    """Run a web search via any SearxNG instance (defaults to *nicfab.eu*).

    Args:
        query: The search string.
        num_results: Max results to return (default 10 — max 20 is polite).
        category: SearxNG category: *general*, *images*, *videos*, *news*,
            *map*, *music*, *it*, *science*, *files*, *social media*. 
            Categories not listed here can still be searched with the Search syntax.
            Defaults to *general*.
        language: Two‑letter language code (default "en").
        time_range: Optional freshness filter: "day" | "week" | "month" | "year".
        safe_search: 0 = off • 1 = moderate • 2 = strict (default 1).
        host: Full base‑URL of the SearxNG instance to query.

    Returns:
        A list of dicts with **title**, **link**, **snippet** keys.
    """
    _check_safe(safe_search)
    _check_time_range(time_range)

    params: Dict[str, Any] = {
        "q": query,
        "format": "json",
        "language": language,
        "categories": category or DEFAULT_CATEGORY,
        "pageno": 1,
        "safe": safe_search,
    }
    if time_range:
        params["time_range"] = time_range

    url = f"{host.rstrip('/')}/search"

    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT, headers={"User-Agent": "fastmcp-search"}) as client:
        errors: list[str] = []
        for attempt in range(1, SEARCH_ATTEMPTS + 1):
            try:
                r = await client.get(url, params=params)
                r.raise_for_status()
                payload = r.json()
                results = payload.get("results", [])[: num_results]
                if results:
                    break

                unavailable = payload.get("unresponsive_engines", [])
                detail = f"; unresponsive engines: {unavailable[:6]}" if unavailable else ""
                raise RuntimeError(f"SearXNG returned no results{detail}")
            except Exception as exc:  # network / JSON / upstream engine errors
                errors.append(f"attempt {attempt}: {exc}")
                if attempt < SEARCH_ATTEMPTS:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        else:
            results = []

        if not results:
            return [{
                "title": "Search error",
                "link": "",
                "snippet": " | ".join(errors),
            }]

    return [
        {
            "title": it.get("title", ""),
            "link": it.get("url", ""),
            "snippet": it.get("content", ""),
        }
        for it in results
    ]


# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")  # or: mcp.run(host="0.0.0.0", port=5002)
