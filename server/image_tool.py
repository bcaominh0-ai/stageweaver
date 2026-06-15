"""
mcp_image_analysis.py
FastMCP server – vision tools (image → caption / VQA)
"""

# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #
import base64
import io
import os
from typing import Optional

import anyio
import openai
import requests
from PIL import Image
from urllib.parse import urlparse
from openai import AsyncOpenAI              # ← new

from mcp.server.fastmcp import FastMCP
from loguru import logger


from dotenv import load_dotenv
load_dotenv(".env")

def _env_nonempty(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        return ""
    return value.strip()


VISION_MODEL = _env_nonempty("VISION_MODEL") or "gemini-2.5-pro-preview-05-06"
EXPLICIT_VISION_OVERRIDE = any(
    _env_nonempty(name)
    for name in ("VISION_OPENAI_API_KEY", "VISION_OPENAI_BASE_URL", "VISION_MODEL")
)

REMOTE_IMAGE_HEADERS = [
    {"User-Agent": "Mozilla/5.0 (StageWeaver image_tool)"},
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    },
]


def _build_vision_client() -> AsyncOpenAI:
    if EXPLICIT_VISION_OVERRIDE:
        api_key = _env_nonempty("VISION_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VISION_* override is active but VISION_OPENAI_API_KEY is missing. "
                "Fill VISION_OPENAI_API_KEY for the configured vision endpoint."
            )
        base_url = _env_nonempty("VISION_OPENAI_BASE_URL") or "https://api.openai.com/v1"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for image_tool fallback configuration.")
        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


# --------------------------------------------------------------------------- #
#  Helper class
# --------------------------------------------------------------------------- #
class ImageAnalysisToolkit:
    """
    Very small wrapper around OpenAI Vision GPT models.
    Provides two public coroutines:
        • image_to_text
        • ask_question_about_image
    """

    def __init__(self, timeout: float | None = None):
        self.timeout = timeout or 15

    # ---------------- public API ------------------------------------------ #
    async def image_to_text(
        self, image_path: str, sys_prompt: Optional[str] = None
    ) -> str:
        """
        Return a detailed caption of *image_path*.
        """
        default_sys = (
            "You are an expert image analyst. Provide a rich, concise "
            "description of everything visible, including any text."
        )
        return await self._chat_with_image(
            image_path,
            user_prompt="Please describe the contents of this image.",
            system_prompt=sys_prompt or default_sys,
        )

    async def ask_question_about_image(
        self,
        image_path: str,
        question: str,
        sys_prompt: Optional[str] = None,
    ) -> str:
        """
        Answer *question* about *image_path*.
        """
        default_sys = (
            "You answer questions about images by careful visual inspection, "
            "reading any text, and reasoning from what you see. Please consider the reqirements of the question carefully"
        )
        return await self._chat_with_image(
            image_path,
            user_prompt=question,
            system_prompt=sys_prompt or default_sys,
        )

    # ---------------- implementation -------------------------------------- #
    async def _chat_with_image(
        self, image_path: str, user_prompt: str, system_prompt: str
    ) -> str:
        """
        Core routine: prepare image, run OpenAI vision chat, return content.
        """
        image_url = await self._prepare_image(image_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        openai_client = _build_vision_client()

        try:
            logger.info("Sending image to OpenAI ChatCompletion (vision)…")
            response = await openai_client.chat.completions.create(
                model=VISION_MODEL,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            raise

    async def _prepare_image(self, path: str) -> str:
        """
        Turn *path* (local path or URL) into a URL or data‑URL acceptable to
        the OpenAI Vision endpoint.
        """
        parsed = urlparse(path)

        # Remote URL – download it first so all backends receive a data URL.
        if parsed.scheme in ("http", "https"):
            logger.debug(f"Downloading remote image URL: {path}")
            def _download_remote_image() -> bytes:
                errors = []
                for idx, headers in enumerate(REMOTE_IMAGE_HEADERS, start=1):
                    try:
                        response = requests.get(
                            path,
                            timeout=self.timeout,
                            headers=headers,
                        )
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").lower()
                        if not content_type.startswith("image/"):
                            raise ValueError(
                                f"Remote URL did not return an image content-type: {content_type or 'unknown'}"
                            )
                        return response.content
                    except Exception as exc:
                        errors.append(f"attempt {idx}: {exc}")
                raise RuntimeError(
                    "Failed to download remote image with available request headers. "
                    + " | ".join(errors)
                )

            data = await anyio.to_thread.run_sync(_download_remote_image)
            mime = Image.open(io.BytesIO(data)).get_format_mimetype()
            b64 = base64.b64encode(data).decode()
            return f"data:{mime};base64,{b64}"

        # Local file – read & encode
        logger.debug(f"Encoding local image: {path}")
        data = await anyio.to_thread.run_sync(lambda: open(path, "rb").read())
        mime = Image.open(io.BytesIO(data)).get_format_mimetype()
        b64 = base64.b64encode(data).decode()
        return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
#  FastMCP server
# --------------------------------------------------------------------------- #
mcp = FastMCP("image_analysis")
toolkit = ImageAnalysisToolkit()


@mcp.tool()
async def image_to_text(image_path: str, sys_prompt: Optional[str] = None) -> str:
    """
    Generates a detailed and descriptive caption of the image located at *image_path*.

    Parameters:
    - image_path (str): The file path or URL of the image to analyze.
    - sys_prompt (Optional[str]): An optional system prompt that can guide or influence the image captioning behavior, allowing for customization of the description style, detail level, or focus.

    Returns:
    - str: A detailed natural language description of the content, objects, scene, and relevant features in the image.
    """
    return await toolkit.image_to_text(image_path, sys_prompt)


@mcp.tool()
async def ask_question_about_image(
    image_path: str, question: str, sys_prompt: Optional[str] = None
) -> str:
    """
    Answers a specific question related to the content of the image located at *image_path*.

    Parameters:
    - image_path (str): The file path or URL of the image to analyze.
    - question (str): The question to be answered about the image content.
    - sys_prompt (Optional[str]): An optional system prompt to guide the reasoning or answering style, providing context or desired behavior for the image analysis.

    Returns:
    - str: The answer to the question based on visual analysis and understanding of the image content.
    """
    return await toolkit.ask_question_about_image(image_path, question, sys_prompt)

# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    mcp.run(transport="stdio")
