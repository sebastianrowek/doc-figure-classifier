"""
Classify a local image with Google Gemini 3.7 Flash via OpenRouter.
Returns strictly {"label": <one of LABELS>, "confidence": <0.0-1.0>}.

pip install openai
export OPENROUTER_API_KEY=sk-or-...
"""

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
import math
from typing import Any, Optional, TypedDict
from logging import Logger
import asyncio

from openai import OpenAI, AsyncOpenAI

from DocumentFigureClassifier.schemas import (
    LLM_CLASS_SCHEMA,
    LlmClassifyConfig,
)

from dotenv import load_dotenv
load_dotenv()

# Canonical settings for a classification call. Override by passing a custom
# LlmClassifyConfig to the classify functions.
DEFAULT_CONFIG = LlmClassifyConfig()

from io import BytesIO
from PIL import Image

ImageInput = str | Path | Image.Image

class Prediction(TypedDict):
    label: str
    confidence: float


def _extract_json(content: str) -> Any:
    """Parse a JSON object from a model reply.

    Some models wrap the JSON in prose ("Here is the JSON requested: {...}") or
    ```json fences instead of honoring strict structured output. Try a direct
    parse first, then fall back to the outermost {...} block. Raises
    json.JSONDecodeError if no JSON object can be recovered.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", content, re.DOTALL)  # first '{' .. last '}'
    if m:
        return json.loads(m.group(0))  # may raise -> caller treats as invalid JSON
    raise json.JSONDecodeError("no JSON object found in response", content or "", 0)


def _process_response(
        response,
        logger=None
) -> tuple[Prediction | None, float | None, str | None, str | None]:
    prediction: Prediction | None = None
    cost: float | None = None
    error: str | None = None

    content = response.choices[0].message.content if response.choices else None
    if not content:
        error = "no message in LLM response"
        if logger:
            logger.error(f"Error: {error}")
    else:
        try:
            prediction = _parse_prediction(_extract_json(content))
        except json.JSONDecodeError:
            error = "invalid JSON in LLM response"
            if logger:
                logger.error(f"Error: {error}")

        if prediction is None and error is None:
            error = f"response did not match the expected schema: {content!r}"
            if logger:
                logger.error(f"Error: {error}")
        elif prediction is not None:
            if logger:
                logger.info(
                    f"Prediction: {prediction['label']} "
                    f"(confidence: {prediction['confidence']:.2f})"
                )

    usage_cost = getattr(response.usage, "cost", None) if response.usage else None
    total_tokens = getattr(response.usage, "total_tokens", None) if response.usage else None
    if isinstance(usage_cost, (int, float)):
        cost = float(usage_cost)
        if logger:
            logger.info(f"Cost: {cost * 100:.3f} $ct")
    if isinstance(total_tokens, int):
        if logger:
            logger.info(f"Total tokens: {total_tokens}")

    return prediction, cost, error, content

def _request_kwargs(image: ImageInput, config: LlmClassifyConfig = DEFAULT_CONFIG) -> dict:
    kwargs = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format":{"type": "json_schema", "json_schema": LLM_CLASS_SCHEMA},  # type: ignore
        "messages":[
            {"role": "system", "content": config.prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this chart."},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image)},
                    },
                ],
            },
        ],
        "extra_body":{
            "reasoning": {"effort": config.reasoning_effort},
            "usage": {"include": True},
            # only route to endpoints that actually honor response_format
            "provider": {"require_parameters": config.require_parameters},
        },
    }
    return kwargs

def _parse_prediction(raw: Any) -> Prediction | None:
    """Return a well-typed Prediction, or None if the payload doesn't fit."""
    if not isinstance(raw, dict):
        return None

    label = raw.get("label")
    if not isinstance(label, str) or not label.strip():
        return None

    confidence = raw.get("confidence")
    # bool is a subclass of int — exclude it explicitly.
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    confidence = float(confidence)
    if math.isnan(confidence) or not 0.0 <= confidence <= 1.0:
        return None

    return {"label": label, "confidence": confidence}

def image_to_data_url(image: ImageInput) -> str:
    if isinstance(image, Image.Image):
        fmt = (image.format or "PNG").upper()
        if fmt not in {"PNG", "JPEG", "WEBP"}:
            fmt = "PNG"
        img = image
        if fmt == "JPEG" and img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
        elif img.mode in {"P", "CMYK"}:
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format=fmt)
        data = buf.getvalue()
        mime = Image.MIME.get(fmt, f"image/{fmt.lower()}")
    else:
        path = Path(image)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

async def async_llm_classify_image(
        client: AsyncOpenAI,
        image: ImageInput,
        config: LlmClassifyConfig = DEFAULT_CONFIG,
        logger: Optional[Logger] = None,
) -> tuple[Prediction | None, float | None, str | None, str | None]:

    try:
        response = await client.chat.completions.create(
            **_request_kwargs(image, config)
        )

    except Exception as e:
        error = f"LLM request failed: {e}"
        if logger:
            logger.error(error)
        return None, None, error, None

    return _process_response(response, logger)


def llm_classify_image(
        client: OpenAI,
        image: ImageInput,
        config: LlmClassifyConfig = DEFAULT_CONFIG,
        logger: Optional[Logger] = None,
) -> tuple[Prediction | None, float | None, str | None, str | None]:

    try:
        response = client.chat.completions.create(
            **_request_kwargs(image, config)
        )

    except Exception as e:
        error = f"LLM request failed: {e}"
        if logger:
            logger.error(error)
        return None, None, error, None

    return _process_response(response, logger)