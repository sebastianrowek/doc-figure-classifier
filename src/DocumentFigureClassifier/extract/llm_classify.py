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
from pathlib import Path
import math
from typing import Any, Optional, TypedDict
from logging import Logger
import asyncio

from openai import OpenAI, AsyncOpenAI

from DocumentFigureClassifier.schemas import LLM_CLASS_SCHEMA, LLM_CLASS_SYSTEM_PROMPT

from dotenv import load_dotenv
load_dotenv()

MODEL = "google/gemini-3.7-flash"

from io import BytesIO
from PIL import Image

ImageInput = str | Path | Image.Image

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


class Prediction(TypedDict):
    label: str
    confidence: float


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


def llm_classify_image(
        image: ImageInput,
        logger: Optional[Logger] = None
) -> tuple[Prediction | None, float | None]:

    prediction: Prediction | None = None
    cost: float | None = None

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_schema", "json_schema": LLM_CLASS_SCHEMA},  # type: ignore
            messages=[
                {"role": "system", "content": LLM_CLASS_SYSTEM_PROMPT},
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
            extra_body={
                "reasoning": {"effort": "minimal"},
                "usage": {"include": True},
            },
        )

    except Exception as e:
        if logger:
            logger.error(f"LLM request failed: {e}")
        return prediction, cost

    content = response.choices[0].message.content if response.choices else None
    if not content:
        if logger:
            logger.error("Error: no message in LLM response")
    else:
        try:
            prediction = _parse_prediction(json.loads(content))
        except json.JSONDecodeError:
            if logger:
                logger.error("Error: invalid JSON in LLM response")

        if prediction is None:
            if logger:
                logger.error(f"Error: response did not match the expected schema: {content!r}")
        else:
            if logger:
                logger.info(
                    f"Prediction: {prediction['label']} "
                    f"(confidence: {prediction['confidence']:.2f})"
                )

    usage_cost = getattr(response.usage, "cost", None) if response.usage else None
    if isinstance(usage_cost, (int, float)):
        cost = float(usage_cost)
        if logger:
            logger.info(f"Cost: {cost * 100:.3f} ct for {response.usage.total_tokens} tokens")

    return prediction, cost