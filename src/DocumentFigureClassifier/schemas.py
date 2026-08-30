from .taxonomy import TIER1_LABELS
from .prompt import CLASSIFY_V1
from typing import TypedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class LlmClassifyConfig:
    """Light, serializable settings for one classification run.

    Recorded as run metadata by the judge runner so a result file can always be
    traced back to the exact model / decoding settings (and prompt) that
    produced it.
    """
    model: str = "google/gemini-3.7-flash"
    # System prompt sent with every classification call. Defaults to the current
    # baseline; the judge runner overrides it with the active prompt version.
    prompt: str = CLASSIFY_V1
    temperature: float = 0.0
    max_tokens: int = 1024
    reasoning_effort: str = "medium"  # OpenRouter reasoning effort: minimal|low|medium|high
    # OpenRouter routes per-endpoint; strict schema compliance is only guidance
    # (not guaranteed on every endpoint). require_parameters makes OpenRouter skip
    # endpoints that would ignore response_format, cutting prose-wrapped replies.
    require_parameters: bool = True



# Allowlist of OpenRouter model slugs the judge eval may use. The runner's
# ``--model`` flag validates against this, so it is the single source of truth
# for which models are candidates. Add slugs as you decide to evaluate them.
CANDIDATE_MODELS: tuple[str, ...] = (
    "google/gemini-3.7-flash",
    "z-ai/glm-5.3-flash"
)


class LlmJudgeCallResult(TypedDict):
    file: str  # path relative to data/llm_judge, forward slashes, e.g. "bar/bar__web04__quanthub-com.png"
    true: str
    pred: str | None
    confidence: float | None
    call_idx: int
    cost: float | None
    error: str | None

LLM_CLASS_SCHEMA = {
    "name": "chart_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": TIER1_LABELS,
                "description": "The chart type shown in the image.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Probability that the label is correct.",
            },
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    },
}