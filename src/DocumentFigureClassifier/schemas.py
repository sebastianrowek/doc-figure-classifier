from .taxonomy import TIER1_LABELS

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

LLM_CLASS_SYSTEM_PROMPT = (
    "You classify chart images extracted from business documents. "
    "Return only the label and a calibrated confidence value. "
    "Use 'other' if the image is not a chart or the type is not in the list."
)