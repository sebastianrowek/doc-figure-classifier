"""System-prompt versions for the LLM chart classifier.

Each constant is one full system-prompt version. The judge runner selects the
active one in code (see ``ACTIVE_PROMPT`` / ``EVAL_VERSION`` there); bump the
eval version whenever you switch prompts so every run stays traceable to the
prompt that produced it. The full prompt text is also recorded in each run's
``run.meta.json`` (via the run config), so results are reproducible.
"""

CLASSIFY_V1 = (
    "You classify chart images extracted from business documents. "
    "Return only the label and a calibrated confidence value. "
    "Use 'other' if the image is not a chart or the type is not in the list."
)

# Add new versions below as you iterate, e.g.:
# CLASSIFY_V2 = "..."
