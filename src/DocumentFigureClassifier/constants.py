from pathlib import Path

# repo_root/src/DocumentFigureClassifier/constants.py -> repo_root/data/llm_judge
DEFAULT_LLM_JUDGE_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "llm_judge"