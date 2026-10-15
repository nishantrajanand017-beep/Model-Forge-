"""AutoML: a small, local, no-LLM automated machine-learning pipeline for tabular data."""

from .core import run_automl, detect_task

__all__ = ["run_automl", "detect_task"]
