"""Build a plain-Markdown summary of an AutoML run (downloadable from the UI)."""
from __future__ import annotations

import pandas as pd


def build_report_md(result: dict) -> str:
    lines = ["# AutoML Report", ""]
    lines += [
        f"- **Task:** {result['task']}",
        f"- **Target:** {result['target']}",
        f"- **Best model:** {result['best_model']}",
        f"- **Optimized for:** {result['scoring']}",
        f"- **Train / test rows:** {result['n_train']} / {result['n_test']}",
        f"- **Features used:** {len(result['features'])}",
        "",
        "## Test metrics",
    ]
    for k, v in result["metrics"].items():
        lines.append(f"- {k}: {v:.4f}")

    lines += ["", "## Model leaderboard (cross-validated)"]
    lb = pd.DataFrame(result["leaderboard"])
    lines += ["```", lb.to_string(index=False), "```"]

    if result["tuned_params"]:
        lines += ["", "## Tuned hyperparameters"]
        for k, v in result["tuned_params"].items():
            lines.append(f"- {k}: {v}")

    lines += ["", "## Preprocessing"]
    for k, v in result["preprocess_plan"].items():
        lines.append(f"- {k}: {', '.join(map(str, v)) if v else 'none'}")

    if result["feature_importance"] is not None:
        lines += ["", "## Top features"]
        lines += ["```", result["feature_importance"].to_string(index=False), "```"]

    return "\n".join(lines)
