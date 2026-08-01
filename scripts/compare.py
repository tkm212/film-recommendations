"""Serve the cross-validated model comparison dashboard.

    uv run python scripts/compare.py
    uv run python scripts/compare.py --port 8051

Requires outputs/model_comparison.csv from a prior training run:

    uv run python scripts/main.py --train
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from film_recommendations.comparison_dashboard import create_comparison_app

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_FILE = REPO_ROOT / "outputs" / "model_comparison.csv"
SELECTED_MODEL_FILE = REPO_ROOT / "outputs" / "selected_model.txt"


def load_comparison() -> pd.DataFrame:
    """Load the cached model comparison table.

    Raises:
        FileNotFoundError: If no comparison has been saved yet.
    """
    if not COMPARISON_FILE.exists():
        msg = f"Model comparison not found: {COMPARISON_FILE}. Run `uv run python scripts/main.py --train` first."
        raise FileNotFoundError(msg)
    return pd.read_csv(COMPARISON_FILE)


def load_selected_model() -> str | None:
    """Load the last fitted model name, if saved."""
    if not SELECTED_MODEL_FILE.exists():
        return None
    return SELECTED_MODEL_FILE.read_text(encoding="utf-8").strip() or None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Serve the model comparison dashboard.")
    parser.add_argument("--port", type=int, default=8051, help="Dashboard port (default: 8051).")
    return parser.parse_args()


def main() -> None:
    """Load benchmark results and serve the comparison dashboard."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    comparison = load_comparison()
    selected = load_selected_model()
    logger.info("Loaded %d model benchmarks from %s", len(comparison), COMPARISON_FILE)

    app = create_comparison_app(comparison, selected_model=selected)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
