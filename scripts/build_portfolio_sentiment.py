"""Build portfolio news context from existing saved Part B outputs."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.portfolio_sentiment import save_portfolio_sentiment_output

ROOT = pathlib.Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    print(f"portfolio sentiment: {save_portfolio_sentiment_output(ROOT / 'results')}")
