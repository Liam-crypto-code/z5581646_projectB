"""Build the precomputed native-calendar asset artifact used by Streamlit."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.app_data import save_asset_performance_outputs
from src.etl import run_station1
from src.features import run_station2
from src.portfolio_sentiment import save_portfolio_sentiment_output

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    """Reuse the verified Part A pipeline and write app-only derived outputs."""
    station1 = run_station1()
    station2 = run_station2(station1.equities, station1.crypto, station1.headlines)
    data_path, methodology_path = save_asset_performance_outputs(
        station2.equity_returns,
        station2.crypto_returns,
        ROOT / "results",
    )
    print(f"asset performance: {data_path}")
    print(f"methodology: {methodology_path}")
    print(f"portfolio sentiment: {save_portfolio_sentiment_output(ROOT / 'results')}")


if __name__ == "__main__":
    main()
