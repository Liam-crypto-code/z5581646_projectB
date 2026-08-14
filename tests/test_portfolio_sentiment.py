"""Tests for the precomputed descriptive portfolio news context."""

import numpy as np
import pandas as pd
import pytest
from src.portfolio_sentiment import (
    OUTPUT_COLUMNS,
    build_portfolio_sentiment,
    validate_portfolio_sentiment_output,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.DataFrame(
        {
            "fund": ["Fund", "Fund", "Mapping", "Mapping"],
            "fund_id": ["fund", "fund", "mapping", "mapping"],
            "universe": ["Combined"] * 4,
            "method": ["equal_weight"] * 4,
            "effective_date": pd.to_datetime(["2022-01-04"] * 4),
            "asset": ["equity__AAA", "equity__BBB", "equity__AAA", "equity__BBB"],
            "asset_class": ["equity"] * 4,
            "weight": [0.3, 0.5, 0.5, 0.5],
            "sector": [None, None, "Tech", "Energy"],
        }
    )
    sentiment = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-04", "2022-01-04"]),
            "sector": ["Tech", "Energy"],
            "lagged_source_date": pd.to_datetime(["2022-01-03", "2022-01-03"]),
            "lagged_sector_sentiment": [0.4, np.nan],
        }
    )
    return weights, sentiment


def test_portfolio_sentiment_excludes_missing_weight_instead_of_imputing_zero():
    weights, sentiment = _inputs()
    output = build_portfolio_sentiment(weights, sentiment)
    row = output.loc[output["fund_id"].eq("fund")].iloc[0]

    assert tuple(output.columns) == OUTPUT_COLUMNS
    assert not output.duplicated(["fund_id", "date"]).any()
    assert row["portfolio_lagged_sentiment"] == pytest.approx(0.4)
    assert row["total_equity_weight"] == pytest.approx(0.8)
    assert row["observed_equity_weight"] == pytest.approx(0.3)
    assert row["missing_equity_weight"] == pytest.approx(0.5)
    assert row["sentiment_weight_coverage"] == pytest.approx(0.3 / 0.8)


def test_portfolio_sentiment_rejects_nonlagged_signal_dates():
    weights, sentiment = _inputs()
    sentiment.loc[0, "lagged_source_date"] = sentiment.loc[0, "date"]

    with pytest.raises(ValueError, match="earlier trading date"):
        build_portfolio_sentiment(weights, sentiment)


def test_portfolio_sentiment_schema_validator_rejects_invalid_coverage():
    weights, sentiment = _inputs()
    output = build_portfolio_sentiment(weights, sentiment)
    output.loc[output["fund_id"].eq("fund"), "observed_equity_weight"] = 0.9

    with pytest.raises(ValueError, match="sum to total equity weight"):
        validate_portfolio_sentiment_output(output)
