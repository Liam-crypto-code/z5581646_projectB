"""Tests for the precomputed, results-only Streamlit data layer."""

from pathlib import Path

import pandas as pd
import pytest
from src.app_data import (
    APP_ARTIFACTS,
    ASSET_PERFORMANCE_COLUMNS,
    build_asset_performance,
    load_app_data,
    selected_asset_statistics,
)


def _return_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03", "2020-01-06"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Technology", "Technology"],
            "return": [None, 0.10],
        }
    )
    crypto = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03", "2020-01-04", "2020-01-05"]),
            "ticker": ["BTC-USD"] * 3,
            "return": [None, 0.20, -0.10],
        }
    )
    return equity, crypto


def test_asset_performance_preserves_native_calendars_and_growth():
    equity, crypto = _return_panels()
    output = build_asset_performance(equity, crypto)

    assert tuple(output.columns) == ASSET_PERFORMANCE_COLUMNS
    assert len(output[output["asset_type"] == "Equity"]) == 2
    crypto_output = output[output["asset_type"] == "Crypto"]
    assert len(crypto_output) == 3
    assert pd.Timestamp("2020-01-04") in set(crypto_output["date"])
    assert crypto_output["calendar"].eq("native calendar daily").all()
    assert crypto_output["periods_per_year"].eq(365).all()
    assert crypto_output["growth_of_1"].iloc[-1] == pytest.approx(1.08)


def test_asset_performance_has_unique_asset_dates_and_missing_first_returns():
    equity, crypto = _return_panels()
    output = build_asset_performance(equity, crypto)

    assert not output.duplicated(["asset_type", "ticker", "date"]).any()
    first_rows = output.groupby(["asset_type", "ticker"], observed=True).head(1)
    assert first_rows["daily_return"].isna().all()
    assert first_rows["growth_of_1"].eq(1.0).all()


def test_selected_range_statistics_include_first_saved_return():
    equity, crypto = _return_panels()
    output = build_asset_performance(equity, crypto)
    crypto_output = output.loc[output["asset_type"].eq("Crypto") & output["daily_return"].notna()]

    statistics = selected_asset_statistics(crypto_output).iloc[0]
    assert statistics["total_return"] == pytest.approx(0.08)
    assert statistics["maximum_drawdown"] == pytest.approx(-0.10)


def test_results_loader_enforces_every_saved_schema(tmp_path: Path):
    for relative_path, required in APP_ARTIFACTS.values():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{column: None for column in required}]).to_csv(path, index=False)

    bundle = load_app_data(tmp_path)
    assert set(bundle.frames) == set(APP_ARTIFACTS)

    asset_path = tmp_path / APP_ARTIFACTS["asset_performance"][0]
    pd.DataFrame({"date": ["2020-01-01"]}).to_csv(asset_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_app_data(tmp_path)


def test_portfolio_sentiment_is_registered_under_exact_loader_key():
    relative_path, required = APP_ARTIFACTS["portfolio_sentiment"]

    assert relative_path == "data/portfolio_sentiment_context.csv"
    assert {
        "portfolio_lagged_sentiment",
        "observed_equity_weight",
        "missing_equity_weight",
        "sentiment_weight_coverage",
    }.issubset(required)
