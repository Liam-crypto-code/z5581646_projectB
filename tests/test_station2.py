"""Regression tests for native-calendar returns and headline alignment."""

import pandas as pd
import pytest
from src.features import (
    align_headline_dates,
    align_return_calendars,
    assemble_headline_panel,
    daily_returns,
    descriptive_return_statistics,
)


def test_returns_are_calculated_within_ticker_from_adjusted_close():
    prices = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"] * 2),
            "adjClose": [100.0, 110.0, 50.0, 40.0],
            "close": [100.0, 100.0, 50.0, 50.0],
        }
    )
    returns = daily_returns(prices)

    assert pd.isna(returns.loc[returns["ticker"] == "AAA", "return"].iloc[0])
    assert returns.loc[returns["ticker"] == "AAA", "return"].iloc[1] == pytest.approx(0.10)
    assert returns.loc[returns["ticker"] == "BBB", "return"].iloc[1] == pytest.approx(-0.20)


def test_crypto_returns_are_computed_before_equity_calendar_alignment():
    equity = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "adjClose": [100.0, 101.0],
        }
    )
    crypto = pd.DataFrame(
        {
            "ticker": ["BTC-USD"] * 4,
            "date": pd.to_datetime(["2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07"]),
            "adjClose": [100.0, 200.0, 220.0, 242.0],
        }
    )
    combined = align_return_calendars(daily_returns(equity), daily_returns(crypto))

    monday_return = combined.loc[
        combined["date"] == pd.Timestamp("2020-01-06"), "crypto__BTC-USD"
    ].item()
    assert monday_return == pytest.approx(0.10)


def test_headlines_map_to_same_or_next_trading_day_and_assemble():
    headlines = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "date": pd.to_datetime(["2020-01-04", "2020-01-05", "2020-01-06", "2020-01-08"]),
            "sector": ["Tech"] * 4,
            "title": ["Saturday gain", "Sunday risk", "Monday update", "Beyond calendar"],
        }
    )
    calendar = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
    aligned = align_headline_dates(headlines, calendar)
    panel = assemble_headline_panel(headlines, calendar)

    assert aligned["trading_date"].tolist()[:3] == [pd.Timestamp("2020-01-06")] * 3
    assert pd.isna(aligned["trading_date"].iloc[3])
    assert panel.loc[0, "headline_count"] == 3
    assert "Saturday gain" in panel.loc[0, "headline_text"]
    assert "Monday update" in panel.loc[0, "headline_text"]


def test_descriptive_statistics_use_percentage_units():
    equity = pd.DataFrame({"return": [None, 0.01, -0.01]})
    crypto = pd.DataFrame({"return": [None, 0.02, -0.02]})
    stats = descriptive_return_statistics(equity, crypto)

    assert stats.loc[stats["asset_class"] == "Equities", "observations"].item() == 2
    assert stats.loc[
        stats["asset_class"] == "Equities", "maximum_daily_return_pct"
    ].item() == 1
