"""Regression tests for the verified Station 1 cleaning foundation."""

import pandas as pd
from src.etl import clean_headlines, clean_price_panel


def _price_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "AAA", "BBB"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-02", "2024-01-01", "2020-01-01"],
            "open": [10, 20, 20, 30, 5],
            "high": [11, 21, 21, 31, 6],
            "low": [9, 19, 19, 29, 4],
            "close": [10, 20, 20, 30, 5],
            "adjClose": [10, 20, 20, 30, 5],
            "volume": [100, 100, 100, 100, 100],
        }
    )


def test_price_cleaning_filters_period_deduplicates_and_keeps_extreme_returns():
    clean, audit, extremes, _ = clean_price_panel(_price_rows(), "test_prices", "equity", 0.30)

    assert len(clean) == 3
    assert clean.duplicated(["ticker", "date"]).sum() == 0
    assert clean["date"].max() == pd.Timestamp("2020-01-02")
    assert extremes.loc[0, "return"] == 1.0
    assert extremes.loc[0, "action"] == "retained for review"
    assert len(clean.loc[(clean["ticker"] == "AAA") & (clean["date"] == "2020-01-02")]) == 1
    assert audit.loc[audit["check"] == "rows outside 2020-2023", "count"].item() == 1


def test_daily_calendar_audit_detects_missing_crypto_date():
    frame = _price_rows().iloc[[0, 1]].copy()
    frame.loc[:, "date"] = ["2020-01-01", "2020-01-03"]
    _, _, _, gaps = clean_price_panel(frame, "test_crypto", "daily", 0.50)

    assert gaps.loc[0, "expected_dates"] == 1461
    assert gaps.loc[0, "missing_dates"] == 1459
    assert gaps.loc[0, "first_missing_date"] == pd.Timestamp("2020-01-02")


def test_headline_cleaning_preserves_distinct_same_day_titles_and_normalises_timezone():
    headlines = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": ["2020-01-04T12:00:00Z"] * 3,
            "sector": ["Tech"] * 3,
            "title": ["First headline", "First headline", "Second headline"],
            "url": ["u1", "u1", "u2"],
            "publisher": [None, None, "Publisher"],
        }
    )
    clean, audit = clean_headlines(headlines)

    assert clean["title"].tolist() == ["First headline", "Second headline"]
    assert clean["date"].dt.tz is None
    duplicate_count = audit.loc[
        audit["check"] == "duplicate ticker-date-title rows", "count"
    ].item()
    assert duplicate_count == 1
