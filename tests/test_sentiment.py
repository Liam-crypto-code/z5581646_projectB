"""Focused tests for VADER scoring, aggregation, coverage, and signal timing."""

import pandas as pd
import pytest
from src.features import align_headline_dates
from src.sentiment import (
    company_sentiment_index,
    score_headlines,
    sector_sentiment_index,
    ticker_day_sentiment,
)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Tech", "Utilities"],
        }
    )


def test_vader_scores_original_text_without_modifying_it():
    titles = ["Company reports EXCELLENT growth!!!", "Company is not good"]
    aligned = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2022-01-03", "2022-01-03"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "title": titles,
        }
    )

    scores = score_headlines(aligned)

    assert scores["title"].tolist() == titles
    assert scores.loc[0, "vader_compound"] > 0
    assert scores.loc[1, "vader_compound"] < 0


def test_weekend_and_monday_headlines_are_unavailable_until_tuesday():
    calendar = pd.to_datetime(["2022-01-07", "2022-01-10", "2022-01-11"])
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-08", "2022-01-10"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "title": ["Strong weekend gain", "Excellent Monday result"],
        }
    )
    aligned = align_headline_dates(headlines, calendar)
    scores = ticker_day_sentiment(score_headlines(aligned))
    index = sector_sentiment_index(scores, calendar, _universe())
    tech = index[index["sector"] == "Tech"].set_index("date")

    assert aligned["trading_date"].tolist() == [pd.Timestamp("2022-01-10")] * 2
    assert pd.isna(tech.loc[pd.Timestamp("2022-01-10"), "lagged_sector_sentiment"])
    assert tech.loc[pd.Timestamp("2022-01-11"), "lagged_sector_sentiment"] == pytest.approx(
        tech.loc[pd.Timestamp("2022-01-10"), "sector_sentiment"]
    )


def test_sector_signal_is_lagged_by_exactly_one_equity_trading_day():
    calendar = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
    ticker_scores = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2022-01-03", "2022-01-04"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "ticker_sentiment": [0.25, -0.40],
            "headline_count": [1, 1],
        }
    )
    tech = sector_sentiment_index(ticker_scores, calendar, _universe())
    tech = tech[tech["sector"] == "Tech"].set_index("date")

    assert tech.loc[pd.Timestamp("2022-01-04"), "lagged_sector_sentiment"] == pytest.approx(0.25)
    assert tech.loc[pd.Timestamp("2022-01-05"), "lagged_sector_sentiment"] == pytest.approx(-0.40)
    assert tech.loc[pd.Timestamp("2022-01-04"), "lagged_source_date"] == pd.Timestamp("2022-01-03")


def test_sector_index_equal_weights_tickers_not_headline_counts():
    date = pd.Timestamp("2022-01-03")
    headline_scores = pd.DataFrame(
        {
            "trading_date": [date, date, date, date],
            "ticker": ["AAA", "AAA", "AAA", "BBB"],
            "sector": ["Tech"] * 4,
            "vader_compound": [1.0, 1.0, 1.0, -1.0],
        }
    )
    ticker_scores = ticker_day_sentiment(headline_scores)
    index = sector_sentiment_index(ticker_scores, pd.DatetimeIndex([date]), _universe())
    tech = index[index["sector"] == "Tech"].iloc[0]

    assert tech["sector_sentiment"] == pytest.approx(0.0)
    assert tech["ticker_coverage"] == 2
    assert tech["headline_count"] == 4


def test_no_news_days_remain_missing_instead_of_becoming_neutral():
    calendar = pd.to_datetime(["2022-01-03", "2022-01-04"])
    ticker_scores = pd.DataFrame(
        {
            "trading_date": [pd.Timestamp("2022-01-03")],
            "ticker": ["AAA"],
            "sector": ["Tech"],
            "ticker_sentiment": [0.0],
            "headline_count": [1],
        }
    )
    index = sector_sentiment_index(ticker_scores, calendar, _universe())
    tech = index[index["sector"] == "Tech"].set_index("date")
    utilities = index[index["sector"] == "Utilities"].set_index("date")

    assert tech.loc[pd.Timestamp("2022-01-03"), "sector_sentiment"] == pytest.approx(0.0)
    assert tech.loc[pd.Timestamp("2022-01-03"), "ticker_coverage"] == 1
    assert pd.isna(utilities.loc[pd.Timestamp("2022-01-03"), "sector_sentiment"])
    assert utilities.loc[pd.Timestamp("2022-01-03"), "ticker_coverage"] == 0
    assert pd.isna(tech.loc[pd.Timestamp("2022-01-04"), "sector_sentiment"])
    assert tech.loc[pd.Timestamp("2022-01-04"), "ticker_coverage"] == 0


def test_company_grid_distinguishes_observed_zero_from_no_news():
    calendar = pd.to_datetime(["2022-01-03", "2022-01-04"])
    ticker_scores = pd.DataFrame(
        {
            "trading_date": [pd.Timestamp("2022-01-03")],
            "ticker": ["AAA"],
            "sector": ["Tech"],
            "ticker_sentiment": [0.0],
            "headline_count": [2],
            "neutral_headline_count": [2],
        }
    )
    index = company_sentiment_index(ticker_scores, calendar, _universe())
    indexed = index.set_index(["date", "ticker"])
    observed_zero = indexed.loc[(pd.Timestamp("2022-01-03"), "AAA")]
    no_news = indexed.loc[(pd.Timestamp("2022-01-03"), "BBB")]

    assert len(index) == len(calendar) * len(_universe())
    assert observed_zero["vader_company_sentiment"] == pytest.approx(0.0)
    assert observed_zero["news_observed"]
    assert observed_zero["news_coverage"] == 1.0
    assert pd.isna(no_news["vader_company_sentiment"])
    assert not no_news["news_observed"]
    assert no_news["news_coverage"] == 0.0


def test_company_signal_uses_exactly_the_previous_equity_trading_day():
    calendar = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
    ticker_scores = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2022-01-03", "2022-01-05"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "ticker_sentiment": [0.0, 0.6],
            "headline_count": [1, 1],
            "neutral_headline_count": [1, 0],
        }
    )
    company = company_sentiment_index(ticker_scores, calendar, _universe())
    aaa = company[company["ticker"] == "AAA"].set_index("date")

    assert aaa.loc[pd.Timestamp("2022-01-04"), "lagged_source_date"] == pd.Timestamp(
        "2022-01-03"
    )
    assert aaa.loc[
        pd.Timestamp("2022-01-04"), "lagged_vader_company_sentiment"
    ] == pytest.approx(0.0)
    assert aaa.loc[pd.Timestamp("2022-01-04"), "signal_available"]
    assert pd.isna(
        aaa.loc[pd.Timestamp("2022-01-05"), "lagged_vader_company_sentiment"]
    )
    assert not aaa.loc[pd.Timestamp("2022-01-05"), "signal_available"]


def test_weekend_company_news_aligned_to_monday_is_first_usable_tuesday():
    calendar = pd.to_datetime(["2022-01-07", "2022-01-10", "2022-01-11"])
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-08"]),
            "ticker": ["AAA"],
            "sector": ["Tech"],
            "title": ["Excellent weekend gain"],
        }
    )
    aligned = align_headline_dates(headlines, calendar)
    ticker_scores = ticker_day_sentiment(score_headlines(aligned))
    company = company_sentiment_index(ticker_scores, calendar, _universe())
    aaa = company[company["ticker"] == "AAA"].set_index("date")

    assert pd.isna(
        aaa.loc[pd.Timestamp("2022-01-10"), "lagged_vader_company_sentiment"]
    )
    assert aaa.loc[
        pd.Timestamp("2022-01-11"), "lagged_vader_company_sentiment"
    ] == pytest.approx(aaa.loc[pd.Timestamp("2022-01-10"), "vader_company_sentiment"])
