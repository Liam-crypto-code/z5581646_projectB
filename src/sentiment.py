"""Standalone VADER sentiment index with explicit coverage and trading-day lag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer(Protocol):
    """Minimal analyzer interface used by the scoring function."""

    def polarity_scores(self, text: str) -> dict[str, float]: ...


@dataclass
class SentimentResults:
    """Headline, ticker-day, and complete sector-calendar sentiment outputs."""

    headline_scores: pd.DataFrame
    ticker_day_scores: pd.DataFrame
    sector_index: pd.DataFrame


def score_headlines(
    aligned_headlines: pd.DataFrame,
    analyzer: SentimentAnalyzer | None = None,
) -> pd.DataFrame:
    """Score each original aligned headline with VADER without changing its text."""
    required = {"trading_date", "ticker", "sector", "title"}
    missing = required.difference(aligned_headlines.columns)
    if missing:
        raise ValueError(f"aligned_headlines is missing required columns: {sorted(missing)}")

    usable = aligned_headlines.dropna(subset=["trading_date", "ticker", "sector", "title"]).copy()
    usable["trading_date"] = pd.to_datetime(usable["trading_date"])
    scorer = analyzer or SentimentIntensityAnalyzer()
    vader = usable["title"].map(lambda title: scorer.polarity_scores(str(title)))
    score_frame = pd.DataFrame(vader.tolist(), index=usable.index).rename(
        columns={
            "neg": "vader_negative",
            "neu": "vader_neutral",
            "pos": "vader_positive",
            "compound": "vader_compound",
        }
    )
    expected = {"vader_negative", "vader_neutral", "vader_positive", "vader_compound"}
    if not expected.issubset(score_frame.columns):
        raise ValueError("sentiment analyzer did not return the required VADER score fields")
    return pd.concat([usable, score_frame[list(sorted(expected))]], axis=1).reset_index(drop=True)


def ticker_day_sentiment(headline_scores: pd.DataFrame) -> pd.DataFrame:
    """Average headline compound scores within each ticker and aligned trading day."""
    required = {"trading_date", "ticker", "sector", "vader_compound"}
    missing = required.difference(headline_scores.columns)
    if missing:
        raise ValueError(f"headline_scores is missing required columns: {sorted(missing)}")

    return (
        headline_scores.groupby(["trading_date", "ticker", "sector"], observed=True, as_index=False)
        .agg(
            ticker_sentiment=("vader_compound", "mean"),
            headline_count=("vader_compound", "size"),
            neutral_headline_count=("vader_compound", lambda values: int(values.eq(0).sum())),
        )
        .sort_values(["trading_date", "sector", "ticker"])
        .reset_index(drop=True)
    )


def company_sentiment_index(
    ticker_day_scores: pd.DataFrame,
    trading_calendar: pd.Series | pd.DatetimeIndex,
    sector_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Build a complete ticker-calendar VADER index with an exact one-day lag.

    A ticker-day with observed headlines can have a genuine sentiment score of
    zero. A ticker-day without headlines has zero coverage and missing
    sentiment. Lagging is by the preceding equity-calendar row, never by the
    preceding available-news observation.
    """
    score_columns = {
        "trading_date",
        "ticker",
        "sector",
        "ticker_sentiment",
        "headline_count",
        "neutral_headline_count",
    }
    missing = score_columns.difference(ticker_day_scores.columns)
    if missing:
        raise ValueError(f"ticker_day_scores is missing required columns: {sorted(missing)}")
    universe_columns = {"ticker", "sector"}
    missing_universe = universe_columns.difference(sector_universe.columns)
    if missing_universe:
        raise ValueError(f"sector_universe is missing required columns: {sorted(missing_universe)}")

    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar).unique()).sort_values()
    if calendar.empty:
        raise ValueError("trading_calendar must contain at least one date")
    universe = sector_universe[["ticker", "sector"]].drop_duplicates().copy()
    if universe["ticker"].duplicated().any():
        raise ValueError("each ticker must map to exactly one sector")

    observed = ticker_day_scores[
        [
            "trading_date",
            "ticker",
            "sector",
            "ticker_sentiment",
            "headline_count",
            "neutral_headline_count",
        ]
    ].rename(
        columns={
            "trading_date": "date",
            "ticker_sentiment": "vader_company_sentiment",
        }
    )
    grid = pd.MultiIndex.from_product(
        [calendar, universe["ticker"]], names=["date", "ticker"]
    ).to_frame(index=False)
    index = grid.merge(universe, on="ticker", how="left", validate="many_to_one")
    index = index.merge(
        observed,
        on=["date", "ticker", "sector"],
        how="left",
        validate="one_to_one",
    )
    index["headline_count"] = index["headline_count"].fillna(0).astype(int)
    index["neutral_headline_count"] = index["neutral_headline_count"].fillna(0).astype(int)
    index["news_observed"] = index["headline_count"].gt(0)
    index["news_coverage"] = index["news_observed"].astype(float)

    index = index.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = index.groupby("ticker", observed=True, sort=False)
    index["lagged_source_date"] = grouped["date"].shift(1)
    index["lagged_vader_company_sentiment"] = grouped["vader_company_sentiment"].shift(1)
    index["lagged_headline_count"] = grouped["headline_count"].shift(1).astype("Int64")
    index["lagged_news_observed"] = grouped["news_observed"].shift(1).astype("boolean")
    index["signal_available"] = index["lagged_vader_company_sentiment"].notna()
    return index.sort_values(["date", "sector", "ticker"]).reset_index(drop=True)


def sector_sentiment_index(
    ticker_day_scores: pd.DataFrame,
    trading_calendar: pd.Series | pd.DatetimeIndex,
    sector_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Build a complete daily sector index and lag it by one equity trading day.

    The contemporaneous sector score is an equal-weight mean of observed
    ticker-day scores. Days without headlines remain missing and carry zero
    coverage; they are not converted to neutral sentiment.
    """
    score_columns = {
        "trading_date",
        "ticker",
        "sector",
        "ticker_sentiment",
        "headline_count",
    }
    missing = score_columns.difference(ticker_day_scores.columns)
    if missing:
        raise ValueError(f"ticker_day_scores is missing required columns: {sorted(missing)}")
    universe_columns = {"ticker", "sector"}
    missing_universe = universe_columns.difference(sector_universe.columns)
    if missing_universe:
        raise ValueError(f"sector_universe is missing required columns: {sorted(missing_universe)}")

    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar).unique()).sort_values()
    if calendar.empty:
        raise ValueError("trading_calendar must contain at least one date")

    universe = sector_universe[["ticker", "sector"]].drop_duplicates().copy()
    if universe["ticker"].duplicated().any():
        raise ValueError("each ticker must map to exactly one sector")
    universe_counts = universe.groupby("sector", observed=True)["ticker"].nunique()
    sectors = pd.Index(universe_counts.index).sort_values()

    observed = (
        ticker_day_scores.groupby(["trading_date", "sector"], observed=True, as_index=False)
        .agg(
            sector_sentiment=("ticker_sentiment", "mean"),
            ticker_coverage=("ticker", "nunique"),
            headline_count=("headline_count", "sum"),
        )
        .rename(columns={"trading_date": "date"})
    )
    grid = pd.MultiIndex.from_product([calendar, sectors], names=["date", "sector"]).to_frame(
        index=False
    )
    index = grid.merge(observed, on=["date", "sector"], how="left", validate="one_to_one")
    index["universe_tickers"] = index["sector"].map(universe_counts).astype(int)
    index["ticker_coverage"] = index["ticker_coverage"].fillna(0).astype(int)
    index["headline_count"] = index["headline_count"].fillna(0).astype(int)
    index["coverage_ratio"] = index["ticker_coverage"] / index["universe_tickers"]

    index = index.sort_values(["sector", "date"]).reset_index(drop=True)
    grouped = index.groupby("sector", observed=True, sort=False)
    index["lagged_source_date"] = grouped["date"].shift(1)
    index["lagged_sector_sentiment"] = grouped["sector_sentiment"].shift(1)
    index["lagged_ticker_coverage"] = grouped["ticker_coverage"].shift(1).astype("Int64")
    index["lagged_headline_count"] = grouped["headline_count"].shift(1).astype("Int64")
    index["lagged_coverage_ratio"] = grouped["coverage_ratio"].shift(1)
    index["signal_available"] = index["lagged_sector_sentiment"].notna()
    return index.sort_values(["date", "sector"]).reset_index(drop=True)


def run_sentiment_index(
    aligned_headlines: pd.DataFrame,
    trading_calendar: pd.Series | pd.DatetimeIndex,
    sector_universe: pd.DataFrame,
    analyzer: SentimentAnalyzer | None = None,
) -> SentimentResults:
    """Run headline scoring, ticker-day aggregation, and sector indexing."""
    headline_scores = score_headlines(aligned_headlines, analyzer=analyzer)
    ticker_scores = ticker_day_sentiment(headline_scores)
    sector_index = sector_sentiment_index(ticker_scores, trading_calendar, sector_universe)
    return SentimentResults(
        headline_scores=headline_scores,
        ticker_day_scores=ticker_scores,
        sector_index=sector_index,
    )


def sentiment_summary_by_sector(sector_index: pd.DataFrame) -> pd.DataFrame:
    """Summarise observed sentiment and coverage for report audit tables."""
    rows = []
    for sector, frame in sector_index.groupby("sector", observed=True):
        observed = frame["sector_sentiment"].notna()
        rows.append(
            {
                "sector": sector,
                "trading_days": len(frame),
                "observed_sentiment_days": int(observed.sum()),
                "no_news_days": int((~observed).sum()),
                "observed_day_ratio": float(observed.mean()),
                "mean_ticker_coverage": float(frame["ticker_coverage"].mean()),
                "mean_coverage_ratio": float(frame["coverage_ratio"].mean()),
                "mean_sentiment_observed_days": float(
                    frame.loc[observed, "sector_sentiment"].mean()
                ),
                "sentiment_std_observed_days": float(frame.loc[observed, "sector_sentiment"].std()),
                "total_headlines": int(frame["headline_count"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("sector").reset_index(drop=True)


def company_sentiment_summary_by_ticker(company_index: pd.DataFrame) -> pd.DataFrame:
    """Summarise news availability and observed VADER tone for each company."""
    rows = []
    for (ticker, sector), frame in company_index.groupby(
        ["ticker", "sector"], observed=True, sort=True
    ):
        observed = frame["news_observed"]
        observed_scores = frame.loc[observed, "vader_company_sentiment"]
        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "start_date": frame["date"].min(),
                "end_date": frame["date"].max(),
                "trading_days": len(frame),
                "observed_news_days": int(observed.sum()),
                "no_news_days": int((~observed).sum()),
                "observed_news_ratio": float(observed.mean()),
                "genuine_zero_sentiment_days": int(observed_scores.eq(0).sum()),
                "total_headlines": int(frame["headline_count"].sum()),
                "mean_headlines_observed_days": float(
                    frame.loc[observed, "headline_count"].mean()
                ),
                "mean_sentiment_observed_days": float(observed_scores.mean()),
                "sentiment_std_observed_days": float(observed_scores.std()),
            }
        )
    return pd.DataFrame(rows).sort_values(["sector", "ticker"]).reset_index(drop=True)


def company_sentiment_coverage_by_sector(company_index: pd.DataFrame) -> pd.DataFrame:
    """Aggregate company-day availability without replacing missing tone by zero."""
    rows = []
    for sector, frame in company_index.groupby("sector", observed=True, sort=True):
        observed = frame["news_observed"]
        observed_scores = frame.loc[observed, "vader_company_sentiment"]
        rows.append(
            {
                "sector": sector,
                "tickers": frame["ticker"].nunique(),
                "ticker_calendar_rows": len(frame),
                "observed_company_days": int(observed.sum()),
                "no_news_company_days": int((~observed).sum()),
                "observed_company_day_ratio": float(observed.mean()),
                "genuine_zero_sentiment_days": int(observed_scores.eq(0).sum()),
                "total_headlines": int(frame["headline_count"].sum()),
                "mean_sentiment_observed_days": float(observed_scores.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("sector").reset_index(drop=True)


def vader_distribution_summary(headline_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarise VADER compound classifications using its conventional thresholds."""
    compound = headline_scores["vader_compound"]
    labels = np.select(
        [compound <= -0.05, compound >= 0.05],
        ["negative", "positive"],
        default="neutral",
    )
    summary = (
        pd.Series(labels).value_counts().reindex(["negative", "neutral", "positive"], fill_value=0)
    )
    return pd.DataFrame(
        {
            "classification": summary.index,
            "headline_count": summary.to_numpy(),
            "headline_ratio": summary.to_numpy() / len(headline_scores),
            "threshold": ["compound <= -0.05", "-0.05 < compound < 0.05", "compound >= 0.05"],
        }
    )
