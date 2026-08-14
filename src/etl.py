"""Verified Station 1 ETL, cleaning, and data-integrity checks from Part A."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from src import data_access

SAMPLE_START = pd.Timestamp("2020-01-01")
SAMPLE_END = pd.Timestamp("2023-12-31")
PRICE_COLUMNS = ("ticker", "date", "open", "high", "low", "close", "adjClose", "volume")
NEWS_COLUMNS = ("ticker", "date", "sector", "title", "url", "publisher")
EQUITY_EXTREME_THRESHOLD = 0.30
CRYPTO_EXTREME_THRESHOLD = 0.50


@dataclass
class Station1Results:
    """Clean data and reproducible Station 1 audit outputs."""

    equities: pd.DataFrame
    crypto: pd.DataFrame
    headlines: pd.DataFrame
    inventory: pd.DataFrame
    integrity: pd.DataFrame
    extreme_returns: pd.DataFrame
    missing_dates: pd.DataFrame


def _normalise_dates(values: pd.Series) -> pd.Series:
    """Convert tz-aware or tz-naive values to tz-naive midnight timestamps."""
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def _integrity_row(dataset: str, check: str, count: int, action: str) -> dict:
    return {"dataset": dataset, "check": check, "count": int(count), "action": action}


def clean_price_panel(
    frame: pd.DataFrame,
    dataset: str,
    calendar: str,
    extreme_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean one price panel and return data, audit, extremes, and date gaps.

    Extreme returns are a review rule only. They are calculated temporarily from
    adjusted close on each asset's native calendar and are never deleted here.
    """
    missing_columns = set(PRICE_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{dataset} is missing required columns: {sorted(missing_columns)}")
    if calendar not in {"equity", "daily"}:
        raise ValueError("calendar must be 'equity' or 'daily'")

    clean = frame.copy()
    clean["date"] = _normalise_dates(clean["date"])
    for column in ("open", "high", "low", "close", "adjClose", "volume"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    invalid_dates = clean["date"].isna()
    out_of_period = clean["date"].notna() & ~clean["date"].between(SAMPLE_START, SAMPLE_END)
    missing_tickers = clean["ticker"].isna() | clean["ticker"].astype("string").str.strip().eq("")
    clean = clean.loc[~(invalid_dates | out_of_period | missing_tickers)].copy()
    clean["ticker"] = clean["ticker"].astype("string").str.strip()

    duplicate_keys = clean.duplicated(["ticker", "date"], keep="first")
    duplicate_count = int(duplicate_keys.sum())
    clean = clean.loc[~duplicate_keys].sort_values(["ticker", "date"]).reset_index(drop=True)

    missing_price_cells = int(clean[list(PRICE_COLUMNS[2:])].isna().sum().sum())
    nonpositive_prices = int(
        (clean[["open", "high", "low", "close", "adjClose"]] <= 0).any(axis=1).sum()
    )
    negative_volume = int((clean["volume"] < 0).sum())
    bad_high = clean["high"] < clean[["open", "low", "close"]].max(axis=1)
    bad_low = clean["low"] > clean[["open", "high", "close"]].min(axis=1)
    inconsistent_ohlc = int((bad_high | bad_low).sum())

    if calendar == "equity":
        expected_dates = pd.DatetimeIndex(clean["date"].drop_duplicates().sort_values())
        frequency = "equity trading days"
    else:
        expected_dates = pd.date_range(SAMPLE_START, SAMPLE_END, freq="D")
        frequency = "calendar daily"

    observed_by_ticker = clean.groupby("ticker", observed=True)["date"].agg(set)
    gap_rows = []
    for ticker, observed in observed_by_ticker.items():
        missing = expected_dates.difference(pd.DatetimeIndex(observed))
        gap_rows.append(
            {
                "dataset": dataset,
                "ticker": ticker,
                "calendar": frequency,
                "expected_dates": len(expected_dates),
                "observed_dates": len(observed),
                "missing_dates": len(missing),
                "first_missing_date": missing.min() if len(missing) else pd.NaT,
                "last_missing_date": missing.max() if len(missing) else pd.NaT,
            }
        )
    gaps = pd.DataFrame(gap_rows)

    review = clean[["ticker", "date", "adjClose"]].copy()
    review["return"] = review.groupby("ticker", observed=True)["adjClose"].pct_change(
        fill_method=None
    )
    review = review.loc[review["return"].abs() > extreme_threshold].copy()
    review.insert(0, "dataset", dataset)
    review["abs_return"] = review["return"].abs()
    review["threshold"] = extreme_threshold
    review["action"] = "retained for review"
    review = (
        review.drop(columns="adjClose")
        .sort_values("abs_return", ascending=False)
        .reset_index(drop=True)
    )

    audit = pd.DataFrame(
        [
            _integrity_row(
                dataset, "invalid date rows", invalid_dates.sum(), "removed: unusable date key"
            ),
            _integrity_row(
                dataset,
                "rows outside 2020-2023",
                out_of_period.sum(),
                "removed: outside project sample",
            ),
            _integrity_row(
                dataset,
                "missing ticker rows",
                missing_tickers.sum(),
                "removed: unusable ticker key",
            ),
            _integrity_row(
                dataset,
                "duplicate ticker-date rows",
                duplicate_count,
                "removed duplicate; kept first occurrence",
            ),
            _integrity_row(
                dataset,
                "missing numeric price/volume cells",
                missing_price_cells,
                "retained and reported for review",
            ),
            _integrity_row(
                dataset,
                "rows with non-positive prices",
                nonpositive_prices,
                "retained and reported for review",
            ),
            _integrity_row(
                dataset,
                "rows with negative volume",
                negative_volume,
                "retained and reported for review",
            ),
            _integrity_row(
                dataset,
                "rows with inconsistent OHLC bounds",
                inconsistent_ohlc,
                "retained and reported for review",
            ),
            _integrity_row(
                dataset,
                f"missing ticker-dates vs {frequency}",
                gaps["missing_dates"].sum(),
                "reported; no prices imputed",
            ),
            _integrity_row(
                dataset,
                f"absolute returns above {extreme_threshold:.0%}",
                len(review),
                "retained for review; not auto-deleted",
            ),
        ]
    )
    return clean, audit, review, gaps


def clean_headlines(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalise and de-duplicate headlines without altering the raw title text."""
    missing_columns = set(NEWS_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"news_headlines is missing required columns: {sorted(missing_columns)}")

    clean = frame.copy()
    clean["date"] = _normalise_dates(clean["date"])
    invalid_dates = clean["date"].isna()
    out_of_period = clean["date"].notna() & ~clean["date"].between(SAMPLE_START, SAMPLE_END)
    missing_ticker = clean["ticker"].isna() | clean["ticker"].astype("string").str.strip().eq("")
    missing_title = clean["title"].isna() | clean["title"].astype("string").str.strip().eq("")
    unusable_key = invalid_dates | out_of_period | missing_ticker | missing_title
    clean = clean.loc[~unusable_key].copy()
    clean["ticker"] = clean["ticker"].astype("string").str.strip()

    duplicate_keys = clean.duplicated(["ticker", "date", "title"], keep="first")
    duplicate_count = int(duplicate_keys.sum())
    clean = (
        clean.loc[~duplicate_keys].sort_values(["ticker", "date", "title"]).reset_index(drop=True)
    )

    audit = pd.DataFrame(
        [
            _integrity_row(
                "news_headlines",
                "invalid date rows",
                invalid_dates.sum(),
                "removed: unusable date key",
            ),
            _integrity_row(
                "news_headlines",
                "rows outside 2020-2023",
                out_of_period.sum(),
                "removed: outside project sample",
            ),
            _integrity_row(
                "news_headlines",
                "missing ticker rows",
                missing_ticker.sum(),
                "removed: unusable ticker key",
            ),
            _integrity_row(
                "news_headlines",
                "missing title rows",
                missing_title.sum(),
                "removed: unusable headline key",
            ),
            _integrity_row(
                "news_headlines",
                "duplicate ticker-date-title rows",
                duplicate_count,
                "removed duplicate; preserved distinct same-day titles",
            ),
            _integrity_row(
                "news_headlines",
                "missing publisher cells",
                clean["publisher"].isna().sum(),
                "retained: publisher is optional metadata",
            ),
            _integrity_row(
                "news_headlines",
                "missing sector cells",
                clean["sector"].isna().sum(),
                "retained and reported for review",
            ),
        ]
    )
    return clean, audit


def build_dataset_inventory(
    raw: dict[str, pd.DataFrame], clean: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Summarise raw and cleaned coverage for the three provided datasets."""
    frequencies = {
        "equity_prices": "equity trading days",
        "crypto_prices": "calendar daily",
        "news_headlines": "event-level headlines",
    }
    rows = []
    for name in ("equity_prices", "crypto_prices", "news_headlines"):
        frame = clean[name]
        sectors = frame["sector"].nunique(dropna=True) if "sector" in frame else np.nan
        rows.append(
            {
                "dataset": name,
                "raw_rows": len(raw[name]),
                "clean_rows": len(frame),
                "rows_removed": len(raw[name]) - len(frame),
                "start_date": frame["date"].min().date().isoformat(),
                "end_date": frame["date"].max().date().isoformat(),
                "frequency": frequencies[name],
                "ticker_count": frame["ticker"].nunique(dropna=True),
                "sector_count": sectors,
                "coverage": f"{frame['ticker'].nunique(dropna=True)} tickers"
                + (f"; {int(sectors)} sectors" if pd.notna(sectors) else "; no sector field"),
            }
        )
    return pd.DataFrame(rows)


def run_station1() -> Station1Results:
    """Load all hosted datasets and execute the verified Station 1 workflow."""
    raw = {
        "equity_prices": data_access.load_equity_prices().copy(),
        "crypto_prices": data_access.load_crypto_prices().copy(),
        "news_headlines": data_access.load_news_headlines().copy(),
    }
    equities, equity_audit, equity_extremes, equity_gaps = clean_price_panel(
        raw["equity_prices"], "equity_prices", "equity", EQUITY_EXTREME_THRESHOLD
    )
    crypto, crypto_audit, crypto_extremes, crypto_gaps = clean_price_panel(
        raw["crypto_prices"], "crypto_prices", "daily", CRYPTO_EXTREME_THRESHOLD
    )
    headlines, news_audit = clean_headlines(raw["news_headlines"])
    clean = {"equity_prices": equities, "crypto_prices": crypto, "news_headlines": headlines}
    return Station1Results(
        equities=equities,
        crypto=crypto,
        headlines=headlines,
        inventory=build_dataset_inventory(raw, clean),
        integrity=pd.concat([equity_audit, crypto_audit, news_audit], ignore_index=True),
        extreme_returns=pd.concat([equity_extremes, crypto_extremes], ignore_index=True),
        missing_dates=pd.concat([equity_gaps, crypto_gaps], ignore_index=True),
    )


def load_clean_equities() -> pd.DataFrame:
    """Load and clean the equity price panel."""
    clean, _, _, _ = clean_price_panel(
        data_access.load_equity_prices(), "equity_prices", "equity", EQUITY_EXTREME_THRESHOLD
    )
    return clean


def load_clean_crypto() -> pd.DataFrame:
    """Load and clean the crypto price panel on its native daily calendar."""
    clean, _, _, _ = clean_price_panel(
        data_access.load_crypto_prices(), "crypto_prices", "daily", CRYPTO_EXTREME_THRESHOLD
    )
    return clean
