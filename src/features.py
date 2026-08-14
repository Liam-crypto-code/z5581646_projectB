"""Verified Part A return processing and headline assembly for Part B reuse."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Station2Results:
    """Derived return panels and aligned headline data from Station 2."""

    equity_returns: pd.DataFrame
    crypto_returns: pd.DataFrame
    combined_returns: pd.DataFrame
    descriptive_stats: pd.DataFrame
    headline_panel: pd.DataFrame
    headline_alignment: pd.DataFrame
    alignment_summary: pd.DataFrame


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
    asset_class: str | None = None,
) -> pd.DataFrame:
    """Calculate simple adjusted-close returns within ticker on native dates."""
    required = {"ticker", "date", price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices are missing required columns: {sorted(missing)}")

    columns = ["ticker", "date", price_col]
    if "sector" in prices.columns:
        columns.append("sector")
    result = prices[columns].copy().sort_values(["ticker", "date"])
    result["return"] = result.groupby("ticker", observed=True)[price_col].pct_change(
        fill_method=None
    )
    result = result.drop(columns=price_col).reset_index(drop=True)
    if asset_class is not None:
        result["asset_class"] = asset_class
    return result


def align_return_calendars(
    equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame
) -> pd.DataFrame:
    """Left-align already-computed crypto returns to the equity calendar."""
    equity_wide = equity_returns.pivot(index="date", columns="ticker", values="return")
    crypto_wide = crypto_returns.pivot(index="date", columns="ticker", values="return")
    equity_wide.columns = [f"equity__{ticker}" for ticker in equity_wide.columns]
    crypto_wide.columns = [f"crypto__{ticker}" for ticker in crypto_wide.columns]
    combined = equity_wide.join(crypto_wide, how="left").sort_index()
    combined.index.name = "date"
    return combined.reset_index()


def descriptive_return_statistics(
    equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame
) -> pd.DataFrame:
    """Summarise daily return distributions by asset class in percentage units."""
    rows = []
    for label, frame in (("Equities", equity_returns), ("Crypto", crypto_returns)):
        values = frame["return"].dropna()
        rows.append(
            {
                "asset_class": label,
                "observations": len(values),
                "mean_daily_return_pct": values.mean() * 100,
                "daily_volatility_pct": values.std(ddof=1) * 100,
                "minimum_daily_return_pct": values.min() * 100,
                "maximum_daily_return_pct": values.max() * 100,
                "skewness": values.skew(),
                "excess_kurtosis": values.kurt(),
                "sample_period": "2020-01-01 to 2023-12-31",
            }
        )
    return pd.DataFrame(rows)


def align_headline_dates(
    headlines: pd.DataFrame, trading_calendar: pd.Series | pd.DatetimeIndex
) -> pd.DataFrame:
    """Map each headline to the same or next observed equity trading day.

    Headlines after the final observed trading day retain a missing
    ``trading_date`` so they cannot leak beyond the project sample.
    """
    required = {"ticker", "date", "sector", "title"}
    missing = required.difference(headlines.columns)
    if missing:
        raise ValueError(f"headlines are missing required columns: {sorted(missing)}")

    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar).unique()).sort_values()
    if calendar.empty:
        raise ValueError("trading_calendar must contain at least one date")

    aligned = headlines.copy()
    positions = calendar.searchsorted(aligned["date"], side="left")
    in_range = positions < len(calendar)
    aligned["trading_date"] = pd.NaT
    aligned.loc[in_range, "trading_date"] = calendar.take(positions[in_range]).to_numpy()
    aligned["calendar_shift_days"] = (aligned["trading_date"] - aligned["date"]).dt.days.astype(
        "Int64"
    )
    return aligned


def assemble_headline_panel(
    headlines: pd.DataFrame, trading_calendar: pd.Series | pd.DatetimeIndex
) -> pd.DataFrame:
    """Assemble unmodified headline text by trading date, ticker, and sector."""
    aligned = align_headline_dates(headlines, trading_calendar)
    usable = aligned.dropna(subset=["trading_date"]).copy()
    return (
        usable.groupby(["trading_date", "ticker", "sector"], observed=True, as_index=False)
        .agg(
            headline_count=("title", "size"),
            headline_text=("title", lambda values: " || ".join(values.astype(str))),
            first_source_date=("date", "min"),
            last_source_date=("date", "max"),
        )
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )


def run_station2(
    equities: pd.DataFrame, crypto: pd.DataFrame, headlines: pd.DataFrame
) -> Station2Results:
    """Execute the verified Part A Station 2 data foundation."""
    equity_returns = daily_returns(equities, asset_class="Equity")
    crypto_returns = daily_returns(crypto, asset_class="Crypto")
    combined_returns = align_return_calendars(equity_returns, crypto_returns)
    descriptive_stats = descriptive_return_statistics(equity_returns, crypto_returns)

    trading_calendar = equities["date"].drop_duplicates().sort_values()
    headline_alignment = align_headline_dates(headlines, trading_calendar)
    headline_panel = assemble_headline_panel(headlines, trading_calendar)
    unmapped = headline_alignment["trading_date"].isna()
    shifted = headline_alignment["calendar_shift_days"].fillna(0).gt(0)
    maximum_shift = headline_alignment["calendar_shift_days"].max()
    alignment_summary = pd.DataFrame(
        [
            {
                "input_headlines": len(headline_alignment),
                "aligned_headlines": int((~unmapped).sum()),
                "same_day_headlines": int(
                    headline_alignment["calendar_shift_days"].fillna(-1).eq(0).sum()
                ),
                "shifted_to_next_trading_day": int(shifted.sum()),
                "unmapped_after_final_trading_day": int(unmapped.sum()),
                "maximum_calendar_shift_days": int(maximum_shift) if pd.notna(maximum_shift) else 0,
                "policy": "same day if traded; otherwise next observed equity trading day",
            }
        ]
    )

    return Station2Results(
        equity_returns=equity_returns,
        crypto_returns=crypto_returns,
        combined_returns=combined_returns,
        descriptive_stats=descriptive_stats,
        headline_panel=headline_panel,
        headline_alignment=headline_alignment,
        alignment_summary=alignment_summary,
    )
