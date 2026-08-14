"""Descriptive portfolio-level news context from saved Part B outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = (
    "date",
    "fund",
    "fund_id",
    "universe",
    "method",
    "weight_effective_date",
    "portfolio_lagged_sentiment",
    "total_equity_weight",
    "observed_equity_weight",
    "missing_equity_weight",
    "sentiment_weight_coverage",
    "observed_sector_count",
    "missing_sector_count",
    "latest_signal_source_date",
)


def validate_portfolio_sentiment_output(frame: pd.DataFrame) -> None:
    """Enforce the saved display artifact's schema, timing, and coverage identities."""
    missing = set(OUTPUT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"portfolio sentiment output missing columns: {sorted(missing)}")
    if frame.duplicated(["fund_id", "date"]).any():
        raise ValueError("portfolio sentiment output has duplicate fund-date rows")
    equity = frame["total_equity_weight"].gt(0)
    if not frame.loc[equity, "sentiment_weight_coverage"].between(0, 1).all():
        raise ValueError("portfolio sentiment coverage must be between zero and one")
    total_error = (
        frame["observed_equity_weight"]
        + frame["missing_equity_weight"]
        - frame["total_equity_weight"]
    ).abs()
    if total_error.gt(1e-9).any():
        raise ValueError("observed and missing equity weights must sum to total equity weight")
    expected_coverage = frame.loc[equity, "observed_equity_weight"].div(
        frame.loc[equity, "total_equity_weight"]
    )
    if not np.allclose(
        frame.loc[equity, "sentiment_weight_coverage"],
        expected_coverage,
        atol=1e-9,
    ):
        raise ValueError("portfolio sentiment coverage does not match observed equity weight")
    observed = frame["portfolio_lagged_sentiment"].notna()
    unsafe = observed & (
        frame["latest_signal_source_date"].isna()
        | frame["latest_signal_source_date"].ge(frame["date"])
    )
    if unsafe.any():
        raise ValueError("portfolio sentiment must originate on an earlier trading date")


def _asset_sector_map(weights: pd.DataFrame) -> pd.Series:
    equity = weights.loc[weights["asset_class"].eq("equity")]
    mapping = equity.dropna(subset=["sector"])[["asset", "sector"]].drop_duplicates()
    if mapping["asset"].duplicated().any():
        raise ValueError("an equity asset maps to multiple saved sectors")
    result = mapping.set_index("asset")["sector"]
    missing = set(equity["asset"]).difference(result.index)
    if missing:
        raise ValueError(f"saved weights lack sector labels for {sorted(missing)[:3]}")
    return result


def build_portfolio_sentiment(
    weights: pd.DataFrame,
    sector_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Weight available lagged sector scores by each fund's active equity targets."""
    weights = weights.copy()
    sector_sentiment = sector_sentiment.copy()
    weights["effective_date"] = pd.to_datetime(weights["effective_date"], errors="raise")
    sector_sentiment["date"] = pd.to_datetime(sector_sentiment["date"], errors="raise")
    sector_sentiment["lagged_source_date"] = pd.to_datetime(
        sector_sentiment["lagged_source_date"], errors="coerce"
    )
    unsafe = sector_sentiment["lagged_sector_sentiment"].notna() & (
        sector_sentiment["lagged_source_date"].isna()
        | sector_sentiment["lagged_source_date"].ge(sector_sentiment["date"])
    )
    if unsafe.any():
        raise ValueError("lagged sector sentiment must originate on an earlier trading date")

    sector_map = _asset_sector_map(weights)
    equity = weights.loc[weights["asset_class"].eq("equity")].copy()
    equity["sector"] = equity["asset"].map(sector_map)
    sector_weights = (
        equity.groupby(
            ["fund", "fund_id", "universe", "method", "effective_date", "sector"],
            observed=True,
            as_index=False,
        )["weight"]
        .sum()
        .rename(columns={"weight": "sector_weight"})
    )
    fund_meta = weights[["fund", "fund_id", "universe", "method"]].drop_duplicates()
    calendar = pd.DatetimeIndex(sector_sentiment["date"].drop_duplicates().sort_values())
    rows: list[dict[str, object]] = []

    for fund in fund_meta.itertuples(index=False):
        fund_weights = weights.loc[weights["fund_id"].eq(fund.fund_id)]
        targets = sector_weights.loc[sector_weights["fund_id"].eq(fund.fund_id)]
        effective_dates = pd.DatetimeIndex(
            fund_weights["effective_date"].drop_duplicates().sort_values()
        )
        first_effective = effective_dates.min()
        dates = pd.DataFrame({"date": calendar[calendar >= first_effective]})
        active_dates = pd.merge_asof(
            dates,
            pd.DataFrame({"weight_effective_date": effective_dates}),
            left_on="date",
            right_on="weight_effective_date",
            direction="backward",
        )
        if targets.empty:
            for date, effective_date in active_dates.itertuples(index=False):
                rows.append(
                    {
                        "date": date,
                        "fund": fund.fund,
                        "fund_id": fund.fund_id,
                        "universe": fund.universe,
                        "method": fund.method,
                        "weight_effective_date": effective_date,
                        "portfolio_lagged_sentiment": np.nan,
                        "total_equity_weight": 0.0,
                        "observed_equity_weight": 0.0,
                        "missing_equity_weight": 0.0,
                        "sentiment_weight_coverage": np.nan,
                        "observed_sector_count": 0,
                        "missing_sector_count": 0,
                        "latest_signal_source_date": pd.NaT,
                    }
                )
            continue

        expanded = active_dates.merge(
            targets[["effective_date", "sector", "sector_weight"]],
            left_on="weight_effective_date",
            right_on="effective_date",
            how="left",
            validate="many_to_many",
        ).merge(
            sector_sentiment[
                ["date", "sector", "lagged_sector_sentiment", "lagged_source_date"]
            ],
            on=["date", "sector"],
            how="left",
            validate="many_to_one",
        )
        expanded["observed"] = expanded["lagged_sector_sentiment"].notna()
        expanded["observed_weight"] = expanded["sector_weight"].where(
            expanded["observed"], 0.0
        )
        expanded["weighted_score"] = (
            expanded["sector_weight"] * expanded["lagged_sector_sentiment"]
        ).fillna(0.0)
        expanded["missing_sector"] = ~expanded["observed"]
        daily = (
            expanded.groupby(["date", "weight_effective_date"], as_index=False)
            .agg(
                total_equity_weight=("sector_weight", "sum"),
                observed_equity_weight=("observed_weight", "sum"),
                weighted_score=("weighted_score", "sum"),
                observed_sector_count=("observed", "sum"),
                missing_sector_count=("missing_sector", "sum"),
                latest_signal_source_date=("lagged_source_date", "max"),
            )
        )
        daily["missing_equity_weight"] = (
            daily["total_equity_weight"] - daily["observed_equity_weight"]
        )
        daily["portfolio_lagged_sentiment"] = daily["weighted_score"].div(
            daily["observed_equity_weight"].replace(0, np.nan)
        )
        daily["sentiment_weight_coverage"] = daily["observed_equity_weight"].div(
            daily["total_equity_weight"].replace(0, np.nan)
        )
        for row in daily.itertuples(index=False):
            rows.append(
                {
                    "date": row.date,
                    "fund": fund.fund,
                    "fund_id": fund.fund_id,
                    "universe": fund.universe,
                    "method": fund.method,
                    "weight_effective_date": row.weight_effective_date,
                    "portfolio_lagged_sentiment": row.portfolio_lagged_sentiment,
                    "total_equity_weight": row.total_equity_weight,
                    "observed_equity_weight": row.observed_equity_weight,
                    "missing_equity_weight": row.missing_equity_weight,
                    "sentiment_weight_coverage": row.sentiment_weight_coverage,
                    "observed_sector_count": int(row.observed_sector_count),
                    "missing_sector_count": int(row.missing_sector_count),
                    "latest_signal_source_date": row.latest_signal_source_date,
                }
            )
    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        ["fund_id", "date"]
    ).reset_index(drop=True)
    validate_portfolio_sentiment_output(output)
    return output


def save_portfolio_sentiment_output(results_root: Path) -> Path:
    """Build the display artifact strictly from existing saved results."""
    data_dir = Path(results_root) / "data"
    weights = pd.read_csv(data_dir / "fund_weights.csv")
    sentiment = pd.read_csv(data_dir / "sector_sentiment_index.csv")
    output = data_dir / "portfolio_sentiment_context.csv"
    build_portfolio_sentiment(weights, sentiment).to_csv(output, index=False)
    return output
