"""Build report-ready Part B exhibits strictly from saved CSV artifacts."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from src.figures import (
    plot_all_fund_drawdowns,
    plot_all_fund_growth,
    plot_fact_sheet,
    plot_sharpe_ratios,
    plot_weight_group_history,
)

FUND_SPECS = {
    "combined_equal_weight": ("Combined Equal Weight", "Combined", "equal_weight"),
    "combined_min_variance": ("Combined Minimum Variance", "Combined", "min_variance"),
    "combined_max_sharpe": ("Combined Maximum Sharpe", "Combined", "max_sharpe"),
    "combined_sentiment_tilt": (
        "Sentiment-Tilted Equal Weight",
        "Combined",
        "sentiment_tilt",
    ),
    "equity_equal_weight": ("Equity-Only Equal Weight", "Equity-only", "equal_weight"),
    "equity_min_variance": (
        "Equity-Only Minimum Variance",
        "Equity-only",
        "min_variance",
    ),
    "equity_max_sharpe": ("Equity-Only Maximum Sharpe", "Equity-only", "max_sharpe"),
    "crypto_equal_weight": ("Crypto-Only Equal Weight", "Crypto-only", "equal_weight"),
    "crypto_min_variance": (
        "Crypto-Only Minimum Variance",
        "Crypto-only",
        "min_variance",
    ),
    "crypto_max_sharpe": ("Crypto-Only Maximum Sharpe", "Crypto-only", "max_sharpe"),
}
EXPECTED_FUNDS = set(FUND_SPECS)
EXPECTED_METHODS = {spec[2] for spec in FUND_SPECS.values()}


def validate_presentation_inputs(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Validate the saved artifact contracts without recomputing any model output."""
    identity_columns = {"fund", "fund_id", "universe", "method"}
    return_columns = identity_columns | {
        "date",
        "portfolio_return",
        "growth_of_1",
        "drawdown",
    }
    weight_columns = {
        *identity_columns,
        "effective_date",
        "asset",
        "weight",
        "asset_class",
    }
    metric_columns = {
        *identity_columns,
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "growth_of_1",
        "start_date",
        "end_date",
        "estimation_window",
        "rebalance_frequency",
        "rebalance_convention",
        "constraints",
        "transaction_costs",
        "risk_free_rate",
    }
    for name, frame, required in (
        ("fund_returns", fund_returns, return_columns),
        ("fund_weights", fund_weights, weight_columns),
        ("performance_metrics", metrics, metric_columns),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    for name, frame in (
        ("fund_returns", fund_returns),
        ("fund_weights", fund_weights),
        ("performance_metrics", metrics),
    ):
        funds = set(frame["fund_id"].dropna())
        if funds != EXPECTED_FUNDS:
            raise ValueError(f"{name} funds are {sorted(funds)}, expected {sorted(EXPECTED_FUNDS)}")

    for universe, group in fund_returns.groupby("universe", observed=True):
        date_sets = group.groupby("fund_id")["date"].agg(lambda values: tuple(values))
        if date_sets.nunique() != 1:
            raise ValueError(f"{universe} funds must use identical comparison dates")


def build_fact_sheet_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select and order report-ready fund metrics and methodology assumptions."""
    columns = [
        "fund",
        "fund_id",
        "universe",
        "method",
        "start_date",
        "end_date",
        "observations",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "growth_of_1",
        "estimation_window",
        "rebalance_frequency",
        "rebalance_convention",
        "constraints",
        "transaction_costs",
        "risk_free_rate",
        "tilt_strength",
        "missing_signal_policy",
    ]
    summary = metrics.copy()
    for column in columns:
        if column not in summary:
            summary[column] = np.nan
    return summary[columns].sort_values("fund").reset_index(drop=True)


def _asset_sector_map(weights: pd.DataFrame) -> pd.Series:
    equity = weights[weights["asset"].str.startswith("equity__")].copy()
    observed = equity.dropna(subset=["sector"])[["asset", "sector"]].drop_duplicates()
    if observed["asset"].duplicated().any():
        raise ValueError("an equity asset maps to multiple sectors in fund_weights.csv")
    mapping = observed.set_index("asset")["sector"]
    missing = set(equity["asset"]).difference(mapping.index)
    if missing:
        raise ValueError(
            "saved weights do not contain a complete equity-sector mapping; "
            f"missing {sorted(missing)[:3]}"
        )
    return mapping


def build_latest_holdings(weights: pd.DataFrame) -> pd.DataFrame:
    """Return every fund's latest monthly target holdings with rank and sector."""
    sector_map = _asset_sector_map(weights)
    latest_dates = weights.groupby("fund_id")["effective_date"].max()
    latest = weights[
        weights.apply(lambda row: row["effective_date"] == latest_dates.loc[row["fund_id"]], axis=1)
    ].copy()
    latest["ticker"] = latest["asset"].str.split("__", n=1).str[1]
    latest["holding_group"] = np.where(
        latest["asset"].str.startswith("crypto__"),
        "Crypto",
        latest["asset"].map(sector_map),
    )
    latest["rank"] = (
        latest.groupby("fund_id")["weight"].rank(method="first", ascending=False).astype(int)
    )
    columns = [
        "fund",
        "fund_id",
        "universe",
        "method",
        "effective_date",
        "rank",
        "ticker",
        "asset_class",
        "holding_group",
        "weight",
    ]
    return latest[columns].sort_values(["fund", "rank"]).reset_index(drop=True)


def build_weight_group_history(weights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly targets into ten equity sectors plus the crypto sleeve."""
    sector_map = _asset_sector_map(weights)
    grouped = weights.copy()
    grouped["allocation_group"] = np.where(
        grouped["asset"].str.startswith("crypto__"),
        "Crypto",
        grouped["asset"].map(sector_map),
    )
    history = (
        grouped.groupby(
            ["effective_date", "fund", "fund_id", "universe", "method", "allocation_group"],
            observed=True,
            as_index=False,
        )["weight"]
        .sum()
        .rename(columns={"weight": "target_weight"})
    )
    totals = history.groupby(["effective_date", "fund_id"])["target_weight"].sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-9):
        raise ValueError("grouped target weights do not sum to one")
    return history.sort_values(["fund_id", "effective_date", "allocation_group"]).reset_index(
        drop=True
    )


def build_presentation_outputs(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Read saved artifacts and create all missing presentation tables and figures."""
    data_dir = root / "results" / "data"
    table_dir = root / "results" / "tables"
    figure_dir = root / "results" / "figures"
    fund_returns = pd.read_csv(data_dir / "fund_returns.csv", parse_dates=["date"])
    fund_weights = pd.read_csv(
        data_dir / "fund_weights.csv",
        parse_dates=["rebalance_date", "effective_date", "estimation_start", "estimation_end"],
    )
    metrics = pd.read_csv(
        table_dir / "performance_metrics.csv", parse_dates=["start_date", "end_date"]
    )
    validate_presentation_inputs(fund_returns, fund_weights, metrics)

    summary = build_fact_sheet_summary(metrics)
    latest_holdings = build_latest_holdings(fund_weights)
    weight_history = build_weight_group_history(fund_weights)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, pathlib.Path] = {}
    paths["fact_sheet_summary"] = table_dir / "fact_sheet_summary.csv"
    paths["fund_comparison"] = table_dir / "fund_comparison_report_table.csv"
    paths["latest_holdings"] = table_dir / "latest_holdings_all_funds.csv"
    paths["weight_history"] = table_dir / "portfolio_weight_groups_over_time.csv"
    summary.to_csv(paths["fact_sheet_summary"], index=False)
    comparison_table = summary[
        [
            "fund",
            "fund_id",
            "universe",
            "method",
            "start_date",
            "end_date",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "growth_of_1",
        ]
    ].copy()
    comparison_table = comparison_table.rename(
        columns={
            "annualized_return": "annualized_return_pct",
            "annualized_volatility": "annualized_volatility_pct",
            "maximum_drawdown": "maximum_drawdown_pct",
            "growth_of_1": "growth_of_1_usd",
        }
    )
    for column in (
        "annualized_return_pct",
        "annualized_volatility_pct",
        "maximum_drawdown_pct",
    ):
        comparison_table[column] *= 100
    comparison_table.to_csv(paths["fund_comparison"], index=False)
    latest_holdings.to_csv(paths["latest_holdings"], index=False)
    weight_history.to_csv(paths["weight_history"], index=False)

    for fund_id in sorted(EXPECTED_FUNDS):
        legacy_name = fund_id.removeprefix("combined_")
        holdings_path = table_dir / f"fact_sheet_{legacy_name}_latest_holdings.csv"
        latest_holdings[latest_holdings["fund_id"] == fund_id].to_csv(holdings_path, index=False)
        paths[f"holdings_{fund_id}"] = holdings_path

    paths["growth"] = figure_dir / "growth_of_1_all_funds.png"
    paths["drawdown"] = figure_dir / "drawdown_all_funds.png"
    paths["weights"] = figure_dir / "portfolio_weights_over_time.png"
    paths["sharpe"] = figure_dir / "sharpe_ratio_by_fund.png"
    combined_returns = fund_returns[fund_returns["universe"] == "Combined"]
    combined_summary = summary[summary["universe"] == "Combined"]
    combined_history = weight_history[weight_history["universe"] == "Combined"]
    plot_all_fund_growth(combined_returns, paths["growth"])
    plot_all_fund_drawdowns(combined_returns, paths["drawdown"])
    plot_weight_group_history(combined_history, paths["weights"])
    plot_sharpe_ratios(combined_summary, paths["sharpe"])

    for fund_id in sorted(EXPECTED_FUNDS):
        legacy_name = fund_id.removeprefix("combined_")
        path = figure_dir / f"fact_sheet_{legacy_name}.png"
        plot_fact_sheet(
            fund_returns[fund_returns["fund_id"] == fund_id].set_index("date"),
            summary.loc[summary["fund_id"] == fund_id].iloc[0],
            latest_holdings[latest_holdings["fund_id"] == fund_id],
            path,
        )
        paths[f"fact_sheet_{fund_id}"] = path
    return paths
