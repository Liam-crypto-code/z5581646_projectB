"""Precomputed data contracts and lightweight analytics for the FinMosaic app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from src.portfolio_sentiment import validate_portfolio_sentiment_output

ASSET_PERFORMANCE_COLUMNS = (
    "date",
    "ticker",
    "asset_type",
    "sector",
    "calendar",
    "periods_per_year",
    "daily_return",
    "growth_of_1",
    "drawdown",
)

APP_ARTIFACTS = {
    "asset_performance": ("data/asset_performance.csv", ASSET_PERFORMANCE_COLUMNS),
    "fund_returns": (
        "data/fund_returns.csv",
        (
            "date",
            "fund",
            "fund_id",
            "universe",
            "method",
            "portfolio_return",
            "growth_of_1",
            "drawdown",
        ),
    ),
    "fund_weights": (
        "data/fund_weights.csv",
        (
            "method",
            "fund",
            "fund_id",
            "universe",
            "effective_date",
            "asset",
            "weight",
            "asset_class",
        ),
    ),
    "company_sentiment": (
        "data/company_sentiment_index.csv",
        (
            "date",
            "ticker",
            "sector",
            "vader_company_sentiment",
            "headline_count",
            "news_observed",
            "news_coverage",
            "lagged_vader_company_sentiment",
        ),
    ),
    "sector_sentiment": (
        "data/sector_sentiment_index.csv",
        (
            "date",
            "sector",
            "sector_sentiment",
            "headline_count",
            "coverage_ratio",
            "lagged_sector_sentiment",
        ),
    ),
    "portfolio_sentiment": (
        "data/portfolio_sentiment_context.csv",
        (
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
        ),
    ),
    "fund_metrics": (
        "tables/fact_sheet_summary.csv",
        (
            "fund",
            "fund_id",
            "universe",
            "method",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "growth_of_1",
        ),
    ),
    "latest_holdings": (
        "tables/latest_holdings_all_funds.csv",
        (
            "fund",
            "fund_id",
            "universe",
            "method",
            "effective_date",
            "ticker",
            "asset_class",
            "weight",
        ),
    ),
    "sentiment_distributions": (
        "tables/sentiment_score_distribution_comparison.csv",
        ("model", "headline_count", "mean_score", "score_std", "neutral_ratio"),
    ),
    "sentiment_model_comparison": (
        "tables/sentiment_model_sector_comparison.csv",
        (
            "sector",
            "pearson_correlation",
            "classification_disagreement_ratio",
            "mean_absolute_score_difference",
        ),
    ),
    "sentiment_disagreements": (
        "tables/sentiment_disagreement_examples.csv",
        ("trading_date", "ticker", "sector", "title", "vader_compound", "lm_tone"),
    ),
    "lm_coverage": (
        "tables/lm_coverage_by_sector.csv",
        ("sector", "matched_headline_ratio", "matched_token_ratio"),
    ),
    "vader_methodology": ("tables/sentiment_methodology.csv", ("model", "no_news_policy")),
    "lm_methodology": (
        "tables/lm_sentiment_methodology.csv",
        ("model", "version_date", "scoring_formula", "no_news_policy"),
    ),
    "fusion_methodology": (
        "tables/fusion_methodology.csv",
        ("base_fund", "equation", "signal", "missing_signal_policy"),
    ),
}

DATE_COLUMNS = {
    "asset_performance": ("date",),
    "fund_returns": ("date",),
    "fund_weights": ("effective_date",),
    "company_sentiment": ("date", "lagged_source_date"),
    "sector_sentiment": ("date", "lagged_source_date"),
    "portfolio_sentiment": ("date", "weight_effective_date", "latest_signal_source_date"),
    "fund_metrics": ("start_date", "end_date"),
    "latest_holdings": ("effective_date",),
    "sentiment_disagreements": ("trading_date",),
}


@dataclass(frozen=True)
class AppDataBundle:
    """Validated collection of precomputed app-readable outputs."""

    frames: dict[str, pd.DataFrame]
    results_root: Path

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.frames[name]


def _prepare_asset_panel(
    frame: pd.DataFrame,
    *,
    asset_type: str,
    calendar: str,
    periods_per_year: int,
) -> pd.DataFrame:
    required = {"date", "ticker", "return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{asset_type} returns missing columns: {sorted(missing)}")

    columns = ["date", "ticker", "return"]
    if "sector" in frame.columns:
        columns.append("sector")
    result = frame[columns].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    if "sector" not in result:
        result["sector"] = pd.NA
    result = result.rename(columns={"return": "daily_return"})
    result["asset_type"] = asset_type
    result["calendar"] = calendar
    result["periods_per_year"] = periods_per_year
    result["growth_of_1"] = result.groupby("ticker", observed=True)["daily_return"].transform(
        lambda values: (1.0 + values.fillna(0.0)).cumprod()
    )
    running_peak = result.groupby("ticker", observed=True)["growth_of_1"].cummax()
    result["drawdown"] = result["growth_of_1"] / running_peak - 1.0
    return result.loc[:, ASSET_PERFORMANCE_COLUMNS]


def build_asset_performance(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Build native-calendar asset histories from verified Part A return panels."""
    equity = _prepare_asset_panel(
        equity_returns,
        asset_type="Equity",
        calendar="equity trading calendar",
        periods_per_year=252,
    )
    crypto = _prepare_asset_panel(
        crypto_returns,
        asset_type="Crypto",
        calendar="native calendar daily",
        periods_per_year=365,
    )
    result = pd.concat([equity, crypto], ignore_index=True)
    duplicates = result.duplicated(["asset_type", "ticker", "date"])
    if duplicates.any():
        raise ValueError("asset performance contains duplicate asset-date rows")
    return result.sort_values(["asset_type", "ticker", "date"]).reset_index(drop=True)


def asset_performance_methodology() -> pd.DataFrame:
    """Document the precomputed return artifact and calendar conventions."""
    return pd.DataFrame(
        [
            {
                "output": "results/data/asset_performance.csv",
                "source": "verified Part A adjusted-close simple-return panels",
                "equity_calendar": "observed equity trading dates; 252 periods per year",
                "crypto_calendar": "native calendar dates; 365 periods per year",
                "daily_return": "adjusted-close simple return computed within ticker",
                "growth_of_1": "cumulative product of one plus native-calendar daily returns",
                "drawdown": "growth of $1 divided by its prior running peak, minus one",
                "app_policy": "read saved artifact only; never load prices or recompute returns",
            }
        ]
    )


def save_asset_performance_outputs(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
    results_root: Path,
) -> tuple[Path, Path]:
    """Write the app artifact and its reportable methodology record."""
    data_path = results_root / "data" / "asset_performance.csv"
    methodology_path = results_root / "tables" / "asset_performance_methodology.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    methodology_path.parent.mkdir(parents=True, exist_ok=True)
    build_asset_performance(equity_returns, crypto_returns).to_csv(data_path, index=False)
    asset_performance_methodology().to_csv(methodology_path, index=False)
    return data_path, methodology_path


def _read_validated_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required precomputed app artifact not found: {path}")
    frame = pd.read_csv(path)
    missing = set(required_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")
    return frame


def load_app_data(results_root: Path) -> AppDataBundle:
    """Load and validate app inputs exclusively from a results directory."""
    results_root = Path(results_root)
    frames: dict[str, pd.DataFrame] = {}
    for name, (relative_path, required_columns) in APP_ARTIFACTS.items():
        frame = _read_validated_csv(results_root / relative_path, required_columns)
        for column in DATE_COLUMNS.get(name, ()):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if name == "portfolio_sentiment":
            validate_portfolio_sentiment_output(frame)
        frames[name] = frame
    return AppDataBundle(frames=frames, results_root=results_root)


def selected_asset_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise risk from saved daily returns over a selected date interval."""
    rows = []
    for (asset_type, ticker), sample in frame.groupby(["asset_type", "ticker"], observed=True):
        sample = sample.sort_values("date")
        returns = sample["daily_return"].dropna()
        if returns.empty:
            continue
        periods = int(sample["periods_per_year"].iloc[0])
        saved_growth = sample.loc[returns.index, "growth_of_1"]
        starting_value = saved_growth.iloc[0] / (1.0 + returns.iloc[0])
        growth = saved_growth / starting_value
        drawdown = growth / growth.cummax().clip(lower=1.0) - 1.0
        rows.append(
            {
                "asset_type": asset_type,
                "ticker": ticker,
                "observations": len(returns),
                "total_return": growth.iloc[-1] - 1.0,
                "annualized_return": growth.iloc[-1] ** (periods / len(returns)) - 1.0,
                "annualized_volatility": returns.std(ddof=1) * np.sqrt(periods),
                "maximum_drawdown": drawdown.min(),
                "best_day": returns.max(),
                "worst_day": returns.min(),
                "calendar": sample["calendar"].iloc[0],
            }
        )
    return pd.DataFrame(rows)
