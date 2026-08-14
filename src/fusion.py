"""Look-ahead-safe sentiment tilt of the Combined Equal Weight fund.

At each existing monthly rebalance, equity asset ``i`` in sector ``s`` receives
the raw target

    raw_weight[i,t] = base_weight[i,t] * (1 + 0.50 * lagged_sentiment[s,t]).

Missing sector signals use a multiplier of one (no directional tilt) while
remaining missing in the sentiment dataset. Raw equity targets are normalized
back to the base fund's total equity allocation. Every cryptocurrency target is
copied exactly from the base fund. The fixed 0.50 strength is pre-specified and
is not selected using out-of-sample performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from src.portfolios import BacktestResult, performance_metrics

SENTIMENT_TILT_STRENGTH = 0.50
FUSION_METHOD = "sentiment_tilt"


@dataclass
class FusionResult:
    """Daily tilted-fund results, monthly targets, metrics, and fixed rule metadata."""

    daily: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, object]
    method: str = FUSION_METHOD
    estimation_window: int = 252
    rebalance_convention: str = "first trading day; effective next trading day"
    transaction_costs: float = 0.0


def _prepare_return_panel(returns: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    panel = returns.copy()
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"], errors="raise")
        panel = panel.set_index("date")
    else:
        panel.index = pd.to_datetime(panel.index, errors="raise")
    missing = set(assets).difference(panel.columns)
    if missing:
        raise ValueError(f"returns is missing base-fund assets: {sorted(missing)}")
    panel = panel[assets].sort_index().apply(pd.to_numeric, errors="coerce")
    return panel


def _validate_sentiment_index(sector_index: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "sector", "lagged_source_date", "lagged_sector_sentiment"}
    missing = required.difference(sector_index.columns)
    if missing:
        raise ValueError(f"sector_index is missing required columns: {sorted(missing)}")
    sentiment = sector_index[list(required)].copy()
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="raise")
    sentiment["lagged_source_date"] = pd.to_datetime(
        sentiment["lagged_source_date"], errors="coerce"
    )
    if sentiment.duplicated(["date", "sector"]).any():
        raise ValueError("sector_index contains duplicate date-sector rows")
    unsafe = sentiment["lagged_sector_sentiment"].notna() & (
        sentiment["lagged_source_date"].isna()
        | (sentiment["lagged_source_date"] >= sentiment["date"])
    )
    if unsafe.any():
        raise ValueError("lagged sentiment must come from an earlier trading date")
    return sentiment


def apply_sentiment_tilt(
    returns: pd.DataFrame,
    base_result: BacktestResult,
    sector_index: pd.DataFrame,
    equity_sector_map: pd.DataFrame,
    periods_per_year: int = 252,
) -> FusionResult:
    """Apply the fixed sector tilt to Equal Weight monthly targets only."""
    if base_result.method != "equal_weight":
        raise ValueError("sentiment fusion requires Combined Equal Weight as the base fund")
    map_columns = {"ticker", "sector"}
    missing_map = map_columns.difference(equity_sector_map.columns)
    if missing_map:
        raise ValueError(f"equity_sector_map is missing columns: {sorted(missing_map)}")
    ticker_to_sector = (
        equity_sector_map[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]
    )
    if ticker_to_sector.index.duplicated().any():
        raise ValueError("each equity ticker must map to exactly one sector")

    base_weights = base_result.weights.copy()
    required_weights = {"rebalance_date", "effective_date", "asset", "weight"}
    missing_weights = required_weights.difference(base_weights.columns)
    if missing_weights:
        raise ValueError(f"base weights is missing columns: {sorted(missing_weights)}")
    base_weights["rebalance_date"] = pd.to_datetime(base_weights["rebalance_date"])
    base_weights["effective_date"] = pd.to_datetime(base_weights["effective_date"])
    assets = base_weights["asset"].drop_duplicates().tolist()
    panel = _prepare_return_panel(returns, assets)
    sentiment = _validate_sentiment_index(sector_index)
    sentiment_lookup = sentiment.set_index(["date", "sector"])

    schedule: dict[pd.Timestamp, np.ndarray] = {}
    weight_rows: list[dict[str, object]] = []
    for effective_date, group in base_weights.groupby("effective_date", sort=True):
        group = group.set_index("asset").reindex(assets)
        if group["weight"].isna().any():
            raise ValueError(f"base target is incomplete on {effective_date.date()}")
        base_target = group["weight"].to_numpy(dtype=float)
        equity_mask = np.array([asset.startswith("equity__") for asset in assets])
        crypto_mask = np.array([asset.startswith("crypto__") for asset in assets])
        if not (equity_mask | crypto_mask).all():
            raise ValueError("assets must use equity__ or crypto__ prefixes")

        equity_total = float(base_target[equity_mask].sum())
        fused_target = base_target.copy()
        multipliers = np.ones(len(assets))
        signals = np.full(len(assets), np.nan)
        sectors: list[str | None] = [None] * len(assets)
        for position, asset in enumerate(assets):
            if not equity_mask[position]:
                continue
            ticker = asset.split("__", maxsplit=1)[1]
            if ticker not in ticker_to_sector.index:
                raise ValueError(f"missing sector mapping for {ticker}")
            sector = str(ticker_to_sector.loc[ticker])
            sectors[position] = sector
            try:
                row = sentiment_lookup.loc[(effective_date, sector)]
            except KeyError as exc:
                raise ValueError(
                    f"sentiment index has no {sector} row for {effective_date.date()}"
                ) from exc
            signal = row["lagged_sector_sentiment"]
            if pd.notna(signal):
                signals[position] = float(signal)
                multipliers[position] = 1.0 + SENTIMENT_TILT_STRENGTH * float(signal)

        raw_equity = base_target[equity_mask] * multipliers[equity_mask]
        if (raw_equity < 0).any() or raw_equity.sum() <= 0:
            raise ValueError("sentiment tilt produced invalid raw equity weights")
        fused_target[equity_mask] = equity_total * raw_equity / raw_equity.sum()
        fused_target[crypto_mask] = base_target[crypto_mask]

        if (fused_target < -1e-12).any() or not np.isclose(fused_target.sum(), 1.0, atol=1e-10):
            raise RuntimeError("fused target violates long-only fully invested constraints")
        if not np.isclose(fused_target[equity_mask].sum(), equity_total, atol=1e-10):
            raise RuntimeError("fused target changes the total equity allocation")
        if not np.allclose(fused_target[crypto_mask], base_target[crypto_mask], atol=1e-12):
            raise RuntimeError("fused target changes cryptocurrency weights")
        schedule[pd.Timestamp(effective_date)] = fused_target

        for position, asset in enumerate(assets):
            original = group.loc[asset]
            weight_rows.append(
                {
                    "method": FUSION_METHOD,
                    "rebalance_date": original["rebalance_date"],
                    "effective_date": effective_date,
                    "estimation_start": original.get("estimation_start", pd.NaT),
                    "estimation_end": original.get("estimation_end", pd.NaT),
                    "asset": asset,
                    "weight": float(fused_target[position]),
                    "base_weight": float(base_target[position]),
                    "sector": sectors[position],
                    "lagged_sector_sentiment": signals[position],
                    "signal_missing": bool(equity_mask[position] and np.isnan(signals[position])),
                    "tilt_multiplier": float(multipliers[position]),
                }
            )

    evaluation_dates = base_result.daily.index
    panel = panel.reindex(evaluation_dates)
    if panel.isna().any().any():
        raise ValueError("returns does not cover every base-fund evaluation date")
    current_weights: np.ndarray | None = None
    daily_rows = []
    for date, asset_returns in panel.iterrows():
        if date in schedule:
            current_weights = schedule[date].copy()
        if current_weights is None:
            raise RuntimeError("no fused target is effective on the first comparison date")
        return_vector = asset_returns.to_numpy(dtype=float)
        portfolio_return = float(return_vector @ current_weights)
        daily_rows.append({"date": date, "portfolio_return": portfolio_return})
        current_weights = current_weights * (1.0 + return_vector) / (1.0 + portfolio_return)

    daily = pd.DataFrame(daily_rows).set_index("date")
    if not daily.index.equals(base_result.daily.index):
        raise RuntimeError("fusion and base fund evaluation dates differ")
    daily["growth_of_1"] = (1.0 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["growth_of_1"] / daily["growth_of_1"].cummax() - 1.0
    daily.insert(0, "method", FUSION_METHOD)

    metrics = performance_metrics(daily["portfolio_return"], periods_per_year)
    metrics.update(
        {
            "method": FUSION_METHOD,
            "estimation_window": base_result.estimation_window,
            "rebalance_frequency": "monthly",
            "rebalance_convention": base_result.rebalance_convention,
            "constraints": (
                "long-only; fully invested; crypto targets unchanged; equity sleeve conserved"
            ),
            "maximum_weight": np.nan,
            "transaction_costs": 0.0,
            "risk_free_rate": 0.0,
            "tilt_strength": SENTIMENT_TILT_STRENGTH,
            "missing_signal_policy": "multiplier one; retain missing value",
        }
    )
    return FusionResult(
        daily=daily,
        weights=pd.DataFrame(weight_rows),
        metrics=metrics,
        estimation_window=base_result.estimation_window,
        rebalance_convention=base_result.rebalance_convention,
    )
