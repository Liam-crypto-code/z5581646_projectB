"""Standalone equity-only and crypto-only fund universe construction."""

from __future__ import annotations

import pandas as pd
from src.portfolios import BacktestResult, oos_backtest

PORTFOLIO_METHODS = ("equal_weight", "min_variance", "max_sharpe")


def _wide_return_panel(returns: pd.DataFrame, prefix: str) -> pd.DataFrame:
    required = {"date", "ticker", "return"}
    missing = required.difference(returns.columns)
    if missing:
        raise ValueError(f"return panel is missing columns: {sorted(missing)}")
    if returns.duplicated(["ticker", "date"]).any():
        raise ValueError("return panel contains duplicate ticker-date rows")
    return (
        returns.pivot(index="date", columns="ticker", values="return")
        .sort_index()
        .add_prefix(prefix)
        .reset_index()
    )


def build_native_universe_panels(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create wide panels without aligning crypto to the equity calendar."""
    equity_panel = _wide_return_panel(equity_returns, "equity__")
    crypto_panel = _wide_return_panel(crypto_returns, "crypto__")
    return equity_panel, crypto_panel


def run_standalone_fund_universes(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> dict[str, BacktestResult]:
    """Run six walk-forward funds with their native calendar conventions."""
    equity_panel, crypto_panel = build_native_universe_panels(equity_returns, crypto_returns)
    results = {
        f"equity_{method}": oos_backtest(
            equity_panel,
            method=method,
            estimation_window=252,
            periods_per_year=252,
        )
        for method in PORTFOLIO_METHODS
    }
    results.update(
        {
            f"crypto_{method}": oos_backtest(
                crypto_panel,
                method=method,
                estimation_window=365,
                periods_per_year=365,
            )
            for method in PORTFOLIO_METHODS
        }
    )
    return results
