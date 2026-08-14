"""Transparent walk-forward portfolio backtesting for Station 3.

The engine accepts an already prepared native-calendar return panel. A rebalance
is scheduled on the first observed date of each month. Its weights use the
specified trailing window ending on the previous observed date, are formed at
the rebalance close, and become effective on the following observed date. This
keeps the rebalance-date return out of the new weights and out of the return
earned by those weights.

Initial assumptions: long-only, fully invested, zero risk-free rate, and zero
transaction costs. The caller supplies the estimation window and annualisation
convention appropriate to the asset universe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MAX_SHARPE_WEIGHT_CAP = 0.10
SUPPORTED_METHODS = {"equal_weight", "max_sharpe", "min_variance"}


@dataclass
class BacktestResult:
    """Daily out-of-sample results, rebalance weights, metrics, and assumptions."""

    daily: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, object]
    method: str
    estimation_window: int
    rebalance_convention: str
    transaction_costs: float


def _prepare_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate a wide return panel and remove only leading incomplete rows."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")

    panel = returns.copy()
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"], errors="raise")
        panel = panel.set_index("date")
    else:
        panel.index = pd.to_datetime(panel.index, errors="raise")
        panel.index.name = "date"

    if panel.index.has_duplicates:
        raise ValueError("returns contains duplicate dates")
    panel = panel.sort_index()
    if panel.shape[1] < 2:
        raise ValueError("a portfolio requires at least two assets")

    panel = panel.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    complete_rows = panel.notna().all(axis=1)
    if not complete_rows.any():
        raise ValueError("returns has no complete cross-section")
    first_complete = int(np.flatnonzero(complete_rows.to_numpy())[0])
    panel = panel.iloc[first_complete:]
    if panel.isna().any().any():
        missing_dates = panel.index[panel.isna().any(axis=1)]
        raise ValueError(
            "returns contains missing values after the first complete date; "
            f"first affected date is {missing_dates[0].date()}"
        )
    if (panel <= -1).any().any():
        raise ValueError("simple returns must be greater than -100%")
    return panel.astype(float)


def _first_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the first observed trading date in each calendar month."""
    dates = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(dates.groupby(index.to_period("M")).first().to_numpy())


def _equal_weights(asset_count: int) -> np.ndarray:
    return np.full(asset_count, 1.0 / asset_count)


def _minimum_variance_weights(history: pd.DataFrame) -> np.ndarray:
    """Solve the long-only, fully invested sample minimum-variance portfolio."""
    covariance = history.cov().to_numpy(dtype=float)
    if not np.isfinite(covariance).all():
        raise ValueError("covariance matrix contains non-finite values")

    asset_count = covariance.shape[0]
    initial = _equal_weights(asset_count)
    average_variance = float(np.diag(covariance).mean())
    if average_variance <= 0:
        raise ValueError("covariance matrix has no positive asset variance")
    # Normalise the objective to order one without changing its minimiser.
    scale = 1.0 / average_variance

    result = minimize(
        fun=lambda weights: scale * float(weights @ covariance @ weights),
        x0=initial,
        jac=lambda weights: scale * 2.0 * covariance @ weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * asset_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"minimum-variance optimisation failed: {result.message}")

    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    return weights


def _maximum_sharpe_weights(history: pd.DataFrame) -> np.ndarray:
    """Solve the long-only maximum-Sharpe portfolio with a fixed 10% asset cap."""
    asset_count = history.shape[1]
    if asset_count * MAX_SHARPE_WEIGHT_CAP < 1.0 - 1e-12:
        raise ValueError(
            f"at least {int(np.ceil(1.0 / MAX_SHARPE_WEIGHT_CAP))} assets are required "
            "for a fully invested portfolio with a 10% cap"
        )

    expected_returns = history.mean().to_numpy(dtype=float)
    covariance = history.cov().to_numpy(dtype=float)
    if not np.isfinite(expected_returns).all() or not np.isfinite(covariance).all():
        raise ValueError("return estimates contain non-finite values")

    average_variance = float(np.diag(covariance).mean())
    if average_variance <= 0:
        raise ValueError("covariance matrix has no positive asset variance")
    return_scale = np.sqrt(average_variance)
    scaled_mean = expected_returns / return_scale
    scaled_covariance = covariance / average_variance

    def objective(weights: np.ndarray) -> float:
        variance = float(weights @ scaled_covariance @ weights)
        if variance <= 0:
            return np.inf
        return -float(weights @ scaled_mean) / np.sqrt(variance)

    def gradient(weights: np.ndarray) -> np.ndarray:
        covariance_weights = scaled_covariance @ weights
        variance = float(weights @ covariance_weights)
        volatility = np.sqrt(variance)
        portfolio_mean = float(weights @ scaled_mean)
        return -(scaled_mean / volatility - portfolio_mean * covariance_weights / volatility**3)

    result = minimize(
        fun=objective,
        jac=gradient,
        x0=_equal_weights(asset_count),
        method="SLSQP",
        bounds=[(0.0, MAX_SHARPE_WEIGHT_CAP)] * asset_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"maximum-Sharpe optimisation failed: {result.message}")

    weights = np.clip(result.x, 0.0, MAX_SHARPE_WEIGHT_CAP)
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise RuntimeError("maximum-Sharpe weights do not sum to one after optimisation")
    weights /= weights.sum()
    if weights.max() > MAX_SHARPE_WEIGHT_CAP + 1e-8:
        raise RuntimeError("maximum-Sharpe weights exceed the 10% asset cap")
    return weights


def _estimate_weights(history: pd.DataFrame, method: str) -> np.ndarray:
    if method == "equal_weight":
        return _equal_weights(history.shape[1])
    if method == "min_variance":
        return _minimum_variance_weights(history)
    if method == "max_sharpe":
        return _maximum_sharpe_weights(history)
    raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)}")


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculate geometric return, volatility, Sharpe, drawdown, and terminal growth."""
    values = pd.to_numeric(daily_returns, errors="coerce").dropna().astype(float)
    if values.empty:
        raise ValueError("daily_returns must contain at least one finite observation")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if (values <= -1).any():
        raise ValueError("simple returns must be greater than -100%")

    growth = (1.0 + values).cumprod()
    annualized_return = growth.iloc[-1] ** (periods_per_year / len(values)) - 1.0
    daily_volatility = values.std(ddof=1)
    annualized_volatility = daily_volatility * np.sqrt(periods_per_year)
    sharpe_ratio = (
        values.mean() / daily_volatility * np.sqrt(periods_per_year)
        if daily_volatility > 0
        else np.nan
    )
    drawdown = growth / growth.cummax() - 1.0
    return {
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "maximum_drawdown": float(drawdown.min()),
        "growth_of_1": float(growth.iloc[-1]),
        "observations": len(values),
        "start_date": values.index.min(),
        "end_date": values.index.max(),
        "periods_per_year": int(periods_per_year),
    }


def oos_backtest(
    returns: pd.DataFrame,
    method: str = "min_variance",
    estimation_window: int = 252,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run a monthly walk-forward out-of-sample backtest.

    New weights are estimated on the first trading day of a month using exactly
    ``estimation_window`` observations strictly before that date. They become
    effective on the next observed trading day. Existing weights, when present,
    continue to earn the rebalance-date return.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)}")
    if estimation_window < 2:
        raise ValueError("estimation_window must be at least 2")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    panel = _prepare_returns(returns)
    if len(panel) <= estimation_window:
        raise ValueError("returns does not contain enough observations for an out-of-sample period")

    positions = pd.Series(np.arange(len(panel)), index=panel.index)
    schedules: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for rebalance_date in _first_trading_days(panel.index):
        position = int(positions.loc[rebalance_date])
        if position < estimation_window or position + 1 >= len(panel):
            continue

        history = panel.iloc[position - estimation_window : position]
        effective_date = panel.index[position + 1]
        weights = _estimate_weights(history, method)
        schedules.append({"effective_date": effective_date, "weights": weights})
        for asset, weight in zip(panel.columns, weights, strict=True):
            weight_rows.append(
                {
                    "method": method,
                    "rebalance_date": rebalance_date,
                    "effective_date": effective_date,
                    "estimation_start": history.index[0],
                    "estimation_end": history.index[-1],
                    "asset": asset,
                    "weight": float(weight),
                }
            )

    if not schedules:
        raise ValueError("no monthly rebalance has enough prior observations")

    effective_weights = {
        schedule["effective_date"]: schedule["weights"] for schedule in schedules
    }
    current_weights: np.ndarray | None = None
    daily_rows = []
    first_effective_date = schedules[0]["effective_date"]
    for date, asset_returns in panel.loc[first_effective_date:].iterrows():
        if date in effective_weights:
            current_weights = effective_weights[date]
        if current_weights is None:
            continue
        return_vector = asset_returns.to_numpy()
        portfolio_return = float(return_vector @ current_weights)
        daily_rows.append({"date": date, "portfolio_return": portfolio_return})
        # With monthly rather than daily rebalancing, holdings drift after each
        # close and become the next trading day's beginning weights.
        current_weights = current_weights * (1.0 + return_vector) / (1.0 + portfolio_return)

    daily = pd.DataFrame(daily_rows).set_index("date")
    daily["growth_of_1"] = (1.0 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["growth_of_1"] / daily["growth_of_1"].cummax() - 1.0
    daily.insert(0, "method", method)

    metrics = performance_metrics(daily["portfolio_return"], periods_per_year)
    maximum_weight = MAX_SHARPE_WEIGHT_CAP if method == "max_sharpe" else 1.0
    constraints = "long-only; fully invested"
    if method == "max_sharpe":
        constraints += "; maximum 10% per asset"
    metrics.update(
        {
            "method": method,
            "estimation_window": estimation_window,
            "rebalance_frequency": "monthly",
            "rebalance_convention": "first trading day; effective next trading day",
            "constraints": constraints,
            "maximum_weight": maximum_weight,
            "transaction_costs": 0.0,
            "risk_free_rate": 0.0,
        }
    )
    return BacktestResult(
        daily=daily,
        weights=pd.DataFrame(weight_rows),
        metrics=metrics,
        method=method,
        estimation_window=estimation_window,
        rebalance_convention="first trading day; effective next trading day",
        transaction_costs=0.0,
    )
