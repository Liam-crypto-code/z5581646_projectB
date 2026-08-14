"""Focused tests for walk-forward portfolio construction and timing."""

import numpy as np
import pandas as pd
import pytest
from src.portfolios import oos_backtest


def _timing_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-24", "2022-03-08")
    sequence = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "asset_a": 0.001 + sequence * 0.00001,
            "asset_b": -0.0005 + (sequence % 3) * 0.0002,
        }
    )


def _maximum_sharpe_panel() -> pd.DataFrame:
    """Deterministic 12-asset panel with distinct means and covariance."""
    dates = pd.bdate_range("2022-01-03", "2022-04-08")
    time = np.arange(len(dates), dtype=float)
    data: dict[str, object] = {"date": dates}
    for asset_number in range(12):
        cycle = np.sin(time * (asset_number + 1) / 7.0) * (0.0007 + asset_number * 0.00003)
        common = np.cos(time / 5.0) * 0.0005
        data[f"asset_{asset_number:02d}"] = (
            0.00005 + asset_number * 0.000025 + cycle + common
        )
    return pd.DataFrame(data)


def test_rebalance_uses_only_prior_dates_and_becomes_effective_next_day():
    result = oos_backtest(_timing_panel(), method="equal_weight", estimation_window=5)
    first_weights = result.weights.loc[
        result.weights["rebalance_date"] == result.weights["rebalance_date"].min()
    ]

    rebalance_date = first_weights["rebalance_date"].iloc[0]
    effective_date = first_weights["effective_date"].iloc[0]
    assert first_weights["estimation_end"].max() < rebalance_date < effective_date
    assert result.daily.index.min() == effective_date
    assert rebalance_date not in result.daily.index


def test_rebalance_day_return_cannot_change_weights_formed_that_day():
    baseline = _timing_panel()
    shocked = baseline.copy()
    first_rebalance = pd.Timestamp("2022-02-01")
    shocked.loc[shocked["date"] == first_rebalance, "asset_a"] = 0.90

    base_result = oos_backtest(baseline, method="min_variance", estimation_window=5)
    shock_result = oos_backtest(shocked, method="min_variance", estimation_window=5)
    base_weights = base_result.weights.loc[
        base_result.weights["rebalance_date"] == first_rebalance, "weight"
    ].to_numpy()
    shock_weights = shock_result.weights.loc[
        shock_result.weights["rebalance_date"] == first_rebalance, "weight"
    ].to_numpy()

    assert shock_weights == pytest.approx(base_weights)


def test_equal_weight_return_and_long_only_fully_invested_constraints():
    panel = _timing_panel()
    result = oos_backtest(panel, method="equal_weight", estimation_window=5)
    first_date = result.daily.index.min()
    expected = panel.loc[panel["date"] == first_date, ["asset_a", "asset_b"]].mean(axis=1).item()
    totals = result.weights.groupby("rebalance_date")["weight"].sum()

    assert result.daily.loc[first_date, "portfolio_return"] == pytest.approx(expected)
    assert totals.to_numpy() == pytest.approx(np.ones(len(totals)))
    assert result.weights["weight"].between(0, 1).all()


def test_weights_drift_between_monthly_rebalances_instead_of_resetting_daily():
    dates = pd.bdate_range("2022-01-25", "2022-02-03")
    panel = pd.DataFrame({"date": dates, "asset_a": 0.0, "asset_b": 0.0})
    panel.loc[panel["date"].isin(pd.to_datetime(["2022-02-02", "2022-02-03"])), "asset_a"] = 0.10

    result = oos_backtest(panel, method="equal_weight", estimation_window=5)
    first_return = result.daily.loc[pd.Timestamp("2022-02-02"), "portfolio_return"]
    second_return = result.daily.loc[pd.Timestamp("2022-02-03"), "portfolio_return"]
    drifted_asset_a_weight = 0.5 * 1.10 / 1.05

    assert first_return == pytest.approx(0.05)
    assert second_return == pytest.approx(drifted_asset_a_weight * 0.10)


def test_maximum_sharpe_timing_excludes_rebalance_day_return():
    baseline = _maximum_sharpe_panel()
    shocked = baseline.copy()
    rebalance_date = pd.Timestamp("2022-02-01")
    shocked.loc[shocked["date"] == rebalance_date, "asset_11"] = 0.90

    base_result = oos_backtest(baseline, method="max_sharpe", estimation_window=15)
    shock_result = oos_backtest(shocked, method="max_sharpe", estimation_window=15)
    base_weights = base_result.weights.loc[
        base_result.weights["rebalance_date"] == rebalance_date
    ].sort_values("asset")
    shock_weights = shock_result.weights.loc[
        shock_result.weights["rebalance_date"] == rebalance_date
    ].sort_values("asset")

    assert base_weights["estimation_end"].max() < rebalance_date
    assert base_weights["effective_date"].min() > rebalance_date
    assert shock_weights["weight"].to_numpy() == pytest.approx(base_weights["weight"].to_numpy())


def test_maximum_sharpe_weights_obey_long_only_full_investment_and_ten_percent_cap():
    result = oos_backtest(_maximum_sharpe_panel(), method="max_sharpe", estimation_window=15)
    totals = result.weights.groupby("rebalance_date")["weight"].sum()

    assert result.weights["weight"].ge(-1e-10).all()
    assert result.weights["weight"].le(0.10 + 1e-8).all()
    assert totals.to_numpy() == pytest.approx(np.ones(len(totals)), abs=1e-8)
    assert result.metrics["maximum_weight"] == pytest.approx(0.10)


def test_maximum_sharpe_weights_differ_from_existing_methods():
    panel = _maximum_sharpe_panel()
    results = {
        method: oos_backtest(panel, method=method, estimation_window=15)
        for method in ("equal_weight", "min_variance", "max_sharpe")
    }
    first_rebalance = results["max_sharpe"].weights["rebalance_date"].min()

    def target(method: str) -> np.ndarray:
        return (
            results[method]
            .weights.loc[results[method].weights["rebalance_date"] == first_rebalance]
            .sort_values("asset")["weight"]
            .to_numpy()
        )

    assert not np.allclose(target("max_sharpe"), target("equal_weight"), atol=1e-6)
    assert not np.allclose(target("max_sharpe"), target("min_variance"), atol=1e-6)
