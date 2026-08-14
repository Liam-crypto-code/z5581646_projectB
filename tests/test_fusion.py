"""Focused tests for the fixed, lagged equity-sector sentiment tilt."""

import numpy as np
import pandas as pd
import pytest
from src.fusion import SENTIMENT_TILT_STRENGTH, apply_sentiment_tilt
from src.portfolios import oos_backtest


def _fusion_inputs() -> tuple[pd.DataFrame, object, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2022-01-24", "2022-03-10")
    time = np.arange(len(dates), dtype=float)
    returns = pd.DataFrame(
        {
            "date": dates,
            "equity__AAA": 0.001 + 0.0002 * np.sin(time / 2),
            "equity__BBB": 0.0005 + 0.0003 * np.cos(time / 3),
            "crypto__BTC-USD": 0.0015 + 0.0005 * np.sin(time / 4),
            "crypto__ETH-USD": 0.0010 + 0.0004 * np.cos(time / 5),
        }
    )
    base = oos_backtest(returns, method="equal_weight", estimation_window=5)
    sectors = ["Energy", "Tech"]
    rows = []
    for position, date in enumerate(dates):
        source_date = dates[position - 1] if position > 0 else pd.NaT
        for sector in sectors:
            rows.append(
                {
                    "date": date,
                    "sector": sector,
                    "sector_sentiment": -0.9,
                    "lagged_source_date": source_date,
                    "lagged_sector_sentiment": (
                        0.6
                        if sector == "Tech" and position > 0
                        else -0.4
                        if position > 0
                        else np.nan
                    ),
                }
            )
    sentiment = pd.DataFrame(rows)
    sector_map = pd.DataFrame({"ticker": ["AAA", "BBB"], "sector": ["Tech", "Energy"]})
    return returns, base, sentiment, sector_map


def test_fusion_uses_only_lagged_sentiment_and_rejects_unsafe_dates():
    returns, base, sentiment, sector_map = _fusion_inputs()
    baseline = apply_sentiment_tilt(returns, base, sentiment, sector_map)
    changed_contemporaneous = sentiment.copy()
    changed_contemporaneous["sector_sentiment"] = 0.99
    unchanged = apply_sentiment_tilt(returns, base, changed_contemporaneous, sector_map)

    assert unchanged.weights["weight"].to_numpy() == pytest.approx(
        baseline.weights["weight"].to_numpy()
    )

    unsafe = sentiment.copy()
    effective_date = base.weights["effective_date"].min()
    unsafe.loc[unsafe["date"] == effective_date, "lagged_source_date"] = effective_date
    with pytest.raises(ValueError, match="earlier trading date"):
        apply_sentiment_tilt(returns, base, unsafe, sector_map)


def test_fusion_keeps_each_crypto_target_unchanged():
    returns, base, sentiment, sector_map = _fusion_inputs()
    fused = apply_sentiment_tilt(returns, base, sentiment, sector_map)
    base_crypto = base.weights[base.weights["asset"].str.startswith("crypto__")].sort_values(
        ["effective_date", "asset"]
    )
    fused_crypto = fused.weights[fused.weights["asset"].str.startswith("crypto__")].sort_values(
        ["effective_date", "asset"]
    )

    assert fused_crypto["weight"].to_numpy() == pytest.approx(base_crypto["weight"].to_numpy())


def test_fusion_is_long_only_fully_invested_and_conserves_equity_sleeve():
    returns, base, sentiment, sector_map = _fusion_inputs()
    fused = apply_sentiment_tilt(returns, base, sentiment, sector_map)
    totals = fused.weights.groupby("effective_date")["weight"].sum()
    fused_equity = (
        fused.weights[fused.weights["asset"].str.startswith("equity__")]
        .groupby("effective_date")["weight"]
        .sum()
    )
    base_equity = (
        base.weights[base.weights["asset"].str.startswith("equity__")]
        .groupby("effective_date")["weight"]
        .sum()
    )

    assert fused.weights["weight"].ge(0).all()
    assert totals.to_numpy() == pytest.approx(np.ones(len(totals)))
    assert fused_equity.to_numpy() == pytest.approx(base_equity.to_numpy())
    assert fused.metrics["tilt_strength"] == pytest.approx(SENTIMENT_TILT_STRENGTH)


def test_all_missing_sector_signals_leave_base_target_unchanged():
    returns, base, sentiment, sector_map = _fusion_inputs()
    first_effective = base.weights["effective_date"].min()
    sentiment.loc[sentiment["date"] == first_effective, "lagged_sector_sentiment"] = np.nan
    fused = apply_sentiment_tilt(returns, base, sentiment, sector_map)
    base_target = base.weights[base.weights["effective_date"] == first_effective].sort_values(
        "asset"
    )
    fused_target = fused.weights[fused.weights["effective_date"] == first_effective].sort_values(
        "asset"
    )

    assert fused_target["weight"].to_numpy() == pytest.approx(base_target["weight"].to_numpy())
    assert fused_target.loc[
        fused_target["asset"].str.startswith("equity__"), "signal_missing"
    ].all()


def test_fusion_and_base_have_identical_comparison_dates():
    returns, base, sentiment, sector_map = _fusion_inputs()
    fused = apply_sentiment_tilt(returns, base, sentiment, sector_map)

    assert fused.daily.index.equals(base.daily.index)
    assert len(fused.daily) == len(base.daily)
