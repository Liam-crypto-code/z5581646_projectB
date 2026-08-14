"""Calendar and constraint tests for standalone equity and crypto funds."""

import numpy as np
import pandas as pd
import pytest
from src.fund_universes import build_native_universe_panels, run_standalone_fund_universes


def _long_returns(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    asset_scale: float,
) -> pd.DataFrame:
    time = np.arange(len(dates), dtype=float)
    rows = []
    for position, ticker in enumerate(tickers):
        values = (
            0.0002
            + position * 0.000005
            + np.sin(time / (8.0 + position)) * asset_scale
            + np.cos(time / 17.0) * asset_scale * 0.3
        )
        rows.extend(
            {"date": date, "ticker": ticker, "return": value}
            for date, value in zip(dates, values, strict=True)
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def standalone_results():
    equity_dates = pd.bdate_range("2020-01-02", periods=290)
    crypto_dates = pd.date_range("2020-01-01", periods=405, freq="D")
    equities = _long_returns(
        equity_dates,
        [f"E{number:02d}" for number in range(50)],
        0.0008,
    )
    crypto = _long_returns(
        crypto_dates,
        [f"C{number:02d}-USD" for number in range(10)],
        0.0015,
    )
    return run_standalone_fund_universes(equities, crypto)


def test_native_panels_keep_weekends_out_of_equities_and_in_crypto():
    equity_dates = pd.bdate_range("2020-01-02", periods=10)
    crypto_dates = pd.date_range("2020-01-01", periods=14, freq="D")
    equities = _long_returns(equity_dates, ["AAA", "BBB"], 0.001)
    crypto = _long_returns(crypto_dates, ["BTC-USD", "ETH-USD"], 0.002)

    equity_panel, crypto_panel = build_native_universe_panels(equities, crypto)

    assert not pd.to_datetime(equity_panel["date"]).dt.dayofweek.ge(5).any()
    assert pd.to_datetime(crypto_panel["date"]).dt.dayofweek.ge(5).any()


def test_standalone_funds_use_required_windows_and_annualization(standalone_results):
    assert set(standalone_results) == {
        "equity_equal_weight",
        "equity_min_variance",
        "equity_max_sharpe",
        "crypto_equal_weight",
        "crypto_min_variance",
        "crypto_max_sharpe",
    }
    for fund_id, result in standalone_results.items():
        if fund_id.startswith("equity_"):
            assert result.estimation_window == 252
            assert result.metrics["periods_per_year"] == 252
            assert not result.daily.index.dayofweek.isin([5, 6]).any()
        else:
            assert result.estimation_window == 365
            assert result.metrics["periods_per_year"] == 365
            assert result.daily.index.dayofweek.isin([5, 6]).any()


def test_standalone_rebalance_timing_and_weight_sums(standalone_results):
    for result in standalone_results.values():
        grouped = result.weights.groupby("rebalance_date", observed=True)
        assert (grouped["estimation_end"].max() < grouped["effective_date"].min()).all()
        assert (grouped["rebalance_date"].first() < grouped["effective_date"].min()).all()
        assert grouped["weight"].sum().to_numpy() == pytest.approx(
            np.ones(grouped.ngroups), abs=1e-8
        )
        assert result.weights["weight"].ge(-1e-10).all()
