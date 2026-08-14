"""Schema and artifact tests for the CSV-only presentation layer."""

import pathlib

import pandas as pd
from src.presentation import (
    EXPECTED_FUNDS,
    FUND_SPECS,
    build_latest_holdings,
    build_presentation_outputs,
    build_weight_group_history,
)


def _write_artifacts(root: pathlib.Path) -> None:
    data_dir = root / "results" / "data"
    table_dir = root / "results" / "tables"
    data_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)
    dates = pd.to_datetime(["2022-02-02", "2022-02-03"])

    return_rows = []
    weight_rows = []
    metric_rows = []
    for position, (fund_id, (fund, universe, method)) in enumerate(FUND_SPECS.items()):
        returns = [0.01 + position * 0.001, -0.002]
        growth = 1.0
        peak = 1.0
        for date, daily_return in zip(dates, returns, strict=True):
            growth *= 1 + daily_return
            peak = max(peak, growth)
            return_rows.append(
                {
                    "date": date,
                    "fund": fund,
                    "fund_id": fund_id,
                    "universe": universe,
                    "method": method,
                    "portfolio_return": daily_return,
                    "growth_of_1": growth,
                    "drawdown": growth / peak - 1,
                }
            )
        if universe == "Combined":
            assets = (("equity__AAA", "equity", "Tech"), ("crypto__BTC-USD", "crypto", None))
        elif universe == "Equity-only":
            assets = (("equity__AAA", "equity", "Tech"), ("equity__BBB", "equity", "Bank"))
        else:
            assets = (
                ("crypto__BTC-USD", "crypto", None),
                ("crypto__ETH-USD", "crypto", None),
            )
        for effective_date in pd.to_datetime(["2022-02-02", "2022-03-01"]):
            for asset, asset_class, sector in assets:
                weight_rows.append(
                    {
                        "method": method,
                        "fund": fund,
                        "fund_id": fund_id,
                        "universe": universe,
                        "rebalance_date": effective_date - pd.Timedelta(days=1),
                        "effective_date": effective_date,
                        "estimation_start": pd.Timestamp("2021-01-01"),
                        "estimation_end": effective_date - pd.Timedelta(days=2),
                        "asset": asset,
                        "weight": 0.5,
                        "asset_class": asset_class,
                        "sector": sector,
                    }
                )
        metric_rows.append(
            {
                "fund": fund,
                "fund_id": fund_id,
                "universe": universe,
                "method": method,
                "annualized_return": 0.10 + position * 0.01,
                "annualized_volatility": 0.20,
                "sharpe_ratio": 0.50 + position * 0.05,
                "maximum_drawdown": -0.10,
                "growth_of_1": return_rows[-1]["growth_of_1"],
                "observations": 2,
                "start_date": dates[0],
                "end_date": dates[-1],
                "estimation_window": 365 if universe == "Crypto-only" else 252,
                "rebalance_frequency": "monthly",
                "rebalance_convention": "first trading day; effective next trading day",
                "constraints": "long-only; fully invested",
                "transaction_costs": 0.0,
                "risk_free_rate": 0.0,
                "tilt_strength": 0.5 if method == "sentiment_tilt" else None,
                "missing_signal_policy": ("multiplier one" if method == "sentiment_tilt" else None),
            }
        )
    pd.DataFrame(return_rows).to_csv(data_dir / "fund_returns.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(data_dir / "fund_weights.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(table_dir / "performance_metrics.csv", index=False)


def test_latest_holdings_and_weight_history_have_report_ready_schemas(tmp_path):
    _write_artifacts(tmp_path)
    weights = pd.read_csv(
        tmp_path / "results" / "data" / "fund_weights.csv",
        parse_dates=["effective_date"],
    )
    holdings = build_latest_holdings(weights)
    history = build_weight_group_history(weights)

    assert set(holdings["fund_id"]) == EXPECTED_FUNDS
    assert set(holdings.columns) == {
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
    }
    assert holdings.groupby("fund_id")["weight"].sum().eq(1.0).all()
    assert set(history["allocation_group"]) == {"Bank", "Crypto", "Tech"}
    assert history.groupby(["fund_id", "effective_date"])["target_weight"].sum().eq(1.0).all()


def test_presentation_builder_writes_tables_and_all_fact_sheets(tmp_path):
    _write_artifacts(tmp_path)
    paths = build_presentation_outputs(tmp_path)

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert {
        "growth",
        "drawdown",
        "weights",
        "sharpe",
        "fact_sheet_summary",
        "fund_comparison",
        "latest_holdings",
        *(f"fact_sheet_{fund_id}" for fund_id in EXPECTED_FUNDS),
    }.issubset(paths)
    comparison = pd.read_csv(paths["fund_comparison"])
    assert {
        "annualized_return_pct",
        "annualized_volatility_pct",
        "maximum_drawdown_pct",
        "growth_of_1_usd",
    }.issubset(comparison.columns)
