"""Streamlit smoke tests for each FinMosaic workspace view."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit_app import RESULTS_SCHEMA_VERSION, cached_results, sentiment_display_frame

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.mark.parametrize(
    ("view", "heading"),
    [
        ("My Mosaic", "Your selected markets in one historical view"),
        ("Funds", "Compare systematic funds by investable universe"),
        ("Watchlist", "Track selected assets on their native calendars"),
        ("News", "Separate company signals from sector context"),
        ("Methodology", "VADER and financial-dictionary robustness"),
    ],
)
def test_each_workspace_view_renders_without_exception(view: str, heading: str):
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params["view"] = view
    app.run()

    assert not app.exception
    assert heading in [title.value for title in app.title]
    assert any("does not" in markdown.value for markdown in app.markdown)


def test_streamlit_entrypoint_is_results_only():
    source = APP.read_text(encoding="utf-8")

    forbidden = (
        "data_access",
        "load_equity_prices",
        "load_crypto_prices",
        "oos_backtest",
        "run_sentiment_index",
        "run_lm_robustness",
        "apply_sentiment_tilt",
    )
    assert not any(term in source for term in forbidden)
    assert "load_app_data(RESULTS)" in source


def test_versioned_cached_bundle_contains_portfolio_sentiment():
    bundle = cached_results(RESULTS_SCHEMA_VERSION)

    assert "portfolio_sentiment" in bundle.frames
    assert not bundle["portfolio_sentiment"].empty


def test_funds_view_filters_to_crypto_only_funds():
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params["view"] = "Funds"
    app.run()

    app.selectbox(key="selected_universe").select("Crypto-only").run()

    assert not app.exception
    assert app.selectbox(key="selected_fund").options == [
        "Crypto-Only Equal Weight",
        "Crypto-Only Minimum Variance",
        "Crypto-Only Maximum Sharpe",
    ]
    assert any("no equity sleeve" in info.value for info in app.info)


def test_funds_view_renders_saved_portfolio_news_context():
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params["view"] = "Funds"
    app.run()

    assert not app.exception
    assert "Portfolio news context" in [heading.value for heading in app.subheader]
    assert any(
        "one-trading-day-lagged sector VADER" in caption.value for caption in app.caption
    )


def test_my_mosaic_is_default_and_has_two_accessible_composition_charts():
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()

    assert not app.exception
    assert app.radio[0].value == "My Mosaic"
    assert "Your selected markets in one historical view" in [title.value for title in app.title]
    assert len(app.get("plotly_chart")) == 4
    dataframe_columns = [set(frame.value.columns) for frame in app.dataframe]
    assert {"Asset type", "Selected", "Share"} in dataframe_columns
    assert {"holding_group", "weight"} in dataframe_columns
    assert {
        "sector",
        "date",
        "sector_sentiment",
        "headline_count",
        "coverage_ratio",
    } in dataframe_columns


def test_watchlist_preset_updates_only_shared_selection_controls():
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    initial_fund = app.selectbox(key="selected_fund").value

    app.selectbox(key="watchlist_preset").select("Defensive-sector sample")
    app.button[0].click().run()

    assert not app.exception
    assert app.multiselect[0].value == ["KO", "MRK", "SO", "WMT"]
    assert app.multiselect[1].value == []
    assert app.multiselect[2].value == ["Consumer", "Healthcare", "Utilities"]
    assert app.selectbox(key="selected_fund").value == initial_fund


def test_sidebar_prioritises_selected_fund_before_watchlist():
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()

    assert not app.exception
    assert [heading.value for heading in app.sidebar.subheader][:2] == [
        "Selected fund",
        "Watchlist",
    ]
    assert any(
        "Current fund" in block.value and "Combined Equal Weight" in block.value
        for block in app.sidebar.markdown
    )


def test_news_defaults_to_monthly_and_supports_recent_daily_detail():
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params["view"] = "News"
    app.run()

    resolution = app.selectbox(key="news_resolution")
    assert not app.exception
    assert resolution.value == "Monthly"
    assert len(app.get("plotly_chart")) == 4
    assert any(
        {"trading_date", "ticker", "title", "vader_compound", "headline_count"}
        <= set(frame.value.columns)
        for frame in app.dataframe
    )

    resolution.select("Daily: last 30 days").run()
    assert not app.exception
    assert app.selectbox(key="news_resolution").value == "Daily: last 30 days"


def test_monthly_sentiment_keeps_no_news_missing_and_neutral_observed():
    daily = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "date": pd.to_datetime(["2023-01-03", "2023-01-04", "2023-02-01", "2023-02-02"]),
            "vader_company_sentiment": [0.0, float("nan"), float("nan"), float("nan")],
            "headline_count": [1, 0, 0, 0],
            "news_coverage": [1.0, 0.0, 0.0, 0.0],
        }
    )

    monthly = sentiment_display_frame(
        daily,
        group_column="ticker",
        score_column="vader_company_sentiment",
        coverage_column="news_coverage",
        resolution="Monthly",
        end_date=pd.Timestamp("2023-02-28"),
    )

    january_score = monthly.loc[
        monthly["date"].eq(pd.Timestamp("2023-01-01")), "vader_company_sentiment"
    ].item()
    assert january_score == 0
    assert pd.isna(
        monthly.loc[
            monthly["date"].eq(pd.Timestamp("2023-02-01")), "vader_company_sentiment"
        ].item()
    )
