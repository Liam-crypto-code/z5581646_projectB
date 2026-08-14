"""Reproduce Part B results. Run from the project root:

python scripts/run_part_b.py
"""

import pathlib
import sys

import nltk
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.app_data import save_asset_performance_outputs
from src.etl import run_station1
from src.features import run_station2
from src.figures import (
    plot_company_news_coverage,
    plot_company_vader_sentiment_heatmap,
    plot_fusion_comparison,
    plot_lm_coverage,
    plot_monthly_company_watchlist_sentiment,
    plot_sector_sentiment_index,
    plot_sector_sentiment_model_comparison,
    plot_sentiment_coverage,
    plot_sentiment_model_distributions,
)
from src.fund_universes import run_standalone_fund_universes
from src.fusion import SENTIMENT_TILT_STRENGTH, FusionResult, apply_sentiment_tilt
from src.lm_sentiment import LMRobustnessResults, run_lm_robustness
from src.portfolio_sentiment import save_portfolio_sentiment_output
from src.portfolios import BacktestResult, oos_backtest
from src.presentation import build_presentation_outputs
from src.sentiment import (
    SentimentResults,
    company_sentiment_coverage_by_sector,
    company_sentiment_index,
    company_sentiment_summary_by_ticker,
    run_sentiment_index,
    sentiment_summary_by_sector,
    vader_distribution_summary,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULT_DATA = ROOT / "results" / "data"
RESULT_FIGURES = ROOT / "results" / "figures"
RESULT_TABLES = ROOT / "results" / "tables"

FUND_NAMES = {
    "equal_weight": "Combined Equal Weight",
    "min_variance": "Combined Minimum Variance",
    "max_sharpe": "Combined Maximum Sharpe",
    "sentiment_tilt": "Sentiment-Tilted Equal Weight",
}
PORTFOLIO_METHODS = ("equal_weight", "min_variance", "max_sharpe")

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


def _save_portfolio_outputs(results: dict[str, object]) -> None:
    """Save small portfolio-only artifacts for later reporting and app use."""
    daily_frames = []
    weight_frames = []
    metric_rows = []
    for fund_id, result in results.items():
        fund, universe, method = FUND_SPECS[fund_id]
        daily = result.daily.reset_index().copy()
        daily["method"] = method
        daily.insert(1, "fund", fund)
        daily.insert(2, "fund_id", fund_id)
        daily.insert(3, "universe", universe)
        daily_frames.append(daily)

        weights = result.weights.copy()
        weights["method"] = method
        weights.insert(1, "fund", fund)
        weights.insert(2, "fund_id", fund_id)
        weights.insert(3, "universe", universe)
        weights["asset_class"] = weights["asset"].str.split("__", n=1).str[0]
        weight_frames.append(weights)

        metrics = dict(result.metrics)
        metrics["method"] = method
        metrics["fund"] = fund
        metrics["fund_id"] = fund_id
        metrics["universe"] = universe
        metric_rows.append(metrics)

    RESULT_DATA.mkdir(parents=True, exist_ok=True)
    RESULT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.concat(daily_frames, ignore_index=True).to_csv(RESULT_DATA / "fund_returns.csv", index=False)
    pd.concat(weight_frames, ignore_index=True).to_csv(
        RESULT_DATA / "fund_weights.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(RESULT_TABLES / "performance_metrics.csv", index=False)


def _ensure_vader_lexicon() -> None:
    """Install VADER's small lexicon for this offline build, never in the app."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        if not nltk.download("vader_lexicon", quiet=True):
            raise RuntimeError("could not download the VADER lexicon")


def _save_sentiment_outputs(sentiment_results: SentimentResults) -> None:
    """Save the standalone sector index, audit tables, and report figures."""
    sector_index = sentiment_results.sector_index
    sector_summary = sentiment_summary_by_sector(sector_index)
    vader_summary = vader_distribution_summary(sentiment_results.headline_scores)
    methodology = pd.DataFrame(
        [
            {
                "model": "NLTK VADER",
                "text_unit": "original headline title",
                "ticker_day_aggregation": "equal-weight mean of headline compound scores",
                "sector_day_aggregation": "equal-weight mean of observed ticker-day scores",
                "no_news_policy": "missing sentiment; zero coverage; no neutral imputation",
                "signal_lag": "one equity trading day",
                "coverage_measure": "observed tickers divided by sector-universe tickers",
            }
        ]
    )

    RESULT_DATA.mkdir(parents=True, exist_ok=True)
    RESULT_TABLES.mkdir(parents=True, exist_ok=True)
    RESULT_FIGURES.mkdir(parents=True, exist_ok=True)
    sector_index.to_csv(RESULT_DATA / "sector_sentiment_index.csv", index=False)
    sector_summary.to_csv(RESULT_TABLES / "sentiment_summary_by_sector.csv", index=False)
    vader_summary.to_csv(RESULT_TABLES / "vader_distribution_summary.csv", index=False)
    methodology.to_csv(RESULT_TABLES / "sentiment_methodology.csv", index=False)
    plot_sector_sentiment_index(sector_index, RESULT_FIGURES / "sector_sentiment_index.png")
    plot_sentiment_coverage(sector_summary, RESULT_FIGURES / "sentiment_coverage_by_sector.png")


def _save_company_sentiment_outputs(company_index: pd.DataFrame) -> None:
    """Save the standalone company-level VADER layer and report exhibits."""
    ticker_summary = company_sentiment_summary_by_ticker(company_index)
    sector_coverage = company_sentiment_coverage_by_sector(company_index)
    methodology = pd.DataFrame(
        [
            {
                "model": "NLTK VADER",
                "level": "company ticker-day",
                "universe": "50 equities by complete equity trading calendar",
                "same_day_score": "equal-weight mean of original-headline VADER compound scores",
                "no_news_policy": "missing sentiment; news_observed false; news_coverage zero",
                "neutral_policy": "observed score zero retained with news_observed true",
                "signal_lag": "exactly one equity trading day",
                "lag_missing_policy": (
                    "no signal when the immediately preceding trading day has no news"
                ),
                "downstream_use": "standalone analytic only; existing sector fusion unchanged",
            }
        ]
    )
    company_index.to_csv(RESULT_DATA / "company_sentiment_index.csv", index=False)
    ticker_summary.to_csv(RESULT_TABLES / "company_sentiment_summary_by_ticker.csv", index=False)
    sector_coverage.to_csv(RESULT_TABLES / "company_sentiment_coverage_by_sector.csv", index=False)
    methodology.to_csv(RESULT_TABLES / "company_sentiment_methodology.csv", index=False)
    plot_company_vader_sentiment_heatmap(
        company_index,
        RESULT_FIGURES / "company_vader_sentiment_heatmap.png",
    )
    plot_company_news_coverage(
        ticker_summary,
        RESULT_FIGURES / "company_news_coverage_by_ticker.png",
    )
    plot_monthly_company_watchlist_sentiment(
        company_index,
        RESULT_FIGURES / "monthly_company_sentiment_watchlist.png",
    )


def _save_fusion_outputs(base_result: BacktestResult, fusion_result: FusionResult) -> None:
    """Save the fixed rule and like-for-like before-versus-after comparison."""
    metric_names = (
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "growth_of_1",
    )
    comparison = pd.DataFrame(
        [
            {
                "metric": metric,
                "combined_equal_weight": base_result.metrics[metric],
                "sentiment_tilted_equal_weight": fusion_result.metrics[metric],
                "tilted_minus_base": fusion_result.metrics[metric] - base_result.metrics[metric],
            }
            for metric in metric_names
        ]
    )
    rule = pd.DataFrame(
        [
            {
                "base_fund": "Combined Equal Weight",
                "tilt_strength": SENTIMENT_TILT_STRENGTH,
                "equation": (
                    "raw equity weight = base weight * (1 + 0.50 * lagged sector sentiment); "
                    "renormalize equities to base equity sleeve"
                ),
                "signal": "one-equity-trading-day-lagged VADER sector sentiment",
                "missing_signal_policy": "multiplier one; stored signal remains missing",
                "crypto_policy": "copy every base crypto target unchanged",
                "rebalance_frequency": "monthly; same dates as base fund",
                "transaction_costs": 0.0,
                "parameter_selection": "fixed ex ante; no out-of-sample tuning",
            }
        ]
    )
    comparison.to_csv(RESULT_TABLES / "fusion_comparison.csv", index=False)
    rule.to_csv(RESULT_TABLES / "fusion_methodology.csv", index=False)
    plot_fusion_comparison(
        base_result.daily,
        fusion_result.daily,
        RESULT_FIGURES / "fusion_base_vs_tilt.png",
    )


def _save_lm_robustness_outputs(results: LMRobustnessResults) -> None:
    """Save separate historical-dictionary indexes and comparison exhibits."""
    results.company_index.to_csv(RESULT_DATA / "lm_company_sentiment_index.csv", index=False)
    results.sector_index.to_csv(RESULT_DATA / "lm_sector_sentiment_index.csv", index=False)
    results.distribution_summary.to_csv(
        RESULT_TABLES / "sentiment_score_distribution_comparison.csv", index=False
    )
    results.sector_comparison.to_csv(
        RESULT_TABLES / "sentiment_model_sector_comparison.csv", index=False
    )
    results.coverage_summary.to_csv(RESULT_TABLES / "lm_coverage_by_sector.csv", index=False)
    results.disagreement_examples.to_csv(
        RESULT_TABLES / "sentiment_disagreement_examples.csv", index=False
    )
    results.methodology.to_csv(RESULT_TABLES / "lm_sentiment_methodology.csv", index=False)
    plot_sentiment_model_distributions(
        results.headline_comparison,
        RESULT_FIGURES / "sentiment_score_distributions_vader_vs_lm.png",
    )
    plot_sector_sentiment_model_comparison(
        results.sector_comparison,
        RESULT_FIGURES / "sentiment_model_sector_comparison.png",
    )
    plot_lm_coverage(results.coverage_summary, RESULT_FIGURES / "lm_sentiment_coverage.png")


def main() -> None:
    """Build the foundation and initial combined-fund backtests."""
    station1 = run_station1()
    station2 = run_station2(station1.equities, station1.crypto, station1.headlines)
    save_asset_performance_outputs(
        station2.equity_returns,
        station2.crypto_returns,
        ROOT / "results",
    )
    print(
        "clean rows:",
        f"equities={len(station1.equities):,}",
        f"crypto={len(station1.crypto):,}",
        f"headlines={len(station1.headlines):,}",
    )
    print(
        "derived foundation:",
        f"combined_dates={len(station2.combined_returns):,}",
        f"headline_panel_rows={len(station2.headline_panel):,}",
    )
    combined_results = {
        method: oos_backtest(
            station2.combined_returns,
            method=method,
            estimation_window=252,
            periods_per_year=252,
        )
        for method in PORTFOLIO_METHODS
    }
    standalone_results = run_standalone_fund_universes(
        station2.equity_returns,
        station2.crypto_returns,
    )
    for method, result in combined_results.items():
        print(
            FUND_NAMES[method] + ":",
            f"live={result.metrics['start_date'].date()} to {result.metrics['end_date'].date()}",
            f"rebalances={result.weights['rebalance_date'].nunique()}",
            f"growth=${result.metrics['growth_of_1']:.3f}",
        )

    _ensure_vader_lexicon()
    sector_universe = station1.equities[["ticker", "sector"]].drop_duplicates()
    sentiment_results = run_sentiment_index(
        station2.headline_alignment,
        station1.equities["date"],
        sector_universe,
    )
    _save_sentiment_outputs(sentiment_results)
    company_index = company_sentiment_index(
        sentiment_results.ticker_day_scores,
        station1.equities["date"],
        sector_universe,
    )
    _save_company_sentiment_outputs(company_index)
    observed_sector_days = sentiment_results.sector_index["sector_sentiment"].notna().sum()
    print(
        "Standalone sector sentiment:",
        f"headlines={len(sentiment_results.headline_scores):,}",
        f"ticker_days={len(sentiment_results.ticker_day_scores):,}",
        f"observed_sector_days={observed_sector_days:,}",
        f"calendar_rows={len(sentiment_results.sector_index):,}",
    )
    print(
        "Company VADER sentiment:",
        f"calendar_rows={len(company_index):,}",
        f"observed_company_days={company_index['news_observed'].sum():,}",
        f"coverage={company_index['news_coverage'].mean():.1%}",
    )

    lm_results = run_lm_robustness(
        station2.headline_alignment,
        station1.equities["date"],
        sector_universe,
        sentiment_results.headline_scores,
        sentiment_results.sector_index,
        ROOT / "data" / "reference" / "LoughranMcDonald_SentimentWordLists_2018.xlsx",
    )
    _save_lm_robustness_outputs(lm_results)
    print(
        "Loughran-McDonald robustness:",
        f"company_rows={len(lm_results.company_index):,}",
        f"sector_rows={len(lm_results.sector_index):,}",
        f"matched_headlines={lm_results.headline_comparison['lm_has_match'].mean():.1%}",
        f"mean_sector_correlation={lm_results.sector_comparison['pearson_correlation'].mean():.3f}",
    )

    fusion_result = apply_sentiment_tilt(
        station2.combined_returns,
        combined_results["equal_weight"],
        sentiment_results.sector_index,
        sector_universe,
        periods_per_year=252,
    )
    all_funds = {
        **{f"combined_{method}": result for method, result in combined_results.items()},
        "combined_sentiment_tilt": fusion_result,
        **standalone_results,
    }
    _save_portfolio_outputs(all_funds)
    save_portfolio_sentiment_output(ROOT / "results")
    _save_fusion_outputs(combined_results["equal_weight"], fusion_result)
    sharpe_change = (
        fusion_result.metrics["sharpe_ratio"]
        - combined_results["equal_weight"].metrics["sharpe_ratio"]
    )
    print(
        "Sentiment-Tilted Equal Weight:",
        f"live={fusion_result.metrics['start_date'].date()} "
        f"to {fusion_result.metrics['end_date'].date()}",
        f"growth=${fusion_result.metrics['growth_of_1']:.3f}",
        f"Sharpe change={sharpe_change:+.3f}",
    )
    presentation_paths = build_presentation_outputs(ROOT)
    print(f"Presentation outputs: {len(presentation_paths)} files built from saved CSVs")


if __name__ == "__main__":
    main()
