import numpy as np
import pandas as pd
from src.lm_sentiment import (
    LM_ARCHIVE_DATE,
    LM_SHA256,
    LM_VERSION_DATE,
    LMWordLists,
    headline_model_comparison,
    lm_company_index,
    lm_sector_index,
    methodology_table,
    score_lm_headlines,
    sector_model_comparison,
)


def _word_lists() -> LMWordLists:
    return LMWordLists(
        positive=frozenset({"GAIN", "GROWTH", "PROFITABLE"}),
        negative=frozenset({"LOSS", "DECLINE", "LITIGATION"}),
    )


def test_historical_provenance_is_pre_oos_and_pinned() -> None:
    assert pd.Timestamp(LM_VERSION_DATE) < pd.Timestamp("2021-01-01")
    assert pd.Timestamp(LM_ARCHIVE_DATE) < pd.Timestamp("2021-01-01")
    assert len(LM_SHA256) == 64
    methodology = methodology_table().iloc[0]
    assert methodology["version_date"] == LM_VERSION_DATE
    assert methodology["parameter_selection"].endswith("not used in portfolio fusion")


def test_lm_scoring_preserves_original_text_and_reports_unmatched_coverage() -> None:
    headlines = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2020-01-06", "2020-01-06", "2020-01-06"]),
            "ticker": ["A", "A", "B"],
            "sector": ["S", "S", "S"],
            "title": ["GAIN despite loss", "Ordinary update", "Not-profitable growth"],
        }
    )
    original = headlines["title"].copy()
    scored = score_lm_headlines(headlines, _word_lists())

    pd.testing.assert_series_equal(scored["title"], original, check_names=False)
    assert scored.loc[0, "lm_tone"] == 0.0
    assert scored.loc[0, "lm_matched_token_count"] == 2
    assert scored.loc[1, "lm_tone"] == 0.0
    assert not scored.loc[1, "lm_has_match"]
    assert scored.loc[1, "lm_matched_token_ratio"] == 0.0
    assert scored.loc[2, "lm_positive_count"] == 2


def test_equal_weight_hierarchy_no_news_and_one_trading_day_lag() -> None:
    calendar = pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"])
    universe = pd.DataFrame({"ticker": ["A", "B"], "sector": ["S", "S"]})
    headlines = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(
                ["2020-01-06", "2020-01-06", "2020-01-06", "2020-01-08"]
            ),
            "ticker": ["A", "A", "B", "A"],
            "sector": ["S", "S", "S", "S"],
            "title": ["GAIN", "LOSS", "GAIN", "DECLINE"],
        }
    )
    scored = score_lm_headlines(headlines, _word_lists())
    companies = lm_company_index(scored, calendar, universe)
    sectors = lm_sector_index(companies, calendar, universe)

    day_one = sectors.loc[sectors["date"].eq(calendar[0])].iloc[0]
    assert day_one["lm_sector_sentiment"] == 0.5
    no_news = sectors.loc[sectors["date"].eq(calendar[1])].iloc[0]
    assert np.isnan(no_news["lm_sector_sentiment"])
    assert no_news["headline_count"] == 0
    assert no_news["lagged_source_date"] == calendar[0]
    assert no_news["lagged_lm_sector_sentiment"] == 0.5
    day_three = sectors.loc[sectors["date"].eq(calendar[2])].iloc[0]
    assert np.isnan(day_three["lagged_lm_sector_sentiment"])


def test_comparison_uses_identical_headlines_and_sector_dates() -> None:
    aligned = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "ticker": ["A", "A"],
            "sector": ["S", "S"],
            "title": ["GAIN", "LOSS"],
        }
    )
    lm_scores = score_lm_headlines(aligned, _word_lists())
    vader = aligned.copy()
    vader["vader_compound"] = [-0.2, -0.3]
    comparison = headline_model_comparison(vader, lm_scores)
    assert len(comparison) == len(aligned)
    assert set(comparison["vader_class"]) == {"negative"}
    assert set(comparison["lm_class"]) == {"negative", "positive"}

    vader_sector = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "sector": ["S", "S"],
            "sector_sentiment": [-0.2, -0.3],
        }
    )
    lm_sector = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-06", "2020-01-07"]),
            "sector": ["S", "S"],
            "lm_sector_sentiment": [1.0, -1.0],
        }
    )
    summary = sector_model_comparison(vader_sector, lm_sector).iloc[0]
    assert summary["common_observed_days"] == 2
    assert summary["classification_disagreement_ratio"] == 0.5


def test_comparison_preserves_duplicate_aligned_keys_by_row_order() -> None:
    aligned = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2020-01-06", "2020-01-06"]),
            "ticker": ["A", "A"],
            "sector": ["S", "S"],
            "title": ["Repeated update", "Repeated update"],
        }
    )
    lm_scores = score_lm_headlines(aligned, _word_lists())
    vader = aligned.copy()
    vader["vader_compound"] = [0.1, -0.2]

    comparison = headline_model_comparison(vader, lm_scores)

    assert len(comparison) == 2
    assert comparison["vader_compound"].tolist() == [0.1, -0.2]
