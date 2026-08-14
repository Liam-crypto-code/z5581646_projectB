"""Historical Loughran-McDonald sentiment robustness analytics.

This module is deliberately separate from the canonical VADER index and fusion
path. It uses the official 2018 sentiment word-list workbook, which the
University of Notre Dame page archived on 12 November 2020 described as updated
in March 2019.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

LM_VERSION = "Loughran-McDonald Sentiment Word Lists 2018"
LM_VERSION_DATE = "2019-03"
LM_ARCHIVE_DATE = "2020-11-12"
LM_SHA256 = "ab6e789d8f1441622092e34c96439b29a2b49b47844aef30ee4ddb5229d37882"
LM_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=15UPaF2xJLSVz8DYuphierz67trCxFLcl&export=download&confirm=t"
)
LM_OFFICIAL_ARCHIVE_URL = (
    "https://web.archive.org/web/20201112032520id_/"
    "https://sraf.nd.edu/textual-analysis/resources/"
)
LM_PAPER_DOI = "10.1111/j.1540-6261.2010.01625.x"
LM_FORMULA = "(positive token count - negative token count) / alphabetic token count"

TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}")
XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass(frozen=True)
class LMWordLists:
    """Positive and negative unigrams from the pinned historical workbook."""

    positive: frozenset[str]
    negative: frozenset[str]


@dataclass
class LMRobustnessResults:
    """Company, sector, and model-comparison outputs."""

    headline_scores: pd.DataFrame
    company_index: pd.DataFrame
    sector_index: pd.DataFrame
    headline_comparison: pd.DataFrame
    distribution_summary: pd.DataFrame
    sector_comparison: pd.DataFrame
    coverage_summary: pd.DataFrame
    disagreement_examples: pd.DataFrame
    methodology: pd.DataFrame


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_lm_wordlists(cache_path: pathlib.Path) -> pathlib.Path:
    """Return a checksum-verified local copy of the historical workbook."""
    override = os.environ.get("FINS_LM_WORDLISTS")
    path = pathlib.Path(override).expanduser() if override else cache_path
    if path.exists():
        actual = _sha256(path)
        if actual != LM_SHA256:
            raise RuntimeError(
                f"Loughran-McDonald workbook checksum mismatch: expected {LM_SHA256}, "
                f"received {actual}"
            )
        return path

    if override:
        raise FileNotFoundError(f"FINS_LM_WORDLISTS does not exist: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(LM_DOWNLOAD_URL, timeout=120)
    response.raise_for_status()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(response.content)
    actual = _sha256(temporary)
    if actual != LM_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "official historical Loughran-McDonald download did not match the pinned checksum"
        )
    temporary.replace(path)
    return path


def _shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
    return ["".join(item.itertext()) for item in root.findall("m:si", XLSX_NS)]


def _sheet_paths(workbook: zipfile.ZipFile) -> dict[str, str]:
    book = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall("r:Relationship", REL_NS)
    }
    paths: dict[str, str] = {}
    for sheet in book.findall("m:sheets/m:sheet", XLSX_NS):
        relation_id = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
        paths[sheet.attrib["name"]] = "xl/" + targets[relation_id]
    return paths


def _read_word_sheet(
    workbook: zipfile.ZipFile,
    path: str,
    shared_strings: list[str],
) -> frozenset[str]:
    root = ElementTree.fromstring(workbook.read(path))
    values = []
    for cell in root.findall(".//m:c", XLSX_NS):
        if not cell.attrib.get("r", "").startswith("A") or cell.attrib.get("t") != "s":
            continue
        value = cell.find("m:v", XLSX_NS)
        if value is not None:
            token = shared_strings[int(value.text)].upper()
            if TOKEN_PATTERN.fullmatch(token):
                values.append(token)
    return frozenset(values)


def load_lm_wordlists(path: pathlib.Path) -> LMWordLists:
    """Parse positive and negative sheets without adding an Excel dependency."""
    if _sha256(path) != LM_SHA256:
        raise RuntimeError("refusing to parse an unpinned Loughran-McDonald workbook")
    with zipfile.ZipFile(path) as workbook:
        strings = _shared_strings(workbook)
        paths = _sheet_paths(workbook)
        positive = _read_word_sheet(workbook, paths["Positive"], strings)
        negative = _read_word_sheet(workbook, paths["Negative"], strings)
    if len(positive) != 354 or len(negative) != 2355:
        raise RuntimeError("historical Loughran-McDonald word-list counts are unexpected")
    if positive.intersection(negative):
        raise RuntimeError("positive and negative historical word lists overlap")
    return LMWordLists(positive=positive, negative=negative)


def _score_title(title: str, word_lists: LMWordLists) -> dict[str, object]:
    tokens = [token.upper() for token in TOKEN_PATTERN.findall(title)]
    positive_terms = [token for token in tokens if token in word_lists.positive]
    negative_terms = [token for token in tokens if token in word_lists.negative]
    matched = len(positive_terms) + len(negative_terms)
    token_count = len(tokens)
    tone = (len(positive_terms) - len(negative_terms)) / token_count if token_count else 0.0
    return {
        "lm_tone": tone,
        "lm_positive_count": len(positive_terms),
        "lm_negative_count": len(negative_terms),
        "lm_matched_token_count": matched,
        "lm_token_count": token_count,
        "lm_matched_token_ratio": matched / token_count if token_count else 0.0,
        "lm_has_match": matched > 0,
        "lm_positive_terms": "|".join(positive_terms),
        "lm_negative_terms": "|".join(negative_terms),
    }


def score_lm_headlines(
    aligned_headlines: pd.DataFrame,
    word_lists: LMWordLists,
) -> pd.DataFrame:
    """Score unchanged original titles with the fixed historical dictionary."""
    required = {"trading_date", "ticker", "sector", "title"}
    missing = required.difference(aligned_headlines.columns)
    if missing:
        raise ValueError(f"aligned_headlines is missing required columns: {sorted(missing)}")
    usable = aligned_headlines.dropna(subset=list(required)).copy()
    usable["trading_date"] = pd.to_datetime(usable["trading_date"])
    scores = pd.DataFrame(
        usable["title"].map(lambda title: _score_title(str(title), word_lists)).tolist(),
        index=usable.index,
    )
    return pd.concat([usable, scores], axis=1).reset_index(drop=True)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0)))


def lm_company_index(
    headline_scores: pd.DataFrame,
    trading_calendar: pd.Series | pd.DatetimeIndex,
    sector_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Build a complete ticker-calendar index and one-trading-day lag."""
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar).unique()).sort_values()
    universe = sector_universe[["ticker", "sector"]].drop_duplicates().copy()
    if universe["ticker"].duplicated().any():
        raise ValueError("each ticker must map to exactly one sector")

    observed = (
        headline_scores.groupby(
            ["trading_date", "ticker", "sector"], observed=True, as_index=False
        )
        .agg(
            company_sentiment=("lm_tone", "mean"),
            headline_count=("lm_tone", "size"),
            matched_headline_count=("lm_has_match", "sum"),
            matched_token_count=("lm_matched_token_count", "sum"),
            token_count=("lm_token_count", "sum"),
        )
        .rename(columns={"trading_date": "date"})
    )
    grid = pd.MultiIndex.from_product(
        [calendar, universe["ticker"]], names=["date", "ticker"]
    ).to_frame(index=False)
    index = grid.merge(universe, on="ticker", how="left", validate="many_to_one")
    index = index.merge(observed, on=["date", "ticker", "sector"], how="left")
    count_columns = (
        "headline_count",
        "matched_headline_count",
        "matched_token_count",
        "token_count",
    )
    for column in count_columns:
        index[column] = index[column].fillna(0).astype(int)
    index["matched_headline_ratio"] = _ratio(
        index["matched_headline_count"], index["headline_count"]
    )
    index["matched_token_ratio"] = _ratio(index["matched_token_count"], index["token_count"])
    index = index.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = index.groupby("ticker", observed=True, sort=False)
    index["lagged_source_date"] = grouped["date"].shift(1)
    index["lagged_company_sentiment"] = grouped["company_sentiment"].shift(1)
    index["lagged_headline_count"] = grouped["headline_count"].shift(1).astype("Int64")
    index["lagged_matched_token_ratio"] = grouped["matched_token_ratio"].shift(1)
    index["signal_available"] = index["lagged_company_sentiment"].notna()
    return index.sort_values(["date", "sector", "ticker"]).reset_index(drop=True)


def lm_sector_index(
    company_index: pd.DataFrame,
    trading_calendar: pd.Series | pd.DatetimeIndex,
    sector_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Equal-weight observed company-day scores within each sector."""
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar).unique()).sort_values()
    universe = sector_universe[["ticker", "sector"]].drop_duplicates()
    universe_counts = universe.groupby("sector", observed=True)["ticker"].nunique()
    observed_companies = company_index[company_index["company_sentiment"].notna()]
    observed = (
        observed_companies.groupby(["date", "sector"], observed=True, as_index=False)
        .agg(
            lm_sector_sentiment=("company_sentiment", "mean"),
            ticker_coverage=("ticker", "nunique"),
            headline_count=("headline_count", "sum"),
            matched_headline_count=("matched_headline_count", "sum"),
            matched_token_count=("matched_token_count", "sum"),
            token_count=("token_count", "sum"),
        )
    )
    grid = pd.MultiIndex.from_product(
        [calendar, sorted(universe_counts.index)], names=["date", "sector"]
    ).to_frame(index=False)
    index = grid.merge(observed, on=["date", "sector"], how="left", validate="one_to_one")
    index["universe_tickers"] = index["sector"].map(universe_counts).astype(int)
    count_columns = (
        "ticker_coverage",
        "headline_count",
        "matched_headline_count",
        "matched_token_count",
        "token_count",
    )
    for column in count_columns:
        index[column] = index[column].fillna(0).astype(int)
    index["coverage_ratio"] = index["ticker_coverage"] / index["universe_tickers"]
    index["matched_headline_ratio"] = _ratio(
        index["matched_headline_count"], index["headline_count"]
    )
    index["matched_token_ratio"] = _ratio(index["matched_token_count"], index["token_count"])
    index = index.sort_values(["sector", "date"]).reset_index(drop=True)
    grouped = index.groupby("sector", observed=True, sort=False)
    index["lagged_source_date"] = grouped["date"].shift(1)
    index["lagged_lm_sector_sentiment"] = grouped["lm_sector_sentiment"].shift(1)
    index["lagged_ticker_coverage"] = grouped["ticker_coverage"].shift(1).astype("Int64")
    index["lagged_matched_token_ratio"] = grouped["matched_token_ratio"].shift(1)
    index["signal_available"] = index["lagged_lm_sector_sentiment"].notna()
    return index.sort_values(["date", "sector"]).reset_index(drop=True)


def _classification(values: pd.Series, model: str) -> pd.Series:
    if model == "VADER":
        return pd.Series(
            np.select(
                [values <= -0.05, values >= 0.05],
                ["negative", "positive"],
                default="neutral",
            ),
            index=values.index,
        )
    return pd.Series(
        np.select([values < 0, values > 0], ["negative", "positive"], default="neutral"),
        index=values.index,
    )


def headline_model_comparison(
    vader_scores: pd.DataFrame,
    lm_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Align the two headline scores and preserve interpretable LM match details."""
    keys = ["trading_date", "ticker", "sector", "title"]
    lm_columns = [
        "lm_tone",
        "lm_positive_count",
        "lm_negative_count",
        "lm_matched_token_count",
        "lm_token_count",
        "lm_matched_token_ratio",
        "lm_has_match",
        "lm_positive_terms",
        "lm_negative_terms",
    ]
    vader = vader_scores.reset_index(drop=True)
    lm = lm_scores.reset_index(drop=True)
    if len(vader) != len(lm) or not vader[keys].equals(lm[keys]):
        raise ValueError("VADER and LM headline samples do not match exactly")
    comparison = pd.concat(
        [vader[[*keys, "vader_compound"]], lm[lm_columns]], axis=1
    )
    comparison["vader_class"] = _classification(comparison["vader_compound"], "VADER")
    comparison["lm_class"] = _classification(comparison["lm_tone"], "LM")
    comparison["score_difference"] = comparison["vader_compound"] - comparison["lm_tone"]
    comparison["absolute_score_difference"] = comparison["score_difference"].abs()
    return comparison


def distribution_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarise headline-score distributions under fixed model-native thresholds."""
    rows = []
    for model, score_column, class_column, threshold in (
        ("VADER", "vader_compound", "vader_class", "negative <= -0.05; positive >= 0.05"),
        ("Loughran-McDonald 2018", "lm_tone", "lm_class", "negative < 0; positive > 0"),
    ):
        scores = comparison[score_column]
        counts = comparison[class_column].value_counts()
        rows.append(
            {
                "model": model,
                "headline_count": len(scores),
                "mean_score": scores.mean(),
                "score_std": scores.std(),
                "minimum": scores.min(),
                "p05": scores.quantile(0.05),
                "median": scores.median(),
                "p95": scores.quantile(0.95),
                "maximum": scores.max(),
                "negative_ratio": counts.get("negative", 0) / len(scores),
                "neutral_ratio": counts.get("neutral", 0) / len(scores),
                "positive_ratio": counts.get("positive", 0) / len(scores),
                "classification_rule": threshold,
            }
        )
    return pd.DataFrame(rows)


def sector_model_comparison(
    vader_sector_index: pd.DataFrame,
    lm_index: pd.DataFrame,
) -> pd.DataFrame:
    """Compare contemporaneous daily sector scores on identical observed dates."""
    joined = vader_sector_index[["date", "sector", "sector_sentiment"]].merge(
        lm_index[["date", "sector", "lm_sector_sentiment"]],
        on=["date", "sector"],
        validate="one_to_one",
    )
    joined = joined.dropna(subset=["sector_sentiment", "lm_sector_sentiment"])
    joined["vader_class"] = _classification(joined["sector_sentiment"], "VADER")
    joined["lm_class"] = _classification(joined["lm_sector_sentiment"], "LM")
    rows = []
    for sector, frame in joined.groupby("sector", observed=True):
        rows.append(
            {
                "sector": sector,
                "common_observed_days": len(frame),
                "pearson_correlation": frame["sector_sentiment"].corr(
                    frame["lm_sector_sentiment"], method="pearson"
                ),
                "spearman_correlation": frame["sector_sentiment"].corr(
                    frame["lm_sector_sentiment"], method="spearman"
                ),
                "classification_agreement_ratio": frame["vader_class"].eq(
                    frame["lm_class"]
                ).mean(),
                "classification_disagreement_ratio": frame["vader_class"].ne(
                    frame["lm_class"]
                ).mean(),
                "opposite_sign_ratio": (
                    frame["sector_sentiment"] * frame["lm_sector_sentiment"] < 0
                ).mean(),
                "mean_absolute_score_difference": (
                    frame["sector_sentiment"] - frame["lm_sector_sentiment"]
                ).abs().mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("sector").reset_index(drop=True)


def coverage_summary(
    comparison: pd.DataFrame,
    company_index: pd.DataFrame,
    sector_index: pd.DataFrame,
) -> pd.DataFrame:
    """Report news availability and positive/negative token coverage by sector."""
    rows = []
    for sector, headlines in comparison.groupby("sector", observed=True):
        companies = company_index[company_index["sector"] == sector]
        sectors = sector_index[sector_index["sector"] == sector]
        rows.append(
            {
                "sector": sector,
                "headline_count": len(headlines),
                "matched_headline_count": int(headlines["lm_has_match"].sum()),
                "matched_headline_ratio": headlines["lm_has_match"].mean(),
                "token_count": int(headlines["lm_token_count"].sum()),
                "matched_token_count": int(headlines["lm_matched_token_count"].sum()),
                "matched_token_ratio": (
                    headlines["lm_matched_token_count"].sum()
                    / headlines["lm_token_count"].sum()
                ),
                "company_calendar_rows": len(companies),
                "observed_company_days": int(companies["company_sentiment"].notna().sum()),
                "company_news_coverage_ratio": companies["company_sentiment"].notna().mean(),
                "trading_days": len(sectors),
                "observed_sector_days": int(sectors["lm_sector_sentiment"].notna().sum()),
                "sector_news_coverage_ratio": sectors["lm_sector_sentiment"].notna().mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("sector").reset_index(drop=True)


def disagreement_examples(comparison: pd.DataFrame) -> pd.DataFrame:
    """Select one largest fixed-rule classification disagreement per sector."""
    disagreements = comparison[comparison["vader_class"] != comparison["lm_class"]].copy()
    disagreements = disagreements.sort_values(
        ["absolute_score_difference", "trading_date", "ticker", "title"],
        ascending=[False, True, True, True],
    )
    examples = disagreements.groupby("sector", observed=True, sort=True).head(1)
    columns = [
        "trading_date",
        "ticker",
        "sector",
        "title",
        "vader_compound",
        "vader_class",
        "lm_tone",
        "lm_class",
        "lm_positive_terms",
        "lm_negative_terms",
        "lm_matched_token_ratio",
        "absolute_score_difference",
    ]
    return examples[columns].sort_values("sector").reset_index(drop=True)


def methodology_table() -> pd.DataFrame:
    """Record historical provenance and the fixed, untuned scoring specification."""
    return pd.DataFrame(
        [
            {
                "model": "Loughran-McDonald",
                "resource": LM_VERSION,
                "version_date": LM_VERSION_DATE,
                "official_page_archive_date": LM_ARCHIVE_DATE,
                "official_archived_source": LM_OFFICIAL_ARCHIVE_URL,
                "official_download": LM_DOWNLOAD_URL,
                "sha256": LM_SHA256,
                "paper_doi": LM_PAPER_DOI,
                "scoring_formula": LM_FORMULA,
                "tokenization": "case-insensitive alphabetic unigrams of at least two letters",
                "multi_word_policy": "no phrase scoring; punctuation and hyphens split tokens",
                "unmatched_headline_policy": "observed zero tone; matched-token coverage retained",
                "no_news_policy": "missing company/sector sentiment; zero news coverage",
                "aggregation": (
                    "headline mean to ticker-day; equal-weight ticker-days to sector-day"
                ),
                "signal_lag": "one equity trading day",
                "parameter_selection": "fixed ex ante; no OOS tuning; not used in portfolio fusion",
            }
        ]
    )


def run_lm_robustness(
    aligned_headlines: pd.DataFrame,
    trading_calendar: pd.Series | pd.DatetimeIndex,
    sector_universe: pd.DataFrame,
    vader_headline_scores: pd.DataFrame,
    vader_sector_index: pd.DataFrame,
    cache_path: pathlib.Path,
) -> LMRobustnessResults:
    """Build separate LM indexes and fixed VADER comparison outputs."""
    workbook_path = ensure_lm_wordlists(cache_path)
    word_lists = load_lm_wordlists(workbook_path)
    headlines = score_lm_headlines(aligned_headlines, word_lists)
    companies = lm_company_index(headlines, trading_calendar, sector_universe)
    sectors = lm_sector_index(companies, trading_calendar, sector_universe)
    comparison = headline_model_comparison(vader_headline_scores, headlines)
    return LMRobustnessResults(
        headline_scores=headlines,
        company_index=companies,
        sector_index=sectors,
        headline_comparison=comparison,
        distribution_summary=distribution_summary(comparison),
        sector_comparison=sector_model_comparison(vader_sector_index, sectors),
        coverage_summary=coverage_summary(comparison, companies, sectors),
        disagreement_examples=disagreement_examples(comparison),
        methodology=methodology_table(),
    )
