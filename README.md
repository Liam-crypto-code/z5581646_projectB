# FinTech Project - Part B

> FIRST: rename this folder to <yourZID>_projectB (for example z1234567_projectB)
> and move it into fins-agent/fins2026/. The folder name carrying your zID is your
> submission.

Part B: funds, sentiment, and the app (DFF Stations 3-4). This folder is also your
public GitHub repository; the app entrypoint is streamlit_app.py at the root.

## How to run

    pip install -r requirements.txt -r requirements-dev.txt   # dev adds nltk (VADER)
    python scripts/run_part_b.py            # reproduces your results into results/
    python scripts/build_app_data.py         # rebuilds the native-calendar app artifact only
    streamlit run streamlit_app.py          # runs the app locally

Load raw data through src/data_access.py (see context/DATA_GUIDE.md); never commit
raw data. The deployed app, by contrast, reads your precomputed artifacts from
results/ - those ARE committed.

The deployed app never imports the raw-data loader, optimisation, sentiment, or
fusion modules. It validates and reads saved CSV artifacts under `results/` only.
`results/data/asset_performance.csv` is generated outside Streamlit from the
verified Part A return panels: equities retain their trading-day calendar and
crypto retains its native daily calendar, including weekends.

## Fund universes

The saved fund artifacts contain ten walk-forward out-of-sample funds:

- Combined: Equal Weight, Minimum Variance, Maximum Sharpe, and the existing
  Sentiment-Tilted Equal Weight comparison. These use the equity trading calendar,
  a trailing 252-trading-day window, and 252-day annualisation.
- Equity-only: Equal Weight, Minimum Variance, and Maximum Sharpe across all 50
  equities with the same 252-day convention.
- Crypto-only: Equal Weight, Minimum Variance, and Maximum Sharpe across all 10
  crypto assets on their native daily calendar, including weekends, with a trailing
  365-calendar-day window and 365-day annualisation.

All funds use monthly rebalancing, next-observation effectiveness, weight drift,
long-only fully invested targets, zero transaction costs, and a zero risk-free
rate. The fixed 10% Maximum Sharpe cap forces the 10-asset crypto portfolio to
equal weights; this is a constraint implication, not an out-of-sample calibration.

## What is here

- streamlit_app.py    the app entrypoint (repo root)
- .streamlit/         app config
- PROJECT_BRIEF.md    the full assignment brief for your course (read this first)
- src/                your code (data_access is provided; portfolios/sentiment/fusion are yours)
- scripts/            runnable scripts that reproduce your results
- results/            your outputs: figures in results/figures/, tables in results/tables/, app data artifacts in results/data/
- context/            provided data guide and project context (do not edit)
- report/             your report - see report/OUTLINE.md (author in Word, submit report.pdf)
- ai/                 your prompt logs and AI notes
- requirements-dev.txt build/repro-only deps (nltk); keep them out of the deployed app
- AGENTS.md / CLAUDE.md   replace the stub for your tool (you need just one) with your own

## Deploy + hand in

This folder is its own GitHub repo, independent of fins-agent. Your AI agent can run
the check and push the repo; the browser deploy is yours (it needs your login). See
PROJECT_BRIEF.md Appendix D and docs/STUDENT_DEPLOY.md (in this folder). In short:

    python scripts/check_handin.py        # your agent can run this
    # commit your precomputed app artifacts under results/ (the app reads them)
    # git init in this folder, then push the contents to a NEW private GitHub repo

Then YOU connect the repo on share.streamlit.io (entrypoint streamlit_app.py). At
hand-in, make the repo PUBLIC, submit the live URL + repo link, and also zip this
whole folder and upload the zip to Moodle.
