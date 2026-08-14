# Prompt log - Part B foundation reuse (Stations 1 and 2)

## What I wanted

I wanted to understand the full Part B starter before editing it, then reuse the
verified Part A cleaning, adjusted-close return, calendar-alignment and
headline-assembly work. I did not want portfolio, sentiment or app work to start
until that foundation was reproduced in Part B.

## Prompt(s)

> Please read and follow the AGENTS.md in the current Project B folder. First
> inspect the whole project structure ... Also inspect my completed Part A project
> at ../z5581646_projectA ... Do not modify any files yet.

> Please first reuse the verified Part A data-processing foundation in this Part B
> project. Port the ETL, return-processing, calendar-alignment and
> headline-assembly logic while keeping src/data_access.py unchanged. Do not start
> Station 3 portfolio optimisation, sentiment scoring, fusion or the Streamlit
> dashboard yet.

## What the assistant produced

It ported the reusable Part A ETL and feature logic, updated the Part B runner,
and added regression tests. The runner reproduced the cleaned 50,300 equity rows,
14,610 in-sample crypto rows, 146,836 deduplicated headlines, 1,006 equity-calendar
dates and 37,962 ticker-date headline rows.

## What was wrong or risky

The highest risk was timing and calendar leakage. Crypto must have returns
calculated on its native calendar before it is selected onto equity dates. Headlines
from a weekend or Monday must not be available for a Monday portfolio decision.
The hosted-data smoke test initially failed in the restricted environment because
the download was blocked; that was not treated as evidence that the data logic was
wrong.

## What I changed and why

I kept `src/data_access.py` unchanged and required the runner to regenerate full
panels rather than using small Part A samples as model inputs. I checked that the
foundation kept adjusted-close returns, normalised dates, removed only exact
duplicates, calculated crypto returns before alignment, and preserved the
same-or-next-trading-day headline mapping. I also required focused tests before
moving to Station 3.
