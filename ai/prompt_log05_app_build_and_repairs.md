# Prompt log - FinMosaic app and application repairs

## What I wanted

I wanted a personalised, historical-information Streamlit app that uses only
precomputed artifacts under `results/`. The app must support a fund comparison,
watchlist and news exploration without recalculating models or providing
investment advice.

## Prompt(s)

> Please build the FinMosaic Streamlit app using only the existing precomputed
> outputs under results/. Do not recalculate returns, portfolio weights,
> optimisation, VADER scores, Loughran-McDonald scores, or sentiment fusion inside
> the app.

> Please add a new default workspace called "My Mosaic" ... using only the
> existing precomputed outputs under results/ ... Do not add data sources,
> calculations, forecasts, investment advice, or new portfolio methods.

> Fix the KeyError in My Mosaic ... replace the undefined data["sector_summary"]
> access with the existing validated data["sector_sentiment"] output.

> The Portfolio news context module currently crashes with KeyError:
> 'portfolio_sentiment' ... Finish this feature end to end ...

## What the assistant produced

It built a results-only Streamlit app with My Mosaic, Funds, Watchlist, News and
Methodology views. It added a precomputed asset-performance artifact and later a
precomputed portfolio-news-context artifact. The app can compare combined,
equity-only and crypto-only funds, show holdings, historical performance and
sentiment coverage.

## What was wrong or risky

The first My Mosaic version referenced a non-existent `sector_summary` key.
Later, the Funds page referenced `portfolio_sentiment` before the artifact was
registered in `src/app_data.py`; an old cached bundle meant the KeyError persisted
after a restart. Daily company sentiment was also too dense to be a useful default
chart, and crypto selections cannot honestly show equity-company news coverage.

## What I changed and why

I required the UI to use only validated saved artifacts and asked for focused
rendering/data-loader tests. I required a cache-version change after the new
portfolio-sentiment schema was registered. I changed the default News view to
monthly observed-only sentiment with coverage/headline volume and kept short daily
views for detail. I retained separate labels for no news and neutral sentiment,
and did not pretend that the course's equity-company headlines cover crypto news.
