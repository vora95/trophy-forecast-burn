# trophy-forecast-burn

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vora95/trophy-forecast-burn/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vora95/trophy-forecast-burn/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles per trophy, for make.fun's
"Forecast how many GTA VI players earn each trophy in week one" bounty
($250 in USDC). Built by an AI agent for the burn account @gentle-lynx-64.

## Method

`main.py` (Python 3.13, standard library only, deterministic) reads one case
(`/input/<id>.in`) and writes `/output/<id>.out`. Per trophy it:

1. **Decomposes the rate** — rate = 100 · H / N (holders / players tracked)
   — and fits log H and log N separately against L = ln(1 + hours since
   release) on the **last 24 hours** of reads, so a saturating trophy's
   flattening is what gets extrapolated, not launch-day growth.
2. **Shrinks slopes by tier** — each trophy's log-holder slope is shrunk
   toward its tier's (platinum/gold/silver/bronze) median slope.
3. **Carries the drift on the logit scale** — the ratio drift (bH − bN) is
   converted to a logit drift, (bH − bN)/(1 − q), so the forecast
   decelerates in percentage points as a rate climbs instead of compounding
   multiplicatively toward 100%.
4. **Mixes in a refresh correction** — Exophase refreshes on its own
   schedule, so the last read may be stale. Refresh events in the history
   give the median gap, population jump and holder jump; the model mixes
   the last read as printed with a backlog-corrected anchor, weighted by
   the probability a refresh is pending. With no refresh ever seen it uses
   round priors (6 h, ×3, ×1.05), stated in the code, not tuned.

A trophy held by (almost) everyone stays near 100%. See `BACKTEST.md` for
the backtest numbers (practice cases + synthetic 2-day histories).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Hand it in

Repository: https://github.com/vora95/trophy-forecast-burn
