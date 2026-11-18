# GTA VI Twitch CCU forecast — v5: blended size-regression ensemble

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mmr9759/gta6-twitch-ccu-forecast-01-v5/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mmr9759/gta6-twitch-ccu-forecast-01-v5/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

**Entry:**** `main.py` (Python 3.13, standard library only). Reads one case
(`/input/<id>.in`), writes `/output/<id>.out`: for hours 2–24 of the
2026-11-19 UTC release day, five quantiles (5/25/50/75/95%) of Twitch
concurrent viewers. No network, no model calls, deterministic.

## What it does

The forecast is anchored on the case's known hour-1 print `p` and borrows
the day's shape from the past big launches shipped in the case file. All
work is in log space, on ratios to hour 1, so a miss of 20% costs the same
at 10,000 viewers as at 1,000,000 — the same scale the pinball loss uses.

For each hour `h`, the median log-ratio `ln(v_h / p)` blends three
components, each fitted fresh from the case's own comparables:

1. **All-pool regression (weight 0.50).** OLS of `ln(v_h / v_1)` on
   `ln(v_1)` over the pool. Launch size matters: a launch whose first hour
   is already huge grows differently through the day than one that starts
   quiet (rolling-midnight launches start quiet at 00:00 UTC because most
   regions have not unlocked yet, then surge as the US and Europe wake up).
2. **Rolling regression (weight 0.25).** The same OLS over recent launches
   only (released 2024 or later; the 15 most recent if fewer than 8
   qualify), so the fit tracks how launch-day shapes have drifted.
3. **Nearest-shape median (weight 0.25).** The median log-ratio of the 8
   comparables closest in size (`|ln(v_1) − ln(p)|`), with a +0.35 distance
   penalty for launches whose unlock was not a rolling midnight. This is
   the shape-transfer idea the strongest public entries use: borrow the
   day's shape from launches that unlocked the way GTA VI does, at a
   similar scale — but as a median of 8 neighbours, never a single launch.

**Pool.** Only comparables tagged `Tier: comparable` in their unlock note
(about 26 of 37 in the practice kit; the final case adds the four practice
launches). Early-access-contaminated, weak and free-to-play launches are
excluded: their hour-1 prints are distorted by early access or by not
being paid launches, so their ratios do not transfer. GTA VI's own
pre-release category history is not a launch and is excluded automatically
(it has no release-day hours). If fewer than 8 comparable launches qualify,
the whole usable pool is used; if the pool is still too small, the program
falls back to the house-style persistence forecast.

**Size defence (GTA VI will be bigger than anything in the pool).** The
regression is evaluated at `ln(p)` clamped to the pool's `ln(v_1)` range
±0.3, and when `ln(p)` lies outside the pool's range the regression
predictions are shrunk toward the pool's median shape, up to fully, so a
record hour-1 is never extrapolated linearly into absurdity. Log-ratios
are winsorised to [−2.5, +3.5] before fitting, so one freak launch (a
simultaneous global unlock at 14:00 UTC leaves hour 1 near zero) cannot
drag a regression.

**Spread (quantiles from residuals, not from a normal assumption).** Each
pool launch is forecast without itself in the pool (leave-one-out); the
residuals per hour give a per-hour sigma, shrunk toward the global sigma,
and the quantiles are `median × exp(sigma_h × z_tau)` where `z_tau` are
the empirical quantiles of the pooled standardised residuals. The 5–95%
band is therefore calibrated on past launch days by construction.

## Assumptions

- GTA VI unlocks rolling midnight local (New Zealand first, 2026-11-18
  11:00 UTC), like the rolling-midnight analogues the nearest-shape
  component prefers. If it unlocked simultaneously worldwide instead, the
  day's shape would be sharper and hour 1 larger; the size regression
  would still anchor on the real hour-1 print, but the shape would be
  biased toward the rolling pattern.
- The hour-1 print is informative about the day's scale. For rolling
  launches it is: the regression of log-ratio on log hour-1 is what lets a
  small hour-1 (quiet midnight) predict a large daytime surge.
- Past launch days (2020–2026, 26 comparable launches) are a fair guide
  to GTA VI's shape, up to the size shrinkage above. GTA VI is expected
  to be the biggest launch in the pool; the model does not invent a new
  shape for it, it borrows the biggest launches' shapes and shrinks the
  size extrapolation.
- SullyGnome hourly averages (practice) and the recorder's minute means
  (final) measure the same quantity closely enough that a shape fitted on
  one transfers to the other.

## Backtest (honest, computed from the kit's own data only)

Leave-one-out over the comparables shipped in the kit: each launch is
forecast without itself in the pool, scored with the competition's
pinball loss on the log scale. No practice-case truth is available
locally (the server holds it), so LOO is the honest proxy; the practice
board score is reported by the server, not reproduced here.

| evaluation | mean pinball loss |
|---|---|
| v5, LOO on rolling-midnight + comparable launches (n = 11, the GTA VI-like subset) | **0.244** |
| v5, LOO on all 37 comparable launches | 0.559 |
| v5, walk-forward LOO on the rolling + comparable subset (train only on earlier launches) | 0.280 |
| house persistence baseline, LOO on the rolling + comparable subset | 0.363 |
| house baseline on the practice board (per the rules) | ~0.387 |

On the GTA VI-like subset the 90% band covers 0.88 of scored hours —
close to the honest 0.90. The all-launches LOO is much worse because it
includes simultaneous-global launches (e.g. a 14:00 UTC unlock leaves
hour 1 near zero, then the count explodes 100×), which are not GTA VI
analogues; the model is built for the rolling-midnight case and says so,
and its spread is deliberately calibrated on rolling-midnight launches
only, so it under-covers those (0.58) by design. The walk-forward
coverage (0.73) is lower because early launches train on very few
predecessors; the final trains on the full pool.

For reference, the previous v3 entry ("blended size-regression
ensemble") scored **0.19** on the server's practice board — the best of
v1–v4 — while v4 scored 0.204. v5 keeps v3's blend (same three
components, same 0.50/0.25/0.25 weights, same k = 8) and adds: a
comparable-only pool (v3 used every launch), a rolling-midnight
preference in the nearest-shape component, the size-shrinkage guard, and
empirical residual quantiles for the spread. Weight and k variants were
tested in LOO and moved the score by ≤0.004 — inside noise — so none
were chased; the v3 weights were kept.

Known weak spot: Resident Evil Requiem (2026, rolling midnight) is the
worst LOO target (0.89, coverage 0.43) — a small hour-1 (46k) that grew
to a 563k peak, faster than its neighbours predicted. If GTA VI's
hour-1 turns out small relative to its daytime peak, the same
under-prediction risk applies; the residual-based spread is the
mitigation, and it is wider for exactly those fast-growing hours.

## Verification

- `MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py`
  runs all 4 practice cases in well under 8 s each; two runs produce
  byte-identical outputs (checked with `sha256sum`).
- Outputs validated against the format: exactly 23 rows, hours 2–24,
  five non-decreasing quantiles each, all in (0, 1e9].
- `makefun check` (kit CLI) run locally before submit; see the
  coordinator's report for its verdict.

## Running it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
