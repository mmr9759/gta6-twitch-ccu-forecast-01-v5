#!/usr/bin/env python3
"""GTA VI Twitch CCU forecast — blended size-regression ensemble (v5).

Forecasts hours 2..24 of a launch day's Twitch concurrent viewers, as five
quantiles an hour, from the case's known hour-1 print and the release-day
shapes of past big launches ("comparables") shipped in the case file.

Method (all in log space, ratios to the hour-1 print p):
  For each hour h, the median log-ratio ln(v_h / p) is a blend of three
  components, all fitted fresh from the case's own comparables:
    A. all-pool regression: OLS of ln(v_h / v_1) on ln(v_1) over the pool.
       Big launches grow/decay through the day differently from small ones,
       so the shape is allowed to depend on launch size.
    B. rolling regression: the same OLS over recent launches only
       (released 2024 or later; the 15 most recent if fewer than 8 qualify),
       so the fit tracks how launch-day shapes have drifted lately.
    C. nearest-shape median: the median log-ratio of the k = 8 comparables
       closest in size (|ln(v_1) - ln(p)|), with a small distance penalty
       for launches whose unlock was not a rolling midnight, so the shape
       is borrowed preferentially from launches that unlocked the way
       GTA VI does (rolling midnight local, New Zealand first).
  Blend weights: 0.50 * A + 0.25 * B + 0.25 * C.
  Size defence: the regression is evaluated at ln(p) clamped to the pool's
  ln(v_1) range plus a 0.3 margin, and when ln(p) lies outside the pool's
  range the regression predictions are shrunk toward the pool's median
  shape, so a GTA VI hour-1 far larger than any past launch is never
  extrapolated linearly into absurdity.
  Spread: honest leave-one-out residuals. Each rolling-midnight
  comparable launch is forecast without itself in the pool; the
  residuals per hour give a per-hour sigma (shrunk toward the global
  sigma), and the quantiles are the median times exp(sigma_h * z_tau),
  where z_tau are the empirical quantiles of the pooled standardised
  residuals. Residuals come only from rolling-midnight comparable
  launches — the target's own kind — because simultaneous-global and
  early-access launches have far wilder hour-1-to-day ratios and would
  blow the band up to cover launches the final is nothing like. The
  5%-95% band therefore covers about 90% of past launch days of the
  target's kind by construction.

Pool: only comparables tagged "Tier: comparable" in their unlock note
(early-access-contaminated, weak and free-to-play launches are left out:
their hour-1 prints are distorted by early access or by not being paid
launches, so their ratios do not transfer). If fewer than 8 qualify, the
whole usable pool is used. GTA VI's own pre-release category history is
not a launch and is excluded automatically (it has no release-day hours).

Deterministic: no randomness, no wall clock, no network; every
aggregation sorts first. Two runs on the same case are byte-identical.
Only the standard library is used. Writes only to $MAKEFUN_OUTPUT_DIR
(or /output).
"""

import json
import math
import os
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HOURS = tuple(range(2, 25))
CLIP_LO = -2.5
CLIP_HI = 3.5
W_ALL = 0.50
W_REC = 0.25
W_NEAR = 0.25
K_NEAR = 8
MIN_POOL = 8
RECENT_CUTOFF = "2024-01-01"
SHRINK_PRIOR = 8.0


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def winsor(x):
    return CLIP_LO if x < CLIP_LO else (CLIP_HI if x > CLIP_HI else x)


def median_of(sorted_values):
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    return my - b * mx, b


def day_hours(comp):
    """Release-day hourly viewers {hour 1..24: viewers} or None if incomplete."""
    try:
        start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for point in comp.get("hourly") or []:
        try:
            viewers = point["viewers"]
            at = parse_ts(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if not viewers or viewers <= 0:
            continue
        idx = int((at - start).total_seconds() // 3600) + 1
        if 1 <= idx <= 24:
            hours[idx] = float(viewers)
    if len(hours) < 24 or hours.get(1, 0) <= 0:
        return None
    return hours


def launch_info(comp):
    """(hours, date, is_comparable, is_rolling) or None when unusable."""
    hours = day_hours(comp)
    if hours is None:
        return None
    note = str(comp.get("unlock") or "")
    return {
        "hours": hours,
        "date": str(comp.get("day_start") or "")[:10],
        "comparable": "Tier: comparable" in note,
        "rolling": ("rolling" in note.lower()
                    or "midnight local" in note.lower()),
    }


def regression_prediction(pool, log_p, hour):
    """OLS of winsorised log-ratio on log hour-1, evaluated at log_p."""
    xs = []
    ys = []
    for launch in pool:
        v1 = launch["hours"][1]
        xs.append(math.log(v1))
        ys.append(winsor(math.log(launch["hours"][hour] / v1)))
    a, b = ols(xs, ys)
    lo = min(xs) - 0.3
    hi = max(xs) + 0.3
    clamped = lo if log_p < lo else (hi if log_p > hi else log_p)
    pred = a + b * clamped
    out_frac = 0.0
    if log_p > hi - 0.3:
        out_frac = min(1.0, (log_p - (hi - 0.3)) / 1.5)
    elif log_p < lo + 0.3:
        out_frac = min(1.0, ((lo + 0.3) - log_p) / 1.5)
    if out_frac > 0.0:
        med = median_of(sorted(ys))
        pred = (1.0 - out_frac) * pred + out_frac * med
    return pred


def nearest_prediction(pool, log_p, hour):
    """Median winsorised log-ratio of the K_NEAR closest launches by size."""
    def distance(launch):
        d = abs(math.log(launch["hours"][1]) - log_p)
        if not launch["rolling"]:
            d += 0.35
        if not launch["comparable"]:
            d += 0.35
        return d

    near = sorted(pool, key=lambda c: (distance(c), c["date"], c["hours"][1]))
    ratios = sorted(
        winsor(math.log(c["hours"][hour] / c["hours"][1]))
        for c in near[:K_NEAR]
    )
    return median_of(ratios)


def recent_pool(pool):
    """Recent launches (2024+), else the 15 most recent by date."""
    recent = [c for c in pool if c["date"] >= RECENT_CUTOFF]
    if len(recent) >= MIN_POOL:
        return recent
    ordered = sorted(pool, key=lambda c: (c["date"], c["hours"][1]))
    return ordered[-15:]


def blend_prediction(pool, log_p, hour):
    """0.50 all-pool regression + 0.25 rolling regression + 0.25 nearest."""
    pred_all = regression_prediction(pool, log_p, hour)
    pred_rec = regression_prediction(recent_pool(pool), log_p, hour)
    pred_near = nearest_prediction(pool, log_p, hour)
    return W_ALL * pred_all + W_REC * pred_rec + W_NEAR * pred_near


def loo_spread(pool, log_p):
    """Per-hour sigma and empirical standardised-residual quantiles.

    Each rolling-midnight comparable launch is forecast without itself;
    the residuals per hour give a per-hour sigma, shrunk toward the
    global sigma, and the quantiles are the median times
    exp(sigma_h * z_tau), where z_tau are the empirical quantiles of
    the pooled standardised residuals. Residuals are taken only from
    rolling-midnight comparable launches — the launches shaped like the
    target — because simultaneous-global and early-access launches have
    far wilder hour-1-to-day ratios and would blow the band up to
    cover launches the final is nothing like. The 5%-95% band is
    therefore calibrated on past launch days of the target's own kind,
    not on a normal assumption.
    """
    residuals = {h: [] for h in HOURS}
    for target in pool:
        if not (target["rolling"] and target["comparable"]):
            continue
        rest = [c for c in pool if c is not target]
        if len(rest) < MIN_POOL:
            continue
        t_log_p = math.log(target["hours"][1])
        for h in HOURS:
            pred = blend_prediction(rest, t_log_p, h)
            residuals[h].append(
                math.log(target["hours"][h] / target["hours"][1]) - pred
            )
    all_res = [r for h in HOURS for r in residuals[h]]
    if not all_res:
        return {h: 0.4 for h in HOURS}, {tau: 0.0 for tau in LEVELS}
    global_var = sum(r * r for r in all_res) / len(all_res)
    global_sigma = math.sqrt(global_var)
    sigma = {}
    for h in HOURS:
        rs = residuals[h]
        if len(rs) >= 2:
            var_h = sum(r * r for r in rs) / len(rs)
        else:
            var_h = global_var
        sigma[h] = math.sqrt(
            (len(rs) * var_h + SHRINK_PRIOR * global_var)
            / (len(rs) + SHRINK_PRIOR)
        )
        if sigma[h] <= 1e-9:
            sigma[h] = global_sigma if global_sigma > 1e-9 else 0.4
    standardised = sorted(r / sigma[h] for h in HOURS for r in residuals[h])
    offsets = {}
    for tau in LEVELS:
        pos = tau * (len(standardised) - 1)
        lo = int(pos)
        frac = pos - lo
        hi = lo + 1 if lo + 1 < len(standardised) else lo
        offsets[tau] = standardised[lo] * (1.0 - frac) + standardised[hi] * frac
    return sigma, offsets


def fallback_hours(p):
    """House-style persistence forecast, used only if the pool is unusable."""
    z = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817, 0.5: 0.0,
         0.75: 0.6744897501960817, 0.95: 1.6448536269514722}
    rows = []
    for h in HOURS:
        qs = [p * math.exp(z[tau] * 0.15 * math.sqrt(h - 1)) for tau in LEVELS]
        rows.append({"hour": h, "q": clean_quantiles(qs)})
    return rows


def clean_quantiles(qs):
    """Round, clamp to (0, 1e9], enforce non-decreasing order."""
    out = []
    prev = 0.0
    for q in qs:
        v = round(float(q), 2)
        if not math.isfinite(v) or v <= 0.0:
            v = 1.0
        if v > 1e9:
            v = 1e9
        if v < prev:
            v = prev
        out.append(v)
        prev = v
    return out


def forecast_case(case):
    known = case.get("known") or []
    p = 0.0
    if known:
        try:
            p = float(known[0].get("viewers") or 0)
        except (TypeError, ValueError):
            p = 0.0
    if not p or p <= 0:
        p = 1.0
    log_p = math.log(p)

    launches = []
    for comp in case.get("comparables") or []:
        info = launch_info(comp)
        if info is not None:
            launches.append(info)
    pool = [c for c in launches if c["comparable"]]
    if len(pool) < MIN_POOL:
        pool = launches
    if len(pool) < MIN_POOL:
        return {"hours": fallback_hours(p)}

    sigma, offsets = loo_spread(pool, log_p)
    rows = []
    for h in HOURS:
        med_log_ratio = blend_prediction(pool, log_p, h)
        med = p * math.exp(med_log_ratio)
        qs = [med * math.exp(sigma[h] * offsets[tau]) for tau in LEVELS]
        rows.append({"hour": h, "q": clean_quantiles(qs)})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        with open(os.path.join(out_dir, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
