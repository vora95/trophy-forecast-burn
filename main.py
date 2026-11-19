# SPDX-License-Identifier: MIT
"""GTA VI trophy rarity forecaster (burn entry, @gentle-lynx-64).

Methodology: holder/population decomposition with a refresh-mixture and
tier shrinkage.  This is a different model family from a direct least-squares
fit of logit(rate) on log time:

1.  DECOMPOSITION.  An earn rate is a ratio, rate = 100 * H / N, where H is
    the number of tracked players holding the trophy and N the players
    tracked.  H and N move for different reasons (players progress through
    the game vs new players join and dilute every rate), so each is
    extrapolated separately: log H and log N are each fit by least squares
    against L = ln(1 + hours since release), then anchored at the last read
    and grown to the target.  The forecast rate is the ratio of the two
    extrapolations, so dilution (N outgrowing H) lowers a rate even while
    its holder count rises.  The ratio drift is carried on the LOGIT scale:
    if H and N grow exponentially at rates bH and bN, the instantaneous
    drift of logit(rate) is (bH - bN) / (1 - q), q the current rate as a
    fraction; extrapolating that drift (rather than the ratio in levels)
    makes the forecast decelerate in percentage points as a rate climbs,
    instead of compounding multiplicatively toward 100%.  Slopes are fit
    on the LAST 24 HOURS of reads only: a trophy whose holders have
    stopped growing (its curve saturating) shows a ~flat late slope, so
    the extrapolation does not blow past its asymptote on the strength of
    launch-day growth; a trophy still rising keeps its late slope.  With
    less than 24 h of history all reads are used.

2.  TIER SHRINKAGE.  A trophy's fitted log-holder slope is shrunk toward
    the median slope of its tier (platinum/gold/silver/bronze).  A trophy
    with few distinct reads borrows most of its slope from its tier; a
    trophy with a long, varied history keeps its own.  This pools the
    cross-trophy structure instead of fitting 55 independent curves.

3.  REFRESH MIXTURE (staleness).  Exophase refreshes its counts on its own
    schedule, hours apart, so the last read may already be stale: the true
    holders and tracked players have kept growing since it.  The history's
    own refresh events (reads where N or any rate jumps) give the median
    gap T between refreshes, the median population jump D and the median
    per-trophy holder jump G.  If the page has been unchanged for s hours,
    the model mixes two anchors -- the last read as printed, and the last
    read corrected for the backlog grown over s hours -- with weight
    P = 1 - exp(-s / T) on the corrected anchor.  With no refresh ever seen
    in the history (flat launch night) it falls back to round priors
    (T = 6 h, D = 3, G = 1.05) stated here, not tuned to any answer.
    Quantiles come from inverting the two-component normal mixture on the
    logit scale, so a possibly-stale read yields an honestly wide, skewed
    band instead of a narrow band around a stale number.

4.  SPREAD.  Logit sigma combines a random-walk term (0.5 * sqrt(days)),
    the per-trophy residual of the log-holder fit (floored, so a flat
    history does not claim zero volatility), and an extra term for the
    refresh component.

A trophy held by (almost) everyone in the last read is special-cased to
stay near 100%: new players earn it immediately too.

Standard library only, deterministic (no randomness, no wall-clock use),
reads $MAKEFUN_INPUT_DIR (else /input) and writes $MAKEFUN_OUTPUT_DIR
(else /output), one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Refresh priors used ONLY when the history shows no refresh at all.
PRIOR_GAP_HOURS = 6.0
PRIOR_POP_JUMP = 3.0
PRIOR_HOLDER_JUMP = 1.05
# Shrinkage strength: a trophy's own slope weight is m / (m + SHRINK_K),
# where m is its number of distinct holder counts.
SHRINK_K = 6.0
# Safety rails on fitted log-slopes (per unit of L = ln(1 + hours)).
SLOPE_CAP = 3.0
# Spread floors/caps (logit units).
SIGMA_FIT_FLOOR = 0.30
SIGMA_FIT_CAP = 1.00


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log_time(at):
    """L = ln(1 + hours since release); guarded for pre-release reads."""
    hours = (at - RELEASE).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def ols(xs, ys):
    """Least-squares slope/intercept of y on x; returns (slope, resid_std)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    if n > 2:
        var = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return slope, resid


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantile(level, comps):
    """Invert a mixture of normals on the logit scale by bisection.

    comps: list of (weight, mu, sigma).  Deterministic; 80 iterations.
    """
    mus = [c[1] for c in comps]
    sigs = [c[2] for c in comps]
    lo = min(mus) - 10.0 * max(sigs) - 1.0
    hi = max(mus) + 10.0 * max(sigs) + 1.0

    def cdf(x):
        return sum(w * norm_cdf((x - mu) / sg)
                   for w, mu, sg in comps)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    l_target = log_time(target)

    # ---- shared series: log N on L, and the refresh events -------------
    reads = []  # (time, L, N, [rates])
    for h in history:
        rates = h["rates"]
        reads.append((parse_time(h["at"]), log_time(parse_time(h["at"])),
                      float(h["players_tracked"]), rates))

    # Slope window: last 24 h of reads (all reads when history is shorter),
    # so a saturating curve's flattening is what gets extrapolated.
    cutoff = at - timedelta(hours=24)
    win = [r for r in reads if r[0] >= cutoff]
    if len(win) < 3:
        win = reads
    xs_nw = [r[1] for r in win]
    ys_nw = [math.log(max(r[2], 1.0)) for r in win]
    b_n, _ = ols(xs_nw, ys_nw)
    b_n = min(max(b_n, 0.0), SLOPE_CAP)

    # Refresh events: boundaries where N (or any rate) changes.  An event
    # time is the first read of a new plateau.
    events = []  # (time, D, [G per trophy])
    prev_n = None
    prev_rates = None
    for (t, _l, n, rates) in reads:
        changed = (prev_n is not None and
                   (abs(n - prev_n) > 0.001 * max(prev_n, 1.0) or rates != prev_rates))
        if changed:
            d = n / max(prev_n, 1.0)
            gs = []
            for i, r in enumerate(rates):
                pr = prev_rates[i]
                h_now = (r if r is not None else 0.0) * n / 100.0
                h_prev = (pr if pr is not None else 0.0) * prev_n / 100.0
                gs.append(max(h_now, 0.5) / max(h_prev, 0.5))
            events.append((t, d, gs))
        prev_n, prev_rates = n, rates
    if events:
        gaps = []
        for k in range(1, len(events)):
            gaps.append((events[k][0] - events[k - 1][0]).total_seconds() / 3600.0)
        # Gap from the first read to the first event also informs cadence.
        gaps.append((events[0][0] - reads[0][0]).total_seconds() / 3600.0)
        gaps = [g for g in gaps if g > 0]
        t_refresh = median(gaps) if gaps else PRIOR_GAP_HOURS
        t_refresh = min(max(t_refresh, 0.5), 48.0)
        d_med = median([e[1] for e in events])
        d_med = min(max(d_med, 1.0), 50.0)
        last_change = events[-1][0]
    else:
        t_refresh = PRIOR_GAP_HOURS
        d_med = PRIOR_POP_JUMP
        last_change = reads[0][0]
    stale_hours = max((at - last_change).total_seconds() / 3600.0, 0.0)
    p_refresh = 1.0 - math.exp(-stale_hours / t_refresh)
    p_refresh = min(max(p_refresh, 0.0), 0.8)
    # Backlog grown over the stale interval, geometric in stale / T.
    frac = min(stale_hours / t_refresh, 1.5)
    d_backlog = d_med ** frac

    # ---- per-trophy log-holder fits (same last-24h window) -------------
    slopes = []
    for i in range(len(trophies)):
        xs, ys = [], []
        for (_t, l, n, rates) in win:
            r = rates[i]
            if r is None:
                continue
            xs.append(l)
            ys.append(math.log(max(r * n / 100.0, 0.5)))
        slope, _resid = ols(xs, ys)
        slopes.append(slope)
    tier_slopes = {}
    for tier in {t["tier"] for t in trophies}:
        tier_slopes[tier] = median([slopes[i] for i, t in enumerate(trophies)
                                    if t["tier"] == tier])

    sigma_rw = 0.5 * math.sqrt(horizon_days)
    rows = []
    last = reads[-1]
    l_last = last[1]
    n_last = last[2]
    dl = l_target - l_last

    for i, trophy in enumerate(trophies):
        rates_i = [r[3][i] for r in reads]
        known = [(r[1], r[3][i], r[2]) for r in reads if r[3][i] is not None]
        if not known:
            rows.append({"id": trophy["id"],
                         "q": [round4(v) for v in (0.2, 0.5, 1.0, 2.0, 5.0)]})
            continue
        last_rate = known[-1][1]
        h_last = max(last_rate * n_last / 100.0, 0.5)

        # Entry trophy held by everyone: stays ~100.
        if last_rate >= 99.5:
            mu = logit(99.99)
            qs = [round4(inv_logit(mu + z * 0.05)) for z in Z]
            rows.append({"id": trophy["id"], "q": qs})
            continue

        # Shrink this trophy's holder slope toward its tier median.
        distinct = len({round(r, 4) for r in rates_i if r is not None})
        w = distinct / (distinct + SHRINK_K) if distinct else 0.0
        b_h = w * slopes[i] + (1.0 - w) * tier_slopes[trophy["tier"]]
        b_h = min(max(b_h, 0.0), SLOPE_CAP)

        # Residual of the log-holder fit (same window) -> per-trophy spread.
        xs = [k[0] for k in known if k[0] >= log_time(cutoff)]
        ys = [math.log(max(k[1] * k[2] / 100.0, 0.5)) for k in known
              if k[0] >= log_time(cutoff)]
        if len(xs) < 3:
            xs = [k[0] for k in known]
            ys = [math.log(max(k[1] * k[2] / 100.0, 0.5)) for k in known]
        _s, resid = ols(xs, ys)
        sigma_fit = min(max(resid, SIGMA_FIT_FLOOR), SIGMA_FIT_CAP)
        if len(known) < 3:
            sigma_fit = 0.5
        sigma_a = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2)

        # Anchor A: last read as printed.  Anchor B: corrected for backlog.
        # Both are extrapolated on the logit scale from their anchor rate,
        # with the drift implied by the H/N fits: (bH - bN) / (1 - q).
        q_last = last_rate / 100.0
        drift_logit = (b_h - b_n) / max(1.0 - q_last, 0.05)
        mu_a = logit(last_rate) + drift_logit * dl

        if events:
            g_med_i = median([e[2][i] for e in events])
        else:
            g_med_i = PRIOR_HOLDER_JUMP
        g_med_i = min(max(g_med_i, 0.5), 50.0)
        g_backlog = g_med_i ** frac
        r_b = clamp(100.0 * h_last * g_backlog / (n_last * d_backlog))
        mu_b = logit(r_b) + drift_logit * dl
        # The refresh component's extra spread scales with the size of the
        # backlog correction itself: when the two anchors coincide there is
        # no extra uncertainty to add, so a frequent-refresh history is not
        # charged width for a correction of ~zero.
        shift = abs(mu_b - mu_a)
        sigma_b = math.sqrt(sigma_a ** 2 + (0.5 * shift) ** 2 + 0.05 ** 2)

        if p_refresh < 0.02:
            qs = [round4(clamp(inv_logit(mu_a + z * sigma_a))) for z in Z]
        else:
            comps = [(1.0 - p_refresh, mu_a, sigma_a),
                     (p_refresh, mu_b, sigma_b)]
            qs = [round4(clamp(inv_logit(mixture_quantile(lv, comps))))
                  for lv in LEVELS]
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
