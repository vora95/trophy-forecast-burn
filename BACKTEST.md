# Backtest

Scoring is the exact pinball loss on the logit scale from the competition
rules (mean over trophies × 5 levels). "Baseline" is the kit starter
(persistence + random-walk spread, loss ≈ 1.51 on the real practice per the
rules). "Curve-like" is a re-implementation of the house curve baseline
(logit-rate least squares on ln(1+h) over the last 24 h) for reference.

## Practice cases (p00/p01) — synthetic answers

The kit ships no practice answers (they are scored server-side), so the
answers below are synthetic but aggregate-consistent with the facts the
rules themselves state: the first refresh took players tracked 638 → 3,868
and total trophies earned 943 → 4,276, with the 100% trophy staying at
100%. The 3,230 new players hold 3,333 new trophies; the entry trophy takes
3,230 of them and the remaining 103 are spread over the other 54 trophies —
S1 proportional to current holders, S2 concentrated on the most-held ones.

| model | S1 p00 | S1 p01 | S2 p00 | S2 p01 |
|---|---|---|---|---|
| baseline (starter) | 1.4048 | 1.4048 | 1.4269 | 1.4269 |
| **this model** | **0.9290** | **0.8291** | **0.9511** | **0.8512** |

Typical miss: baseline 0.68 pp, this model 0.49–0.53 pp. Coverage is 0.00
for both: the refresh was a ~6× dilution and no model with a conservative
prior covers it fully — the gain comes from the median shifting down and
the band widening. The refresh prior (population ×3) is a round prior
stated in the code, not fitted to the 6× figure.

## Pseudo-final — synthetic 2-day histories, 4.5-day horizon

Histories were generated with logistic trophy curves in log-time, a
power-law tracked-player base, and stepped Exophase refreshes (3–4 h).
F1/F2: fast progression (trophies mostly saturated by the deadline);
F3/F4: slow progression (still rising at the deadline). Two player-growth
rates (N ∝ (1+h)^0.9 and ^1.3).

| scenario | baseline | this model | curve-like |
|---|---|---|---|
| F1 (fast, N^0.9) | 0.2260 | 0.2350 | 0.2252 |
| F2 (fast, N^1.3) | 0.2268 | 0.2346 | 0.2253 |
| F3 (slow, N^0.9) | 0.3126 | 0.2532 | 0.2353 |
| F4 (slow, N^1.3) | 0.3078 | 0.2501 | 0.2341 |

Median miss (pp): F3 baseline 6.80 vs this model 3.53; F4 baseline 6.48
vs this model 2.84. In the saturated regime the model trails persistence
by ~3–4% (any trend model pays a little when the truth barely moves);
in the rising regime it beats persistence by ~19%. Curve-like is ahead in
these synthetic worlds because the generator's curves are logistic in
log-time, close to its functional form — the real data-generating process
is unknown, and this model's refresh-mixture and dilution decomposition
are the parts a pure curve fit lacks.

## Validity

- Output format validated on p00/p01: 55 rows, exactly `{"id", "q"}`,
  five non-decreasing quantiles in [0, 100], ≤ 65,536 bytes.
- Deterministic: two runs produce byte-identical outputs.
- Runtime: ~0.13 s per case locally (limit 8 s).
