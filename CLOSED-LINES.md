# Closed lines: read this before proposing an arm

*Consolidated 2026-09-23 from `NOTES.md`. Every line below has been tested and closed, with the
number that closed it. This file exists because the ledger is 6,700 lines and twice in one session
work was re-derived that it already held — in both cases the existing version was the better one.*

**How to use it.** Before proposing any change, find the nearest line here and state in one sentence
why the proposal is not that. A difference in label is not a difference in mechanism. If no
distinction survives a hostile reading, the idea is already answered.

---

## The standing conclusion

Production is the plain Dixon-Coles model the project started with. **Thirteen arms have been
attempted; one passed, under an older two-gate rule, and was never wired.**

The gap to the de-vigged closing line is **+0.00824 RPS** (CI [+0.00582, +0.01074], n=2,660,
`avg_closing`, test decade). It is approximately the documented floor for a goals-only public-data
model — Pitcan (2026) gets +0.0067 on Serie A with an overlapping interval, and this project
reproduced that paper on its own corpus on 2026-08-17 (`pl reproduce --paper pitcan2026`).

**The market's edge is sharpness, not calibration.** Our model is *better* calibrated than the
market. That is why anything that only reshapes existing probabilities cannot work.

The literature's only two documented routes to parity: **feed the model the closing line**, or
**feed it team news the line already contains**. Both are decisions, not techniques.

---

## The resolution floor — check this before anything else

| gate configuration | minimum detectable effect |
|---|---|
| gate 1 bootstrap alone | 0.00055 RPS |
| all four gates, family of 1 | 0.0004 |
| all four gates, family of 3 | 0.0008 |
| all four gates, family of 5 | 0.0010 |

Measured by injecting a synthetic improvement with the per-match variance of a real arm (SD 0.01757)
and running the project's own gate machinery. **A candidate must deliver roughly a tenth of the
entire remaining headroom in a single arm.** Arms 11 and 12 died here with real, mechanistically
clean effects of −0.00059 and −0.00030.

Switching the gate metric to the log score does **not** help: as a share of the gap, RPS asks 9.7%
and the log score 8.2–10.7%, non-monotone in family size, i.e. noise.

---

## Closed: model specification

| line | verdict | the number |
|---|---|---|
| Elo vs per-team attack/defence | rejected | +0.0009, wrong-signed vs prediction, replicated on decade two |
| Structural home advantage — time trend, empty stadiums | rejected, all 3 variants | BH rejects 0 of 3 |
| xG / shots-on-target as a **pooled observation channel** | null | xG earned 0.457 pool weight and bought 0.0004 RPS |
| Dixon-Coles + LightGBM hybrids and ensembles | rejected | parent error correlation 0.99; blend weight exactly 0 at 91.3% of barriers |
| Rest, fixture congestion, European-commitment flags | rejected, all 4 | sign-flips between decades; the best one was shrinkage wearing a fatigue label |
| Joint multi-division (tiered) fits | rejected | market gap degraded +0.0082 → +0.0097 |
| Weibull marginals + Frank copula | rejected | +0.00011, and exactly 0.00000 on the sensitivity decade |
| **General hierarchical shrinkage of team parameters** | **CLOSED** | gain concentrates entirely in promoted fixtures; "there is no general shrinkage effect on this corpus" |
| **Promoted-club mispricing as strength estimation** | **CLOSED** | 4 mechanisms failed. "The mean can be fixed. The ordering cannot" — it needs information the project does not have |
| **Recalibration of our own probabilities** | **CLOSED** | temperature and vector scaling both degrade out-of-sample on both decades, intervals excluding zero on the worse side |

---

## Closed: data and features

| line | verdict | the number |
|---|---|---|
| **Half-time goals** | **CLOSED** | team-specific 1H/2H variance measured at **zero** — excess variance negative, p = 0.881. Disattenuated 1H/2H correlation 1.03: one parameter with noise |
| **Referee as a 1X2 feature** | **CLOSED** | between-referee variance in home-win residual is *below* the permutation null, p = 0.850. Largest effect the corpus could hide ≈ 0.00027 RPS, half the floor |
| Asian-handicap / over-under odds as a market family | no gain | reach back less far than the 1X2 families already parsed; add no sensitivity-span coverage; de-vigged AH+OU is *less* sharp than the direct 1X2 (0.17970 vs 0.17937) |
| **Backward xG (2006–2016)** | **CLOSED** | no free per-match xG or shot-location series exists. Earliest free shot coordinates are 2014-15, which only overlaps the Understat mirror already held |
| Transfermarkt (direct or via mirrors) | **blocked on licence** | ToS §11.1 forbids automated collection *and* use for machine learning; §44b UrhG reserved |

**Referee → cards/fouls is NOT closed** — it is real and large (excess SD 0.319 cards/match after
team control, p < 0.0001, odd/even stability r = 0.687). But cards have no market in this corpus, so
gate 2 is permanently NOT EVALUABLE for them.

---

## Not closed — blocked, which is different

**Arm 7: lineups and player availability.** Never refuted, only blocked on data. The ledger's own
assessment: *"it is the only one aimed at the information itself rather than at re-slicing goals."*
A source covering both decades under one schema now appears to exist (11v11.com, verified at both
ends) but is **blocked on a terms-of-use judgement**, not on availability. Nothing has been scraped.

---

## Traps that have already cost this project

**The collection-era trap.** A date range can be real while the *density* is not. `transfermarkt-datasets`
advertises transfers from 1993; it holds 13 rows touching nine of the largest clubs in the world in
2006 against 356 in 2024, because it only carries currently-profiled players and thins backwards.
Net spend computed from it would look like a signal. This is the same failure as the FPL
availability flags, whose population rate quadrupled inside a season.

**robots.txt is not the licence.** Transfermarkt's robots.txt is `Allow: /` and its ToS forbids
exactly what robots.txt appears to permit.

**Dataset titles overstate range.** A CC0 set titled "2014-present", updated six days before it was
checked, stopped in 2019.

**Binned decomposition differences are not findings.** A ~0.0005 resolution or reliability gap is
inside this corpus's grid noise. Standing rule: *a reliability or resolution difference is not
evidence until it survives a second decade.*

**Never select a hyperparameter on an evaluation span.** Dixon-Coles at a 365-day half-life scores
0.00107 better on the test decade — and 0.00001 better on the sensitivity decade. The tuning span
returned UNRESOLVED and 730 stands.

---

## Dual-decade rule — the filter that kills most candidates first

Gate 4 requires the effect to hold on **both** 2006-07→2015-16 and 2016-17→2025-26. A single-span
run cannot accept anything. Apply this before any other test:

```
goals, teams, dates          1993 →   both decades
shots/SoT/corners/cards      2000 →   both decades
referee                      2000 →   both decades
half-time goals              1995 →   both decades  (but closed above)
bet365 1X2 odds              2002 →   both decades  (100% / 100%)
avg_closing 1X2 odds      2019/20 →   TEST ONLY — 0% on the sensitivity decade
kickoff time              2019/20 →   TEST ONLY
xG                     2015–2024 →   NEITHER decade fully; un-gateable
```

**Note for gate 2:** `avg_closing` covers the sensitivity decade at 0%, but `bet365` and
`betbrain_avg` cover it at 100%. Gate 2 there is a *policy* question about mixing settlement
timings, not a coverage one.
