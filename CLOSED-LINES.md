# Closed lines: read this before proposing an arm

*Consolidated 2026-09-23 from `NOTES.md`. Every line below has been tested and closed, with the
number that closed it. This file exists because the ledger is 6,700 lines and twice in one session
work was re-derived that it already held — in both cases the existing version was the better one.*

**How to use it.** Before proposing any change, find the nearest line here and state in one sentence
why the proposal is not that. A difference in label is not a difference in mechanism. If no
distinction survives a hostile reading, the idea is already answered.

---

## The standing conclusion

**Updated 2026-09-23.** Production is Dixon-Coles plus a trailing shots-on-target form term
(`dc+sot-form`), the first arm to pass all four gates and the first seam ever wired. Before it,
thirteen arms had been attempted and one had passed, under an older two-gate rule, and was never
wired. **A new arm is now judged against `dc+sot-form`, not against `dixon-coles`**: the
`dixon-coles` arm stays pinned as the historical baseline every earlier delta was measured against.

The gap to the de-vigged closing line was **+0.00824 RPS** for plain Dixon-Coles (CI [+0.00582,
+0.01074], n=2,660, `avg_closing`, test decade) and is **+0.0050** for `dc+sot-form`. Everything
below was measured against the plain model and is read in that light.

**Updated 2026-11-18.** The independent review splits the +0.0050 three ways:
- about **0.0015** is strength knowledge the market's prices of earlier matches carry and goals and
  shots do not;
- about **0.0022** is match news already in Friday's price;
- about **0.0013** arrives after Friday.

2026-27 to date sits at +0.0055 after the cold-start correction (+0.0125 as shipped).

**Decided 2026-11-18 by the owner: two tracks, and the Friday line as the target.**
- **The independent track** reads match facts only, and production is always its best arm.
- **The market-informed track** reads bookmaker prices. It must beat independent production under
  the same four gates and never replaces it.
- **Gate 2 is scored against bet365's Friday line**, which covers both decades.
  - The model forecasts without team news, so this is the information-matched market.
  - The close stays reported for every arm.
  - The close is gate 2 for an arm that reads its own match's pre-close prices, because its input
    cannot also be its yardstick.

Corrected production's gaps, test decade:

| market | gap |
|---|---|
| bet365 Friday (the gate) | +0.0041 |
| Friday market average | +0.0035 |
| the close | +0.0049 |

On the sensitivity decade the gap to bet365 Friday is +0.0019, the first market comparison that
decade has had. **A smaller headline is a fairer stick, not a better model.**

**The market-informed track's model, accepted 2026-11-18: `odds-ratings`.** It fits ratings to the
goal rates implied by earlier matches' prices and never reads the forecast match's own.

| | test decade | sensitivity decade |
|---|---|---|
| vs `dc+sot-form` | −0.0023, CI [−0.0037, −0.0008] (by season [−0.0038, −0.0007]) | −0.0008, P 0.889 |
| gap to bet365 Friday | +0.0019 | +0.0010 |
| gap to the close | +0.0035 | |

It passes all four gates, never becomes production, and still sits behind the market.

A market-informed candidate is now judged against `odds-ratings` as well as against independent
production. **"Use the market's view of strength" is answered;** what remains open on that track is
match-specific news.

The original gap was approximately the documented floor for a goals-only public-data
model — Pitcan (2026) gets +0.0067 on Serie A with an overlapping interval, and this project
reproduced that paper on its own corpus on 2026-08-17 (`pl reproduce --paper pitcan2026`).

**The market's edge is sharpness, not calibration.** Our model is *better* calibrated than the
market. That is why anything that only reshapes existing probabilities cannot work.

The literature's only two documented routes to parity: **feed the model the closing line**, or
**feed it team news the line already contains**. Both are decisions, not techniques.

---

## The resolution floor — check this before anything else

| gate configuration | MDE, passes half the time | MDE, passes 80% of the time |
|---|---|---|
| gate 1 bootstrap alone | 0.0005 RPS | 0.00075 |
| all four gates, family of 1 | 0.0006 | 0.00085 |
| all four gates, family of 3 | 0.0007 | 0.00095 |
| all four gates, family of 5 | 0.00075 | 0.0010 |

Measured by injecting a synthetic improvement with the per-match variance of a real arm (SD 0.01757)
and running the project's own gate machinery, 1,000 draws a point (re-measured 2026-11-17 against
`dc+sot-form`; an earlier family-1 figure of 0.0004 was impossible, below gate 1 alone). **The MDE
is about 0.040 × the per-match differential's SD at the 50% point**, and the baseline's own losses
do not enter it. An arm that moves forecasts a lot per match needs a proportionally larger gain:
`dc+sot-form` against `dixon-coles` has SD 0.0315, which puts its own family-3 MDE at 0.00125. **A candidate must deliver roughly a tenth of the
entire remaining headroom in a single arm.** Arms 11 and 12 died here with real, mechanistically
clean effects of −0.00059 and −0.00030.

**Updated 2026-11-18: these figures assume the gain is spread evenly across matches.** They were
calibrated by injecting independent per-match noise. A gain that arrives in team-season blocks
(a club whose level moves between seasons, which is how `dc+sot-form` earns its keep) is up to
**twice** as hard to detect on the test decade: resampling seasons instead of matches doubles
`dc+sot-form`'s standard error (0.00051 → 0.00106).
- `dc+sot-form`'s interval is [−0.0032, −0.0012] by match and [−0.0045, −0.0004] by season; it still
  excludes zero, so its acceptance stands.
- No closed line reopens. Every one failed the easier, by-match bar.
- `pl compare` now prints the season-clustered interval beside gate 1. Quote both.

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
| **Promoted clubs' own Championship shots as form** (`dc+sot-form+e1`) | rejected, null | orders promoted clubs the right way on both decades (−0.00036 / −0.00017 on their fixtures) but −0.0001 pooled, P 0.70 / 0.84. The ordering CAN be moved; this source moves it a tenth of the detectable amount |
| **Dropping promoted clubs' stale E0 form** (`dc+sot-form-recent`) | withdrawn at tuning check | +0.00003: stale Premier League form is a weak but real prior on a returning club's level. Read narrowly: it replaced stale 20-match windows with 1-3 match windows at full weight, so it tested stale form against very noisy form, not against none |
| **Mixing production with the market line** (log pool) | **CLOSED** | weight on the model 0.00-0.13 against the close and 0.02-0.19 against the Friday line, season held out; the pool is *worse* than the line alone (+0.00017, +0.00014). Re-checked 2026-11-18 on corrected production |
| **The model's disagreement with the market as a signal** | **CLOSED** | it does not predict the Friday-to-close move (slope −0.0002) or the result beyond the close (0.024 ± 0.122). The model holds no information the market lacks |
| **Recalibration of our own probabilities** | **CLOSED** | temperature and vector scaling both degrade out-of-sample on both decades, intervals excluding zero on the worse side |

---

## Closed: data and features

| line | verdict | the number |
|---|---|---|
| **Half-time goals** | **CLOSED** | team-specific 1H/2H variance measured at **zero** — excess variance negative, p = 0.881. Disattenuated 1H/2H correlation 1.03: one parameter with noise |
| **Referee as a 1X2 feature** | **CLOSED** | between-referee variance in home-win residual is *below* the permutation null, p = 0.850. Largest effect the corpus could hide ≈ 0.00027 RPS, half the floor |
| **Totals (over/under 2.5) and Asian handicap as softer targets** | **not softer** | over/under 2.5 Brier: model 0.24352, closing market 0.24032. The model has 55% of the market's skill on totals against 86% on 1X2, and it under-predicts goals. Independent review's measurement, 2026-11-18, not re-run here |
| Asian-handicap / over-under odds as a market family | no gain | reach back less far than the 1X2 families already parsed; add no sensitivity-span coverage; de-vigged AH+OU is *less* sharp than the direct 1X2 (0.17970 vs 0.17937) |
| **Backward xG (2006–2016)** | **CLOSED** | no free per-match xG or shot-location series exists. Earliest free shot coordinates are 2014-15, which only overlaps the Understat mirror already held |
| **Opponent-adjusted shots-on-target form** | **withdrawn at pre-flight** | correlates 0.986 with the raw term once the 2000-01 warm-up is excluded (0.990 on full windows); 0.949 pooled only because of that warm-up |
| More match counts on top of SoT form (total shots, corners, all counts) | **not eligible** | already seen on both evaluation decades; adding counts made the sensitivity decade worse. Reopen only on live seasons |
| **Managerial change from Wikipedia season tables** | **withdrawn at pre-flight** | 2 of 10 spot-checked dates disagree with the clubs' own lists, both straddling a league match: the tables record announcements, not when a manager took charge |
| **Summer net spend from Wikipedia transfer lists** | **withdrawn at pre-flight** | disclosed fees per club-season 3.33 vs 4.77, ratio 0.698 against a 0.70 bar; undisclosed share 57% → 47%; recent "Undisclosed" rows carry sort-key amounts (1,444 vs 2), a one-decade leak |
| FBref / Sports Reference | **blocked on terms** | data-use page bars use "supporting machine learning methods used to predict ... or score inputs"; custom data from $5,000 |
| Fantasy Premier League, live or archived (vaastav, FPL-Core-Insights) | **blocked on terms** | 2026/27 T&C 28(d) bars automated extraction; clause 29 reserves Game data; live value of flags ~0.0003 RPS anyway |
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

**Updated 2026-11-18: half unblocked.** `pssguy/epldata` (MIT) records starters and substitution
minutes for every Premier League match from 1992-93 to 2017-18 and passes its structure check (two
substitution records in error, accepted by the owner). A registered falsifier on 2012-18 passes:
regulars newly missing from the eleven move the Friday-to-close line at r = −0.182, known absences do
not (−0.033), and the placebo is null. **What still blocks the arm is August 2018 onward**, where no
permitted source is known. FBref, Transfermarkt and FPL are closed on their own terms (see the data
table below). worldfootball.net allows personal use but disallows AI agents in robots.txt.

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

**The source redefines its own series.**
- **Shots on target change definition in 2013-14:** E0 14.24 → 8.92 a match while total shots stay
  flat, and by different factors in each division (E1 ×0.77, E2 ×0.84). Any raw-count shots term
  pools two scales across the break, and E0/E1 comparisons pool two regimes.
- **The market average changes its bookmakers in 2024-26:** its margin goes 0.04 → 0.057 → 0.061.
  It still agrees with bet365 once de-vigged.

Both are in `KNOWN_DISCONTINUITIES`. Check that list before reading a series across seasons.

**A cache keyed on identity serves corrected data stale.** The walk cache's fingerprint now hashes
the content of every match a walk reads, not only which matches they were. A cache made before
2026-11-18 must not be trusted.

**Never select a hyperparameter on an evaluation span.** Dixon-Coles at a 365-day half-life scores
0.00107 better on the test decade — and 0.00001 better on the sensitivity decade. The tuning span
returned UNRESOLVED and 730 stands.

---

## Dual-decade rule — the filter that kills most candidates first

Gate 4 requires the effect to hold on **both** 2006-07→2015-16 and 2016-17→2025-26. A single-span
run cannot accept anything. Apply this before any other test:

```
goals, teams, dates          1993 →   both decades
shots/SoT/corners/cards      2000 →   both decades  (SoT redefined 2013-14: E0 x0.63)
referee                      2000 →   both decades
half-time goals              1995 →   both decades  (but closed above)
bet365 1X2 odds              2002 →   both decades  (100% / 100%)
avg_closing 1X2 odds      2019/20 →   TEST ONLY — 0% on the sensitivity decade
kickoff time              2019/20 →   TEST ONLY
xG                     2015–2024 →   NEITHER decade fully; un-gateable
```

**Note for gate 2:** since 2026-11-18 gate 2 is scored against `bet365`'s Friday line, which covers
both decades at 100%, so it is evaluable on the sensitivity decade for the first time. `avg_closing`
(0% on that decade) is reported, not gated. Gate 2 compares an arm's gap with the baseline's on the
same rows, so the benchmark cancels out of the verdict. Changing it changed coverage, not which arms
pass.
