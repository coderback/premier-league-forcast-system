# NOTES — dated decision ledger

Every hypothesis, hyperparameter and modelling decision, with its date, its number, and its
verdict. Entries are dated, never phase-labelled: the WC2026 project's "P2b" and "P2B-live"
collided in its own log and had to be disambiguated after the fact.

**WEAK-EVIDENCE** marks a choice whose evidence is thin. Every such marker appears both here and
at the code site that reads the value.

Grounding documents (in `docs/`): `RETROSPECTIVE-PLAYBOOK.md` (the WC2026 record — ~25 arms
tested, 2 adopted, with root causes) and `Model Architecture for a Premier League Forecasting
System…md` (the deep-research report — ranked candidates with published effect sizes). Where the
build brief conflicts with either, **the brief wins**; it encodes four corrections to them, all
recorded below.

---

## 2026-08-17 — Entry #1: the acceptance rule, four corrections, and the intended baseline

### The acceptance rule

Stated in `config.yaml`, embedded verbatim in every harness JSON report, and asserted
character-for-character by `tests/test_config.py`:

```
Accept a candidate change iff BOTH:
  (1) its paired-bootstrap RPS delta vs baseline on the test pool is favourable
      (95% CI excludes 0, OR P(better) >= 0.95), AND
  (2) its delta vs the Shin de-vigged market on the odds-covered subset does not degrade.
An accepted variant earns a hyperparameter retune BEFORE production wiring.
```

Why two gates: WC2026's `rsfit` arm passed gate 1 decisively (−0.0011, P=0.99) while degrading the
market scoreboard (+0.0065 → +0.0077). It improved the instrument's own distribution at the expense
of the target's. Gate 2 exists to catch exactly that.

**The two gates deliberately use different pools.** The rule scopes gate 2 to "the odds-covered
subset", so gate 1 keeps full-decade statistical power while gate 2 uses the only closing-market
benchmark that is both 100%-covered and still being published:

| Gate | Pool | Size |
|---|---|---|
| 1 — vs baseline | 2016/17–2025/26, all matches | ~3,800 |
| 2 — vs market | `AvgC*` Shin-de-vigged, 2019/20+ | ~2,660 at 100% coverage |

Pinnacle (`PSC*`) is reported as a wider historical diagnostic back to 2012/13 but is **not** the
gate — see the data survey below for why.

### The intended baseline

Per-team Dixon–Coles: attack αᵢ, defence βⱼ, fitted global home advantage γ, τ low-score correction
ρ, weighted MLE with exponential decay `w = exp(−ξ·Δdays)`. Half-life re-tuned from scratch on PL
data — **1825 days is not inherited from WC2026, and no figure suggested in conversation is
inherited either.**

### Correction 1 — the dynamic-model effect size is inflated by a baseline substitution

The report headlines Koopman & Lit (2019) as 0.2062 → 0.1982 ≈ −0.008 RPS for "going dynamic". But
0.2062 is a **static bivariate Poisson with no time decay**. The same paper's semi-dynamic,
DC-exponentially-weighted variant scores **0.2014** — and that is the model corresponding to our
intended baseline. The honest expected gain of a score-driven dynamic model over a properly
time-decayed per-team DC is therefore **≈ −0.003 RPS, not −0.008**. Most of the advertised gain is
time decay, which we are building anyway.

*Verified:* all three numbers appear in the report's own §B, so this correction is arithmetic on
the report's table rather than a disputed claim. The report's TL;DR nonetheless headlines the
−0.008.

**Consequence:** the dynamic arm is pre-registered against a *tuned time-decayed DC*, expecting
≈ −0.003. The −0.008 figure must not appear in any config comment or ledger entry.

### Correction 2 — Pitcan (2026) is load-bearing and was unverified

Source-checked directly on 2026-08-17. arXiv **2608.11505**, *"Does a Structural Model Add Anything
to the Closing Price? Calibrated forecasting, incremental information, and match leverage in the
Italian Serie A"*, **Yannik Pitcan, submitted 2026-08-11** — six days before this entry. Code and
data pipeline published at `github.com/pitcany/seriea-leverage`.

What it actually reports, from the abstract:

- **Serie A, 19 complete seasons, 7,220 matches** — not the Premier League.
- Dixon–Coles with tuned exponential decay: 53.4% accuracy, RPS **0.1972** vs the market's
  **0.1905**; paired difference **+0.0067, 95% CI [0.0046, 0.0088]**, market wins all seven test
  seasons.
- Fitted logarithmic-pool weight on the structural model **0.000** against the market, with a
  log-loss profile monotone increasing in that weight on validation and test alike — **a genuine
  boundary solution, not an optimisation artefact.**
- Refitting the same machinery to shots on target yields a variant earning weight **0.35 against
  the goals model** — it carries information the goals model lacks — **and 0.000 against the
  market.**
- The structural model is *better calibrated* than the market on the home-win margin (slope 0.995
  vs 1.103) while clearly less sharp: "the market's advantage is discrimination rather than
  honesty".

**The two weights are against different references.** The brief's compressed phrasing reads as
though 0.35 and 0.000 come from one comparison. They do not: 0.000 is model-vs-*market*, 0.35 is
shots-on-target-vs-*goals model*. `pl reproduce --paper pitcan2026` must therefore estimate **both
weights against both references** or it tests the wrong thing.

**Consequence for Arm 4 (xG as second observation channel):** the paper predicts the chance-creation
channel **passes gate 1 and leaves the market gap untouched** — adoptable under the rule, while
closing none of the information gap. That must be built and reported before Arm 4 is scheduled.

The calibration finding independently corroborates WC2026's `cal3` null from the other direction:
the market gap is information, not calibration. Two independent confirmations that no transform of
our own probabilities can close it.

### Correction 3 — RPS is not comparable across competitions

The PL baseline target is ≈ 0.196–0.206. The WC2026 production model scored 0.1835. This will look
like regression and is not: international tournament fields are more lopsided than a 20-team top
division, so outcomes are more predictable and RPS is mechanically lower. **A PL RPS is never
compared to a WC RPS in any report, chart or ledger entry.** Recorded as a comment in
`eval/metrics.py`.

### Correction 4 — data-source reality as of August 2026

FBref lost its Opta advanced-stats licence in January 2026. Its archive through 2025 is intact and
usable for backtesting; it does **not** update. Understat is therefore the **single live xG
source**. No code path may assume cross-provider xG redundancy. The coverage discontinuity at
2026-01 is flagged explicitly in the coverage report rather than smoothed over.

---

## 2026-08-16 — Data survey: football-data.co.uk, checked live

Every season file for E0–E3 was downloaded and inspected before any ingest code was written. These
facts drive the ingest and are the difference between a working one and a silently wrong one.

| Fact | Detail |
|---|---|
| URL pattern | `https://www.football-data.co.uk/mmz4281/{SSSS}/{E0..E3}.csv`, `9394`→`2526` |
| Span | E0 1993/94→2025/26 = 33 seasons; E1–E3 the same span |
| Row counts | E0: 462 (93/94, 94/95 — 22-team league), 380 thereafter. E1–E3: 552/season |
| Corpus | E0 ≈ 12,704 matches; E0–E3 ≈ 66,000 |
| **Encoding** | utf-8 (some with BOM) **except 2004/05, which is cp1252** (byte 0xa0) |
| **Dates** | always dayfirst, but **mixed 2-digit and 4-digit years** |
| **Round column** | **none exists** — the calendar structure must be derived |
| `Time` column | only from 2019/20 → **the walk-forward barrier is date-granular, not kickoff** |
| `HST`/`AST` | shots on target from 2000/01 — a pre-xG chance-creation channel, free in the spine |

Odds column availability, which is what settles the gate-2 choice:

| Family | Span | Note |
|---|---|---|
| `WHH/WHD/WHA` | 2000/01→2024/25 | William Hill, pre-close |
| `B365H/…` | 2002/03→ | pre-close |
| `BbAvH/…` | 2005/06–2018/19 | Betbrain average — **pre-close, not closing** |
| `PSCH/…` | 2012/13→ | Pinnacle **closing**; 100% covered **until 2026-01-08, then absent** |
| `AvgCH/…` | 2019/20→ | market-average **closing**; **100% covered, still live** |

**Pinnacle's feed stops dead mid-season.** Present for every match through 2026-01-08, entirely
absent from 2026-01-17 to the end of 2025/26 (55% season coverage). It is therefore unusable as the
forward-looking benchmark for a model that must score 2026/27 live, despite being the sharper line.
`AvgC*` is the gate-2 benchmark; `PSC*` is a historical diagnostic with its termination flagged.

There are now **two independent data discontinuities at 2026-01**: FBref's Opta licence loss and
Pinnacle leaving this feed. Both are flagged in the coverage report.

*Also verified:* Understat's **league** pages no longer embed their JSON payload in the initial
HTML, but **match** pages still do. The xG loader needs a match-id discovery path — a spike before
Arm 4, not a blocker.

---

## 2026-08-17 — Expectation-setting from the report, with two adjustments

**The 0.196–0.206 band is EPL-grounded; the +0.006–0.007 gap is not.** The band traces to Koopman &
Lit's 2,660 EPL matches (static biv-Poisson 0.2062, semi-dynamic DC-weighted 0.2014, dynamic
0.1982) and Constantinou's EPL work (0.195–0.203, range 0.184–0.213). The gap comes from Pitcan's
Serie A (+0.0067), corroborated by WC2026's own +0.0061 — two competitions, neither the PL. Trust
the level; treat the gap as a strong prior rather than a measured PL quantity.

**Our baseline is Koopman & Lit's semi-dynamic row, 0.2014.** A per-team DC with exponential decay
is precisely that specification. So landing at **0.201–0.206 is on target**, and **landing at 0.196
from a static model is suspicious** — it triggers the leakage hunt before any celebration. This is
recorded now, before the number exists, so it cannot be rationalised afterwards.

**Arm 9's premise is probably false for the PL.** The brief premises Weibull + Frank copula on
*negative* low-score dependence, inherited from the WC corpus's fitted ρ < 0. But the report's §A
notes the five-league study found dependence **positive in four leagues, negative only in Ligue 1**.
The Dixon-Coles fit therefore reports ρ with its sign by era as a free falsifier gating whether Arm 9
is ever scheduled. Cost: zero — the DC fit produces ρ anyway.

---

## 2026-08-17 — Repo skeleton, config, and the magic-number check

Repo skeleton, `config.yaml` carrying the acceptance rule plus deliberately empty tunable sections,
this ledger, and the magic-number test.

**`data.matchweek_gap_days = 3`** — **WEAK-EVIDENCE**: chosen by inspection of the fixture calendar,
not tuned. Within a (division, season), a new matchweek block starts when the gap from the previous
match date exceeds this. 3 days groups a Fri–Mon round into one block while leaving a midweek round
as its own, yielding ~38–42 blocks/season. The per-team round index is kept as a documented
sensitivity, not the primary derivation.

**`seed = 20260822`** — the 2026/27 opening date. Chosen only to be memorable and fixed.

**The magic-number test** (`tests/test_no_magic_numbers.py`) is new; WC2026 had no equivalent. It
AST-walks `model/` and `eval/` and fails on any numeric literal that is not a structural value
(`{0, 1, 2, 3, -1}` — indices, `ndim` checks, arithmetic identities), a literal marked `# MATH:`, or
a documented module-level `UPPER_CASE` constant. It carries eight self-tests against synthetic
clean and dirty sources, so it has teeth while `model/` and `eval/` are still empty rather than
passing vacuously.

One tension recorded in advance: the brief requires `eval/metrics.py` to be ported **verbatim**, and
that file's `paired_delta` carries `n_boot=10000, seed=0` defaults — bare literals the checker will
flag. Resolution: a named `VERBATIM_PORTS` exemption carrying its reason, plus a companion test
asserting every call site passes `n_boot`/`seed` explicitly from config. Both non-negotiables
survive; neither is quietly dropped.

**DoD met:** `uv run pytest -q` green, 22 tests.

---

## 2026-08-17 — Data spine: the football-data.co.uk ingest

`pl ingest` loads **66,908 matches** across E0–E3, 1993-08-14 → 2026-05-24, 33 seasons, 116 clubs.
E0 is 12,704 (462 + 462 + 31 × 380), E1 18,216, E2 18,064, E3 17,924. Coverage report at
`output/coverage.json`; corpus cached as parquet.

### Four source quirks found by building against the live data, not by assuming

**1. The source substitutes another competition's file for an unpublished season — silently.**
Requesting `mmz4281/2627/E0.csv` on 2026-08-17 returns **HTTP 200 and a valid CSV of the National
League** (`Div` = `EC`). `raise_for_status()` does not catch it (the underlying status is 300
"Multiple Choices", not an error), the body parses cleanly, and every column is where it should
be. Ingested unguarded, the 2026/27 Premier League would have consisted of Altrincham, Boreham
Wood and Hartlepool. E1 and E2 returned an HTML page instead, which is the *benign* version of the
same behaviour.

Three guards, all in `data/football_data.py`: the status must be exactly 200; the body must start
with `Div`; and **the served division must equal the requested one**, checked before the file
reaches the cache and again on read. A `MissingSeasonError` distinguishes "not published yet"
(normal near a season boundary — all four 2026/27 files are legitimately absent today) from
"published but wrong".

This is the same failure class as WC2026's two worst bugs — a silent key miss that produces a
well-formed, wrong result — arriving at the file level rather than the config level.

**2. The date-gap matchweek rule failed on contact with the real calendar, and is gone.** The
plan's derivation started a new block whenever consecutive match dates were more than 3 days
apart. Measured on E0 it produced **19–31 blocks per season instead of ~38**: English league
football has matches on most days, so the rule chains transitively and merges whole months. Every
prediction in a merged block would have been made without weeks of available information.

Replaced by **one barrier per distinct match date**, which needs no threshold. E0 averages 106
match dates per season (min 95, max 135), giving **1,153 barriers across the ten-season test
span** — exact, deterministic, and affordable under the planned warm-start. `matchweek_gap_days`
is deleted from config: the ingest now has no WEAK-EVIDENCE tunable at all.

**3. A reconstructed "matchweek" is not recoverable from results, so it is not claimed.** A first
attempt assigned each match `max(games played by either side) + 1`; it inflated E0 seasons to
**40–53 rounds instead of 38**, because rearranged fixtures ratchet both sides forward. There is no
correct single round number for a match whose two teams have played different numbers of games.
Replaced by two exact numbers — `home_match_index` and `away_match_index`, each side's own match
count. Reporting only; the barrier is always the matchday.

Validation of the replacement: across all four divisions, every deviation from the standard season
length is real history — E0/E3 1993-94 and 1994-95 at 42 (22-team divisions), and **E2 at 36 and
E3 at 37 in 2019-20**, which is League One and League Two being abandoned by COVID rather than
completed. E0 2019-20 is 38, correctly, because the Premier League resumed in June.

**4. Structural quirks, all verified across the whole corpus.** 2004/05 is cp1252 in **all four
divisions** (not just E0 as first surveyed) — decoding falls back and records the codec used. 17
files across 1993/94–2004/05 carry rows wider than their own header; every extra field was checked
across the corpus and is empty, so they are absorbed and dropped, and a non-empty one raises rather
than being discarded. Dates mix two- and four-digit years and are parsed with both explicit
formats, never inferred.

### Column availability confirmed on the full corpus

Half-time scores from 1995-96; match statistics, referee and odds from 2000-01; kickoff time only
from 2019-20 — which is why the barrier is date-granular rather than kickoff-granular.

### Team roster

116 distinct spellings, curated into a **closed roster** (`data/static/team_roster.yaml`); an
unrecognised name raises. The source is internally consistent — `find_near_duplicates` reports
nothing — so canonical names are the source's own spellings and the alias file is empty until
Understat and FPL arrive. `Wimbledon`, `Milton Keynes Dons` and `AFC Wimbledon` are deliberately
three separate entries: fuzzy matching would merge them and splice together the histories of
different clubs.

**DoD met:** `pl ingest` produces a validated frame and coverage report; an unmapped team name
raises; row-count floors hold for every completed division-season. 66 tests green.

---

## 2026-08-17 — Scoring rules, Diebold–Mariano, and FDR control

`eval/metrics.py`. RPS, log loss, Brier, `skill`, `outcome_from_scores`, `summary` and
`paired_delta` are ported **verbatim** from `wc2026/eval/metrics.py`. The ported code lines are
untouched — only the module docstring is extended and new functions appended — so a diff against
the source stays readable.

### Byte-identity, established against the source's own code

`tests/fixtures/wc2026_metrics_golden.json` was generated by **running the WC2026 implementation**
under that project's own venv (`tests/fixtures/make_wc2026_golden.py`, seed 20260822, 500 matches,
three Dirichlet forecast families). The test asserts exact equality — `==`, not `approx` — on every
per-match array, every pooled scalar, and both `paired_delta` invocations including the bootstrap
CI and `p_a_better`. The bootstrap reproduces exactly because the RNG stream is seeded and the
resampling is identical.

This matters because the port's entire value is that a number computed here is directly comparable
with a number computed there. Silent drift would invalidate every cross-reference in this ledger
without failing anything.

### A correction to how the 0.2222 floor is usually quoted

The uniform forecast does **not** score 0.2222 per match. With C = (1/3, 2/3) it scores **5/18 ≈
0.2778** on a home or away win and **1/9 ≈ 0.1111** on a draw — a draw is the outcome uniform is
least wrong about. 2/9 = 0.2222 is the *expectation* over a balanced outcome mix. A first pass at
the test asserted the per-match value and failed correctly. Recorded because the pooled number is
quoted often enough that it is easy to mistake for a per-match constant; the golden fixture's own
uniform score is 0.2244, reflecting its outcome mix rather than any error.

### Diebold–Mariano

HAC (Newey–West, Bartlett weights, automatic bandwidth `floor(4(n/100)^(2/9))`) with the
Harvey–Leybourne–Newbold small-sample correction, tested against `t(n-1)`. The **sign convention
matches `paired_delta`**: negative favours arm A. A flip here would invert every verdict in the
ledger, so it is asserted directly.

Reported *alongside* the paired bootstrap, never instead of it — agreement between two tests with
different assumptions is the robustness signal. Diebold's own caveat (2015, NBER w18391) is that
DM compares *forecasts*, not *models*, and can favour the simpler benchmark; that bias is
conservative under a two-gate rule, which is why it is acceptable here.

The HAC estimator earns its place: per-match RPS differences are serially correlated because form
persists across a team's fixtures, so the i.i.d. variance understates the standard error and
overstates significance. Tested on an AR(1) series.

### Benjamini–Hochberg FDR

Step-up control across a family of arms, with **`alpha` a required keyword argument, not a
default** — the false-discovery rate the project tolerates is a decision that belongs in
`config.yaml`, not in a function signature. With ~25 arms in the WC2026 ledger and a comparable
plan here, uncorrected testing is how a two-gate rule eventually passes a false positive by chance.
A test asserts 25 arms of pure noise yield at most one discovery.

### The magic-number exemption is per-literal, not per-file

`paired_delta` keeps its WC2026 defaults (`n_boot=10000, seed=0`), which the checker would flag.
Rather than exempt the project's most load-bearing module wholesale, `VERBATIM_PORTS` now licenses
**named literals** — `{1e-15, 0.5, 10000, 2.5, 97.5}` — so a *new* magic number in `metrics.py`
still fails. Two further guards: a test that the exemption licenses nothing the file no longer
contains (stale holes get removed), and a companion AST test asserting **every `paired_delta` call
site outside the module passes `n_boot` and `seed` explicitly**, so nothing silently inherits the
defaults instead of reading config.

**DoD met:** byte-identity against the WC2026 fixture; DM and the bootstrap agree in sign on the
fixture pair. 93 tests green.

---

## 2026-08-17 — Odds and de-vig

`data/odds.py`. Shin and proportional de-vig ported verbatim from WC2026; both appear in every
report. Shin is primary (Štrumbelj 2016: lowest RPS across 412 bookmaker/competition pairs; Koning
& Zijm 2023: unbiased for the EPL specifically), with proportional retained as a sensitivity
because Shin's theoretical case is contested (Whelan: "relatively weak").

### The gate-2 benchmark, confirmed on measured coverage

| Family | Settlement | E0 priced | Span | Role |
|---|---|---:|---|---|
| `avg_closing` (`AvgC*`) | closing | 2,660 | 2019-20 → live | **gate 2** |
| `pinnacle_closing` (`PSC*`) | closing | 5,150 | 2012-13 → **2026-01-08** | diagnostic |
| `betbrain_avg` (`BbAv*`) | **pre-close** | — | 2005-06 → 2018-19 | not comparable |

`avg_closing` covers exactly **2,660** E0 matches from 2019-20 — the figure the plan committed to,
now measured rather than estimated. Pinnacle reaches further (3,630 of the 3,800 test-decade
matches vs 2,660) and is the sharper line, but its last priced match is **2026-01-08**; a benchmark
that dies mid-season cannot judge a model that must score 2026/27. It is reported as a diagnostic
so a disagreement between the two is visible rather than hidden.

**Settlement timing is recorded on every family and mixing it is refused** (`assert_comparable`).
A pool that is closing odds for one era and pre-close for another silently changes what the gate
measures partway through — which would read as a model effect. This is the odds-side analogue of
the division-substitution guard: a well-formed benchmark that is quietly the wrong benchmark.

### Shin's correction measured on real data, not assumed

On the 2,660 gate matches, Shin moves probability from longshots to favourites, monotonically
across the whole probability range:

| Proportional p | mean(Shin − proportional) |
|---|---:|
| < 0.05 | **−0.0084** |
| 0.05–0.10 | −0.0077 |
| 0.10–0.20 | −0.0056 |
| 0.20–0.35 | −0.0021 |
| 0.35–0.50 | +0.0028 |
| 0.50–0.70 | +0.0082 |
| > 0.70 | **+0.0134** |

Exactly the favourite–longshot correction Shin exists to make, and the monotonicity is asserted as
a test rather than the direction alone. On the most extreme fixture in the corpus (odds 1.04 /
19.42 / 40.86) Shin cuts the longshot's probability from 0.0236 to 0.0155 — a third. Mean absolute
difference between the methods is 0.0044, so the de-vig choice is a real decision, not a formality.

### Two source behaviours found and handled

**`0.0` is the source's "no price" sentinel, not a price.** Six rows corpus-wide carry
`B365H = 0.0` alongside plausible draw and away prices. They are counted as invalid and excluded —
never imputed, never silently dropped from the count. The ported de-vig functions keep their strict
contract (finite, > 1.0) precisely so a sentinel cannot pass for a price; sanitising happens before
they are called.

**Negative overrounds are real.** Averaging decimal odds across bookmakers can push the implied sum
below 1, and the lower divisions contain such rows (`avg_closing` min −0.0446, `pinnacle_closing`
min −0.0665). **E0 contains none** — the gate pool is unaffected — but the sub-fair branch is live
code for the multi-tier work, and is tested.

Also noted, not acted on: `william_hill` stops after 2025-03-27, a third feed discontinuity. It is
not a benchmark here, so it is recorded and left alone.

**DoD met:** de-vigged probabilities sum to 1.0 within 1e-9 (observed worst 2.2e-16); Shin ≠
proportional on a favourite–longshot fixture, with the direction and monotonicity asserted on real
data. 123 tests green.

---

## 2026-08-17 — Walk-forward splitter: the yardstick exists

`eval/backtest.py`. Rolling-origin by matchday: for each distinct match date in the test span,
train on everything strictly before it, predict that day, roll forward.

### The yardstick, measured

| Span | Barriers | Matches | First → last | Train rows |
|---|---:|---:|---|---|
| test `2016-17..2025-26` | **1,153** | **3,800** | 2016-08-13 → 2026-05-24 | 8,904 → 12,694 |
| sensitivity `2006-07..2015-16` | 995 | 3,800 | 2006-08-19 → 2016-05-17 | 5,104 → 8,903 |

Gate 1 pool 3,800; gate 2 subset 2,660 — both exactly the figures the plan committed to. Maximum
matches per barrier is 10, correct for a 20-team round. The earliest barrier already carries 8,904
training matches, so the decade-long test span never fits on thin history.

### Leak-freedom is structural, in three layers

1. **`assert_no_leakage` runs on every split**, not a sample. It checks more than the two frames'
   relative order: training must not reach the barrier, *and* the test set must start exactly at
   it. A split can be internally consistent and still be built at the wrong barrier, and only the
   second check catches that.
2. **`training_frame(source, barrier)` is the only way to pull in extra training data** — a second
   division for the multi-tier fit, or an external feed. Routing every such join through one
   barrier-checked function means extra data cannot bypass the barrier merely because it did not
   come from the prediction frame.
3. **`validate_splits` adds cross-split invariants**: barriers strictly increasing, test slices
   non-overlapping, no fit dated after its own barrier. These catch a splitter bug that leaves
   each split individually valid.

The definition-of-done check builds a deliberately corrupted split — barrier moved forward one
matchday while the training prefix stays put, the exact shape of an off-by-one — and asserts
`LeakageError` fires. `LeakageError` subclasses `AssertionError` by intent: it is a violated
invariant, not a recoverable condition, and nothing may catch it.

### Splits are integers, not frames

Because the corpus is date-sorted, a barrier makes training a *prefix* and the test set a
*contiguous slice*. A split is therefore three integers plus a timestamp. The whole ten-season walk
costs nothing to hold, and `walk_forward` returns a materialised list rather than a generator so
that **every arm in a comparison replays literally the same splits** — structural rather than
conventional. This is the same guarantee the WC2026 harness got from its shared per-edition
precomputation, obtained more cheaply.

### The refit-cadence seam

Prediction always happens at every matchday; *fitting* may happen less often. `refit_every = n`
reuses the most recent fit at or before each barrier: 1,153 fits at `n=1`, 289 at `n=4`, 31 at
`n=38`. Leak-free either way, because a reused fit saw strictly less data, never more — asserted
across cadences 1, 2, 5 and 100.

The seam is inert at its default: `refit_every = 1` makes every split fit at its own barrier, and
a test asserts the cadence changes only *which parameters* are used, never *what is predicted*
(identical barriers and test slices across cadences). Its purpose is to make "does refitting this
often actually matter?" measurable rather than assumed — the WC2026 project's live-update channel
returned null three separate times, and that was only knowable because it was measured.

**DoD met:** `LeakageError` fires on a deliberately corrupted split; the walk is deterministic
(`walk_forward(df) == walk_forward(df)`); all 1,153 real splits pass `validate_splits`. 154 tests
green.
