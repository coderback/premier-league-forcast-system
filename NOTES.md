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

---

## 2026-08-17 — MILESTONE: the harness runs end to end, before any model exists

`pl compare --arms uniform,home-always` emits a full JSON report with the acceptance rule embedded
verbatim, paired deltas with CIs, DM statistics, the market gate, coverage, and calibration slices.
This was the point of the phase ordering: the WC2026 project built model machinery against a
yardstick that could not resolve any two reasonable football models, and lost roughly half its
calendar time to results that had to be re-measured.

```
walk      : 1153 barriers, 2016-08-13 -> 2026-05-24 (refit every 1)
pool      : 3,800 matches
market    : avg_closing (shin) RPS 0.1968 on 2,660 covered

arm                   RPS  log loss   skill  draw res  vs market   vs baseline
uniform            0.2390    1.0986    0.0%   0.00000    +0.0416   (baseline)
home-rate          0.2335    1.0684    2.3%   0.00000    +0.0379   -0.0056 [-0.0082, -0.0030] P=1.000
home-always        0.4375   19.1236  -83.0%   0.00000    +0.2508   +0.1985 [+0.1841, +0.2127] P=0.000
```

### The market number is an independent check on the entire odds path

**Market RPS 0.1968** on the 2,660 covered matches. The research report predicts a de-vigged
closing line at **0.19–0.20** (Pitcan's Serie A 0.1905; Baboota & Kaur's EPL 0.2012). Landing
inside that band is meaningful external validation: it means the column ladder, the Shin de-vig,
the outcome coding, the barrier construction and the pooling are all jointly correct. A bug in any
one of them would almost certainly have pushed this outside the band.

It also sets the target concretely. **The model has to get from 0.2390 (uniform) to somewhere near
0.20 to be worth having, and 0.1968 is the ceiling it will not beat.**

### Three baselines, and what each is for

`uniform` scores 0.2390, not the oft-quoted 0.2222 — the latter is uniform's expectation against a
*balanced* outcome mix, and the Premier League's is not balanced. `home-always` scores 0.4375 with
a log loss of 19.1, the clipping made visible rather than hidden. `home-rate` — the league's own
home/draw/away split, estimated from data before each barrier — scores 0.2335, so **simply knowing
the base rate is worth 2.3% skill**. That is the real floor a model must clear, not uniform.

### Two corrections to the brief's expectations, from the data

**Promoted-team fixtures are ~28% of the pool, not ~15%.** The brief estimates 15%; the slice
counts 1,080 of 3,800. The arithmetic is straightforward once stated: three promoted teams play 38
matches each, minus the handful they play against one another, so ≈108 of a season's 380 fixtures
involve one. This roughly doubles the expected value of the multi-tier promoted-team arm, and
doubles the cost of getting promoted teams wrong.

**Home advantage in the test decade is well below its 25-season average.** Observed base rates on
2016-17..2025-26: **home 44.6%, draw 23.2%, away 32.1%**. The report quotes ~46.2 / 27.5 / 26.3
across 25 seasons. Away wins are ~6 points higher and draws ~4 points lower than the long-run
figures. That is a substantial drift within the very span the model is judged on, and it raises the
prior on the time-varying home-advantage arm well above "small positive".

The by-season slice also caught the natural experiment unprompted: **2020-21 is the only season
where `home-rate` scores *negative* skill (−3.1%)** — a fixed home-weighted forecast is actively
harmful in the empty-stadium season.

### The guards

- **Alignment guard**: identical match identities across arms, asserted; the run fails rather than
  reindexing, because a paired comparison on mismatched rows is meaningless.
- **The does-it-do-anything guard** (`assert_arms_differ`): every non-baseline arm's probability
  vector must differ from the baseline's, and the baseline must be reproducible bit for bit. This
  is the WC2026 false-null trap — a broken experiment returning "no effect" is indistinguishable
  from a correct one — and an assertion is the only defence.
- **An arm may not have partial coverage.** A forecaster that cannot cover the pool is a benchmark,
  not an arm; the market is therefore the gate-2 benchmark rather than an arm.
- **Both gates are computed and reported**, never auto-applied. `gate_verdicts` is tested against
  the `rsfit` shape specifically: better on the pool, worse against the market, correctly rejected.

### A vacuous guard, found and removed

`freeze_matchday` originally asserted that the training frame did not reach the barrier — *after*
filtering it to `date < barrier`. The assertion could never fire, and a test written to prove it
would fire failed correctly. Replaced with something informative: same-day results are excluded
(barriers are date-granular because kickoff times only exist from 2019/20) and the exclusion is
**counted** in the frozen block. Recorded because a guard that cannot fail is worse than no guard —
it reads as protection while providing none.

### The live ledger

`pl live` freezes the next unplayed matchday's forecasts to a dated JSON and refuses to rewrite
one. Currently a no-op: the 2026/27 fixture list is not published yet, so it reports that and exits
cleanly. Honest note recorded in the module: for these three model-free arms a ledger rebuilt in
September is numerically identical to one frozen in August, so this is operational rehearsal. The
discipline becomes load-bearing the moment a fitted model exists.

**DoD met:** `pl compare --arms uniform,home-always` emits a JSON report containing the acceptance
rule, a paired delta with CI, a DM statistic, a coverage report and calibration slices. 188 tests
green.

---

## 2026-08-17 — Production Dixon–Coles: the baseline lands where it was predicted to

```
walk      : 1153 barriers, 2016-08-13 -> 2026-05-24
pool      : 3,800 matches | market avg_closing (shin) RPS 0.1968 on 2,660 covered

arm                   RPS  log loss   skill  draw res  vs market   vs baseline
home-rate          0.2335    1.0684    2.3%   0.00000    +0.0379   (baseline)
dixon-coles        0.2005    0.9718   16.1%   0.00202    +0.0082   -0.0330 [-0.0373, -0.0288] P=1.000

fits: 1153/1153 converged, 17.2 mean iterations, half-life 730d, rho clamped 0 times
```

**Pooled RPS 0.2005**, inside the pre-registered 0.196–0.206 band and within 0.001 of Koopman &
Lit's semi-dynamic EPL row (0.2014) — the specification this model actually is. **Market gap
+0.0082** against a prior of +0.006–0.007: slightly wider, and wider is the reassuring direction.
The entry above recorded in advance that a materially *smaller* gap would be a leakage signal
rather than a triumph, so this is the outcome that requires no investigation.

### Half-life: selected at 730 days, but read it as a plateau

Grid on `tuning_span` (1996-97..2005-06), a window neither evaluation span touches:

| half-life | 30 | 60 | 90 | 120 | 180 | 240 | 365 | 548 | **730** | 1095 | 1460 | 1825 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tuning RPS | .2237 | .2080 | .2053 | .2037 | .2018 | .2008 | .2006 | .2006 | **.2005** | .2007 | .2011 | .2014 |
| sensitivity RPS | .2411 | .2142 | .2120 | .2086 | .2040 | .1971 | .1967 | **.1965** | .1967 | .1975 | .1983 | .1990 |

The winner is interior (not at a grid edge) but wins by **0.00013**. Across 240–1095 the curve is
flat in *both* eras, so the honest statement is that the half-life is **weakly identified above
~240 days**, not that 730 is optimal. 730 is taken because it is the designated selection window's
winner; the sensitivity window (which prefers 548 by 0.0002) confirms the *region* without being
consulted for the choice, which is what keeps it a sensitivity check.

Short half-lives are genuinely bad — 0.2237 and 0.2411 at 30 days — so the plateau is a real one,
not an absence of signal. Against the research report's expectation of a "6-month-to-1-year"
optimum, the data mildly prefers longer memory, but 365 sits 0.0002 from the winner, so this is not
a disagreement worth arguing about.

### Two findings that raise the prior on later arms

**Home advantage is in continuous, monotone decline across twenty years.** The two backtests were
run independently and their fitted home-advantage parameters trace one line:

| window | first barrier | last barrier | multiplier on the home rate |
|---|---|---|---|
| 2006-07 → 2015-16 | +0.3240 | +0.2709 | ×1.38 → ×1.31 |
| 2016-17 → 2025-26 | +0.2711 | +0.1774 | ×1.31 → ×1.19 |

The handover agrees to four decimal places (+0.2709 against +0.2711) despite coming from separate
walks — a continuity check nobody arranged. A *global* home advantage is therefore fitting an
average over regimes that differ visibly within the very span the model is judged on. Combined with
the base-rate drift already recorded (away wins ~6 points above their 25-season norm), the
time-varying home-advantage arm now has a far stronger prior than "small positive".

**rho is negative, decisively: mean −0.0326, negative in 92% of 1,153 fits, and negative at every
half-life on both windows** (−0.041 to −0.058 on tuning, −0.050 to −0.058 on sensitivity in the
stable region). This is the free falsifier gating Arm 9. The research report cited a five-league
study finding dependence *positive* in four leagues and negative only in Ligue 1, which would have
made a negative-dependence copula the wrong tool for the Premier League. Measured here, the
Premier League behaves like Ligue 1 and like the WC2026 corpus. **Arm 9's premise survives** — the
opposite of what the report's §A implied, and it cost nothing to find out.

### Three implementation decisions, and what each bought

**Analytic gradient.** Checked against `approx_fprime` to ~1e-7 relative error across random
parameter draws. A ~100-long parameter vector would need ~100 likelihood evaluations per gradient
step numerically; with the closed form a fit takes 131 ms cold.

**Warm starting.** Mean iterations fall from ~40 cold to **17.2** along the walk, putting a full
1,153-barrier arm at ~2.3 minutes. A test asserts warm and cold starts reach the same optimum, so
it is a speed device and not a modelling change.

**Sum-to-zero by construction.** Verified by simulating from known strengths and recovering them.
The recovery test asserts *unbiasedness across seeds* rather than accuracy on one draw: a single
fit's per-team error is dominated by Poisson noise (SE ~0.03–0.07 on 1,800 matches), so a tight
single-draw bound would have been testing luck. Mean error across six seeds is ~0.01–0.04 against
an SD of 0.02–0.07, and home advantage recovers 0.26 from a truth of 0.26.

### Two bugs found by building, both about scale-dependence

**A fixed cold-start threshold silently biases a half-life sweep.** The first design pinned a team
below 5 *effective* (decay-weighted) matches. At a 30-day half-life a team playing every nine days
accumulates only ~4.6 effective matches in total, so the bar pinned the entire league and the fit
raised. The grid point would have scored badly for a reason with nothing to do with how well a
30-day memory forecasts football. Replaced with a **share of the median team's effective history**
(15%), which means the same thing at every decay rate. This is the kind of interaction that only
shows up when a hyperparameter sweep actually runs.

**The valid range for rho is rate-dependent, so no fixed bound can guarantee it.** The
configuration bound of ±0.2 is safe at typical rates but not universally: the binding cell
`1 − λμρ` goes negative above ρ = 0.16 at λ = μ = 2.5. A fit valid on its own training rates can
still meet an extreme matchup at prediction time that puts the same rho out of range — which is
exactly what happened, at 30- and 60-day half-lives where the fit is degenerate (rho pinned at its
bound in 227 of 342 fits, implied λμ reaching 13.1). Handled by clamping rho **per match** into
that match's valid range and **counting every clamp** into the report. At the production half-life
it triggers zero times, so the counter doubles as a degeneracy detector: a non-zero count means the
configuration is wrong, not the fixture. The config comment claiming "0.2 is a comfortable margin"
was itself wrong and has been corrected.

### Seams

All six (`covariates`, `dynamics`, `observation`, `ensemble`, `home_advantage`, `tiers`) ship off,
each with a byte-identity regression test asserting a SHA-256 digest of the probability matrix is
unchanged when the seam is absent, set explicitly off, or the run is simply repeated. A companion
test flips each seam and asserts the inertness detector *notices* — a detector that never fires
would prove nothing.

### The sensitivity decade agrees

`pl backtest --sensitivity` on 2006-07..2015-16 (995 barriers, 3,800 matches):

```
home-rate          0.2255    3.9% skill   (baseline)
dixon-coles        0.1967   16.2% skill   -0.0288 [-0.0327, -0.0247] P=1.000
rho: mean -0.0539, 100% of 995 fits negative
```

**Skill is 16.2% against the test decade's 16.1%** — indistinguishable. The lower RPS (0.1967 vs
0.2005) is therefore not the model working better in that era; it is that era's outcomes being more
predictable, and the skill figure is what strips that out. Worth remembering when any future arm
appears to do better on one window than the other. The market gate is absent here because
`avg_closing` does not reach back before 2019/20, which is the coverage limit already recorded.

**DoD met:** `pl backtest` reports pooled RPS 0.2005 (band 0.196–0.206) with a market gap of
+0.0082 (prior +0.006–0.007, and on the safe side of it). 239 tests green.

---

## 2026-08-17 — Reproduction: Pitcan (2026) on Premier League data

`pl reproduce --paper pitcan2026`. Three separate conclusions in the build plan rested on this one
days-old, un-peer-reviewed preprint about a league that is not ours, whose central result is a
boundary solution. It is re-run here before anything is built on it.

**The paper's method was read, not guessed.** The arXiv HTML was fetched and §§4, 5.1–5.5 and
6.3–6.4 transcribed, so the reproduction implements the paper's actual specification rather than a
plausible reconstruction of it. That mattered: §5.3 specifies a *league-wide* finishing factor
converting shot rates to goal rates, and specifies that the τ correction is **not** carried across
(the variant is independent-Poisson on converted rates). Both are things a reimplementation from
the abstract would have got wrong.

### Result: three of four weights reproduce exactly

Weights fitted on validation (2013-14..2018-19), the paper's own protocol:

| Pool | ours | paper |
|---|---|---|
| market + goals — weight on **goals** | **0.000** | 0.00 |
| market + shots — weight on **shots** | **0.000** | 0.00 |
| market + goals + shots (simplex) | **1.000 / 0.000 / 0.000** | 1.00 / 0.00 / 0.00 |
| goals + shots — weight on **shots** | 0.170 | 0.35 |

Sample sizes match the paper exactly — validation n = 2,280, test n = 2,660 — because Serie A and
the Premier League both play 380-match seasons. Test coverage is 2,490 of 2,660 rather than the
full set, the missing 170 being the Pinnacle termination at 2026-01-08 already on the record.

**The boundary solution is genuine here too.** Log loss is monotone increasing in the weight on the
goals model across the whole admissible [0, 1], so zero is the argument minimum rather than where
an optimiser stopped. Extending to [−1, 1] the unconstrained minimum is **−0.100** against the
paper's −0.225 — same sign, and the same reading: a negative weight divides the price by the model,
which sharpens it, so the unconstrained optimum acts as a temperature correction rather than as an
information channel.

**The market gap agrees.** Ours +0.0082 [+0.0055, +0.0107]; the paper's +0.0067 [+0.0046, +0.0088].
Overlapping intervals. Our own production backtest independently gave +0.0082 on a different market
family and pool, so three routes now agree the gap is real and near +0.007.

Finishing factors: PL **κ_home 0.304, κ_away 0.290** against Serie A's 0.309 / 0.317. Nearly
identical at home; Premier League away sides convert slightly less.

### The one number that didn't match is a finding, not a failure

`goals + shots` returned 0.170 against the paper's 0.35. The market is not in that pool, so the
odds cannot explain it. Re-estimating across decay rates:

| half-life | weight on shots | goals RPS | shots RPS |
|---|---|---|---|
| 180 | **0.630** | 0.1980 | **0.1975** |
| 365 | 0.433 | 0.1969 | 0.1985 |
| 730 (production) | 0.170 | 0.1969 | 0.2000 |
| 1460 | 0.086 | 0.1987 | 0.2020 |

**The paper's 0.35 sits inside our range.** The discrepancy is the decay rate: the paper fits ξ on
its validation window, while we deliberately carried the production half-life rather than refit on
a window that overlaps our acceptance test span.

The mechanism is visible in the RPS columns. As memory shortens the **shots model improves
monotonically (0.2000 → 0.1975) while the goals model does not (0.1969 → 0.1980)** — and at 180
days the shots model actually *beats* the goals model. Shot counts are a denser signal than goals,
roughly three per goal, so they can be estimated from a shorter window without paying the variance
penalty that sparse goal counts impose.

**Design consequence for the xG arm: a second observation channel must get its own decay rate.**
Inheriting the goals model's half-life understates what the channel carries by a factor of about
four. Neither the paper nor the build brief anticipates this; the research report's §J lists
per-parameter time decay as under-tried and unpublished, and this is adjacent evidence for it.

### Verdict and gate

The Premier League reproduces the Serie A pattern. **Arm 4 (xG as a second observation channel) is
cleared to PROCEED**, with two pre-registered expectations that follow directly:

1. It should **pass gate 1** — the chance-creation channel demonstrably carries information the
   goals model lacks (weight 0.17–0.63 depending on decay).
2. It should **leave the market gap untouched** — against the price the same signal earns 0.000,
   and in the three-way pool both structural models collapse to zero simultaneously.

That combination is adoptable under the acceptance rule, since gate 2 requires only that the gap
not *degrade*. Recording the expectation now means a market-gap improvement would be a surprise
demanding scrutiny rather than a success to be claimed.

The shots-on-target model (`model/shots.py`, arm `dixon-coles-sot`) is retained as production
groundwork: swapping shots for xG leaves the conversion and pooling unchanged.

---

## 2026-08-17 — PRE-REGISTRATION, Arm 1: per-team attack/defence vs an Elo-difference scalar

*Written before the comparison was run. The tuning-span numbers below were available; no test-span
result was.*

### Hypothesis

Per-team attack and defence lower pooled RPS relative to a Dixon–Coles parameterised by a single
Elo rating difference, on the 2016-17..2025-26 test span.

### Why this arm exists at all

The WC2026 project used the Elo scalar **because international football is too sparse for 2N
parameters** — teams play ~10 matches a year and many pairs never meet. A 20-team league playing 38
rounds does not have that constraint, so the choice should reverse. This project already assumed
the reversal and shipped per-team attack/defence as production; this arm is the check that the
assumption was right, run in the direction that could embarrass it.

The research report states that **no clean published head-to-head isolates the two
parameterisations on league data** ("treat the magnitude as unknown but the sign as
well-supported"). So the sign is near-certain and the magnitude is genuinely unmeasured.

### Arm definition — exactly one axis

`elo-dc` changes only how strength enters the goal rates:

```
elo-dc      : log lam = a + h + c*d,  log mu = a - c*d       (4 parameters)
dixon-coles : log lam = c + h + A[home] - D[away], ...       (~2N + 3 parameters)
```

Identical between the two: the τ correction, the exponential decay, the weighted likelihood, the
analytic-gradient optimiser, the scoreline collapse, the walk-forward splits, the pool.

**Both models are tuned before comparison.** Comparing a tuned model against a default-configured
one would measure tuning effort rather than parameterisation. `elo-dc` gets its own K and its own
half-life, selected by one shallow coordinate pass on `tuning_span` (1996-97..2005-06) — the same
window the production half-life came from, and one neither evaluation span touches. Giving it a
separate half-life is a direct application of the reproduction's finding that different
parameterisations genuinely prefer different memory.

**The Elo replay reads only E0.** Ratings could be carried across the promotion boundary from the
lower tiers, which would hand promoted teams a real rating instead of a cold start. That is a real
advantage and it belongs to the multi-tier arm; including it here would confound "per-team vs
scalar" with "one tier vs four" and leave neither interpretable.

### Pre-registered bar

The standing acceptance rule, unchanged. Baseline is **`elo-dc`** and the candidate is
**`dixon-coles`**, so a favourable delta is evidence that the production choice was correct:

1. paired-bootstrap RPS delta favourable — 95% CI excludes 0, or P(better) ≥ 0.95;
2. delta vs the Shin de-vigged market must not degrade.

Corroborated by a HAC-corrected Diebold–Mariano, and FDR-controlled across the family of arms.

### What each outcome will mean

* **Accept** — the production parameterisation is justified on measured evidence rather than on an
  argument about sparsity, and the magnitude fills a gap the literature leaves open.
* **Null** — the ~40 extra parameters are buying nothing on a decade of Premier League matches.
  That would be a genuinely surprising and publishable negative, and would argue for reverting
  production to the four-parameter model on parsimony grounds alone.
* **Reject (Elo wins)** — the sparsity argument applies to club football too, and production is
  wrong. This is the outcome I consider least likely and would investigate hardest before
  believing, starting with whether the per-team model's promoted-team cold start is doing the
  damage.

A null or a reject would be reported as-is. The point of pre-registering the interpretation is that
none of these three readings can be chosen after the number appears.

### Prediction, recorded to be scored later

Per-team att/def wins, by −0.002 to −0.006 RPS. Reasoning: the per-team model already scored 0.2005
against a market at 0.1968, and the scalar cannot express attack/defence asymmetry, which is large
and persistent in a 20-team league. Against that, the scalar's four parameters are far better
determined, and Elo ratings are themselves a strong summary — so a delta smaller than 0.002 would
not shock me.

### Addendum, written after tuning and before the test run

Tuning `elo-dc` on the tuning span produced two things worth recording *before* the confirmatory
run, so the final entry cannot imply I was surprised only at the end.

**The prediction above already looks wrong.** On the tuning span `elo-dc` scores **0.19760** where
the per-team model scores **0.20048** — the scalar is ahead by ~0.0029, in the opposite direction.
The test-span comparison is still the one that counts, and these are unpaired numbers from separate
runs, but the sign is not what I wrote down.

**The half-life sweep hit a grid edge, which the project treats as a red flag**, so the grid was
widened rather than the edge adopted. Extended, it stays monotone decreasing to effectively uniform
weights: 0.19760 at 1825 days → 0.19748 with no decay at all. That is structural rather than a
failed search — **the Elo replay already carries the recency weighting**, so the layer above it
only calibrates a rating-difference-to-goal-rate map, and that relationship is stable enough to
want every match equally. The whole effect is 0.00012 RPS, so it changes nothing material; it is
set correctly for fairness, not for advantage.

Tuned values: **K = 15** (interior to 10..60, shallow), **no decay**.

---

## 2026-08-17 — RESULT, Arm 1: NULL. The prediction was wrong.

```
walk      : 1153 barriers, 2016-08-13 -> 2026-05-24 | pool 3,800 | market 0.1968

arm                   RPS  log loss   skill  draw res  vs market   vs baseline
elo-dc             0.1996    0.9699   16.5%   0.00163    +0.0071   (baseline)
dixon-coles        0.2005    0.9718   16.1%   0.00202    +0.0082   +0.0009 [-0.0005, +0.0022] P=0.099

  reject  dixon-coles   gate1 FAIL (P = 0.099), gate2 FAIL
```

**Per-team attack and defence do not beat a single Elo rating scalar.** The delta is +0.0009 with a
confidence interval straddling zero, so this is a null rather than a reject — but the point estimate
runs the *wrong way*, and the scalar is also ahead on the market gap (+0.0071 against +0.0082).

I predicted −0.002 to −0.006 in the per-team model's favour. The research report predicted the same
direction with more confidence than I did: "expect this to help; the whole lineage is built on it."
Neither survived contact with the data.

### The result replicates on an independent decade

| span | delta (per-team − Elo) | 95% CI | P(per-team better) |
|---|---|---|---|
| test 2016-17..2025-26 | +0.0009 | [−0.0005, +0.0022] | 0.099 |
| sensitivity 2006-07..2015-16 | +0.0008 | [−0.0003, +0.0018] | 0.079 |

Same sign, near-identical magnitude, both nulls, on two decades that share no matches. Diebold–
Mariano agrees with the bootstrap on the test span (statistic 1.25, p = 0.21), which is the
robustness signal the two tests exist to provide.

### The obvious confound was checked and is not the explanation

Elo *retains* a relegated club's rating across its absence, whereas the per-team model resets a
club with too little effective history to league average. That should hand Elo an advantage exactly
on promoted-team fixtures. It does not:

| slice | elo-dc | per-team | delta |
|---|---|---|---|
| involves a promoted team (n=1,080) | 0.1893 | 0.1899 | +0.0007 |
| established clubs only (n=2,720) | 0.2037 | 0.2046 | +0.0009 |

The gap is the same on both sides of the partition, so rating retention is not driving it.

### This is a substantive null, not two copies of the same model

The harness's does-it-do-anything guard passed, which for a null result is the load-bearing check —
it is exactly the WC2026 false-null trap, where a broken arm silently reproducing the baseline
looks identical to an honest null. Quantified, the two arms disagree a great deal:

| | |
|---|---|
| mean absolute probability difference | **0.0331** per outcome |
| largest single-match difference | 0.3573 |
| matches differing by >0.05 on some outcome | **39.5%** |
| correlation of per-match RPS | 0.9535 |

So this is not "two parameterisations that collapse to the same forecast". They disagree materially
on two matches in five, by up to 36 percentage points, and **still score the same in aggregate**.
Their per-match errors are highly correlated, which is precisely the condition that makes the
paired bootstrap sensitive — an unpaired comparison would not have resolved a 0.0009 delta on 3,800
matches, and the interval would have been wide enough to hide a real effect.

### The comparison was fair, and the asymmetry in tuning is justified by the data

`elo-dc` got an extended half-life grid and the production model did not, which looks like unequal
effort until the curves are compared. The per-team model's sweep has a genuine **interior**
optimum — it turns at 730 and rises through 1095, 1460 and 1825 — so extending its grid would have
found nothing. `elo-dc`'s sweep was monotone to the edge, so extending it was mandatory under the
project's own grid-edge rule. The asymmetry follows from what the data did, not from how hard each
model was pushed.

### What the null actually says: 67 parameters, bought nothing

The fits report it directly — **4 parameters against 67** (3 global plus 2×(20−1) team terms), for
statistically indistinguishable accuracy across two decades. In a 20-team league with a two-year
memory, per-team attack and defence are **over-parameterised relative to the information
available**, and the Elo scalar's sequential update is acting as a very effective regulariser.

That reading is supported by where the per-team model does win. It is ahead in 2018-19 through
2021-22 and behind in 2022-23 and 2024-25 — season-level noise of ±0.005 around a zero effect, not
a stable edge. Its largest single-season win is **2020-21 (−0.0046)**, the empty-stadium season,
which is consistent with per-team parameters adapting faster than Elo when home advantage collapses.

### Decision: production stays per-team, and that is a choice about utility, not accuracy

**Neither direction clears the bar.** Reversed — elo-dc as the candidate against per-team as
baseline — the delta is −0.0009 with P(better) = 0.901, still short of 0.95. The acceptance rule
declines to move production in either direction, which is the correct answer when two models are
indistinguishable.

Production therefore stays on per-team attack/defence, on these grounds and no others:

* most of the remaining roadmap is built on per-team structure — the dynamic-states arm is per-team
  by construction, and the xG channel, multi-tier fit and time-varying home advantage all attach to
  team parameters rather than to a scalar;
* the season simulator needs per-team strengths to propagate;
* accuracy is equal, so keeping it costs nothing measurable.

What must **not** be claimed is that the data justified it. It did not. Recorded here so no later
entry can imply otherwise.

### The lead this opens

If the per-team model is over-parameterised and the Elo scalar under-expressive, the synthesis is
**per-team attack/defence shrunk toward a scalar-derived prior** — a hierarchical fit rather than
either extreme. Baio & Blangiardo's over-shrinkage mixture is the published precedent, and the
research report lists hierarchical Bayes in the state-space lineage. This is a recorded lead, not
an adopted change; it goes into the backlog rather than into production, because point estimates
from a losing arm are exactly what the standing discipline says not to act on.

Also worth noting for the dynamics arm: **elo-dc's likelihood prefers no time decay at all**,
because the Elo replay already carries the recency weighting. A score-driven dynamic model does the
same job more principledly, which strengthens the case that the dynamics arm is competing with Elo
rather than with the static per-team fit.

---

## 2026-08-17 — PRE-REGISTRATION, Arm 2: structural home advantage

*Written before the comparison was run. The regime table below is descriptive statistics used to
define the arm; no arm result existed.*

### Hypothesis

Giving home advantage explicit structure — a time trend, an empty-stadium term, or both — lowers
pooled RPS against the production model, which fits a single global `h` per barrier.

### Why the prior is stronger than the brief's "small positive"

The brief expects a small gain. Three measurements already on the record argue for more:

* the fitted `h` falls **+0.3240 → +0.2709 → +0.1774** across two independently-run backtests
  spanning 2006-2026 (×1.38 → ×1.19 on the home rate), agreeing to four decimals at the handover;
* test-decade base rates are home 44.6% / draw 23.2% / away 32.1% against a 25-season norm of
  ~46.2 / 27.5 / 26.3 — away wins some six points above their long-run rate;
* Pitcan's §4 reports the same trajectory in Serie A, home:away goals **1.38 → 1.14**, so this is
  not a Premier League quirk.

A single global `h` fitted on 730-day decay is therefore averaging over regimes that differ visibly
*within the span the model is judged on*.

### The empty-stadium regime, measured

| regime | n | home% | away% | home:away goals | ppg diff |
|---|---|---|---|---|---|
| pre-COVID 2015-08..2020-03 | 1,808 | 45.7 | 30.3 | 1.279 | +0.46 |
| **restart 2020-06..2020-07** | 92 | **46.7** | 31.5 | **1.315** | **+0.46** |
| **2020-21 season** | 380 | **37.9** | **40.3** | **1.008** | **−0.07** |
| 2021-22 | 380 | 42.9 | 33.9 | 1.159 | +0.27 |
| 2022-08..2026-05 | 1,520 | 44.5 | 31.4 | 1.212 | +0.39 |

2020-21 is a collapse — away wins outnumber home wins, for the first time in English top-flight
history. **But the June–July 2020 restart, also played behind closed doors, shows no effect at
all**: home 46.7% and a 1.315 goal ratio, indistinguishable from the pre-COVID baseline.

**The window is defined causally anyway** — every match played without a crowd, 2020-06-17 to
2021-05-23. Dropping the restart because it fails to show the expected effect would be fitting the
definition to the outcome, and would make the dummy look stronger than the evidence deserves. If
the restart dilutes the term, that is a finding about the crowd hypothesis. Ninety-two matches is
in any case too few to conclude the restart was genuinely unaffected. Two known impurities stay in
for the same reason: partial crowds in December 2020 and again in May 2021.

Note also that home advantage has **not** returned to its pre-COVID level (44.5% against 45.7%),
which is why the trend term is tested separately rather than folded into the dummy.

### Arm definitions — one axis each, three arms, FDR-controlled

Baseline is `dixon-coles`. Each arm adds parameters that enter the home rate only, are fitted
jointly by the same weighted likelihood, and **contribute exactly zero at prediction time** — a
future match is neither in the past nor behind closed doors, so the forecast uses the current `h`
rather than a window average. That last property is the whole point of the arm.

* **`ha-trend`** — `h(t) = h + h_trend · years_before_barrier`. One parameter.
* **`ha-empty`** — `h(t) = h + h_empty · 1[no crowd]`. One parameter.
* **`ha-both`** — both. Two parameters.

Three arms is a family, so Benjamini–Hochberg applies across it. Everything else is shared with the
baseline through one code path, so a paired delta is attributable to the seam alone.

### Pre-registered bar

The standing rule: paired-bootstrap delta favourable (95% CI excludes 0 or P(better) ≥ 0.95) **and**
no degradation against the market, with BH-FDR across the three arms and DM as corroboration.

### What each outcome will mean

* **Accept** — home advantage is genuinely non-constant on a decade horizon, and the production
  model has been leaving a measurable amount on the table by averaging over it.
* **Null** — the 730-day decay already tracks the drift well enough that stating it explicitly adds
  nothing. Given the drift is real and large, this would say the *decay* is doing the job, not that
  the drift does not exist — a distinction the entry must make, because "no detectable effect
  through this pipe" is not "no effect".
* **`ha-empty` null while `ha-trend` accepts** — the crowd hypothesis is weaker than the secular
  decline, consistent with the restart anomaly above.

### Prediction, recorded to be scored later

`ha-trend` gains −0.001 to −0.003; `ha-empty` gains less, −0.000 to −0.002, diluted by the restart
and by the decay already down-weighting 2020-21 by the time it matters; `ha-both` lands near
`ha-trend`. I put maybe 50% on any of the three clearing gate 1 — the drift is real but the decay
is already a crude version of the same correction, and Arm 1 has just taught me that extra
parameters do not automatically pay for themselves here.

---

## 2026-08-17 — RESULT, Arm 2: NULL, but an informative one — and a bug found mid-arm

```
arm                   RPS  log loss   skill  draw res  vs market   vs baseline
dixon-coles        0.2005    0.9718   16.1%   0.00202    +0.0082   (baseline)
ha-trend           0.2005    0.9719   16.1%   0.00189    +0.0083   +0.0000 [-0.0001, +0.0002] P=0.306
ha-empty           0.2003    0.9715   16.2%   0.00200    +0.0081   -0.0001 [-0.0004, +0.0001] P=0.865
ha-both            0.2004    0.9717   16.2%   0.00188    +0.0082   -0.0000 [-0.0003, +0.0002] P=0.629

  reject  ha-trend   gate1 FAIL (P=0.306)   gate2 FAIL
  reject  ha-empty   gate1 FAIL (P=0.865)   gate2 pass
  reject  ha-both    gate1 FAIL (P=0.629)   gate2 pass
```

All three rejected. Benjamini–Hochberg across the family: 0 of 3, adjusted p 0.66–0.74. Diebold–
Mariano agrees.

### A mis-specification found and fixed before the result was believed

The first run returned all three arms at exactly +0.0000 — suspiciously clean. The by-season slice
showed why: **`ha-empty` was *worse* in 2020-21 (+0.0006), the one season it exists for.**

The cause was mine. Structural terms were applied when *fitting* but zeroed when *predicting*, so
the empty-stadium arm cleaned the crowd effect out of its historical estimate and then forecast the
2020-21 matches as if crowds were present — the worst of both. Whether an upcoming match is played
behind closed doors is public before kickoff, so applying it is leak-free; omitting it was simply
wrong.

After the fix, that season flips from **+0.0006 to −0.0013**, a swing of 0.0019 in the season that
matters, and `ha-empty` moves from P=0.324 to **P=0.865** and from failing gate 2 to passing it.
`ha-trend` is unchanged to four decimals, which is the consistency check — the trend *is*
legitimately zero at prediction, since a match is zero years before its own barrier.

This is the failure mode the harness's does-it-do-anything guard cannot catch: the arms genuinely
differed, so the guard passed, but one of them was differing *wrongly*. A regression test now pins
it. Had it gone unnoticed, this entry would have recorded a clean null produced by a broken arm.

### The null is real, and it is a dilution result

`ha_empty` is not a term that failed to find anything. Fitted across barriers it lands consistently
at **−0.08 to −0.15** in log-goal units — home sides scored roughly 10% fewer goals behind closed
doors, matching the raw collapse from a 1.279 home:away goal ratio to 1.008. It then improves
2020-21 by −0.0013.

It fails anyway, because **one season in ten, correctly handled, is worth −0.0001 pooled**. The
effect is real, the term captures it, and the yardstick is a decade. This is precisely the
distinction pre-registered above: *no detectable effect through this pipe* is not *no effect*. A
model forecasting the 2020-21 season specifically should carry this term; a model judged on a
decade cannot justify it.

### `ha-trend` returns exactly zero, and that answers the other question

The trend term contributes nothing at all (+0.0000, P=0.306). Given the drift is large and
well-measured — `h` falling ×1.38 → ×1.19 over twenty years — the reading is that **the 730-day
exponential decay is already tracking it**. Stating the drift explicitly adds nothing the decay was
not already doing implicitly. That is a satisfying answer to a question the brief left open, and it
retires the "time-varying home advantage" idea in its simple form rather than leaving it as a
perpetual maybe.

It also revises the prior I raised after the production backtest. The *measurement* of drift was
correct and stands; the inference that a model would gain by modelling it explicitly does not.

### Scoring the prediction

I predicted `ha-trend` −0.001 to −0.003 and `ha-empty` −0.000 to −0.002, with ~50% that any arm
clears gate 1. `ha-trend` came in at +0.0000 — outside my range and on the wrong side.
`ha-empty` at −0.0001 was inside it. None cleared gate 1, so the scepticism was warranted and the
specific trend prediction was not. Two arms now, two wrong directional calls: worth noting as a
pattern rather than as two coincidences.

### On leakage

The empty-stadium window is a calendar fact about public health policy, not an outcome-derived
quantity, and each match's status was known before its own kickoff. The fixed end date is only used
to classify training rows, which all lie before the barrier. No path from a result to its own
forecast.

Seams remain off in production; all four arms leave `dixon-coles` byte-identical.

---

## 2026-08-18 — BLOCKED, Arm 4: there is currently no accessible xG source

Arm 4 (xG as a second observation channel) is cleared on the evidence — the `pitcan2026`
reproduction confirmed the pattern and predicted the arm would pass gate 1 — and its *mechanism* is
already built and tested: the log-pool weight estimator, the second-channel model, the conversion
from a chance-creation rate to a goal rate. Only the input is missing.

### What was checked, 2026-08-18

| source | status |
|---|---|
| **Understat** `robots.txt` | **`User-agent: * / Disallow: /`** — crawling not permitted |
| Understat `/match/{id}`, `/league/EPL/{year}` | **404** (both returned data on 2026-08-16) |
| Understat `/office` | 307 redirect, appears to be a login portal |
| **FBref** `/en/comps/9/Premier-League-Stats` | **403** |
| football-data.co.uk `notes.txt` + all 132 cached files | **no xG columns for England**, confirmed |
| **StatsBomb open data** | Premier League: **2003/04 and 2015/16 only** |
| football-data.org API | 403 without a key |
| local Kaggle credentials | present but a 37-byte stub, not valid JSON |

This is a **material update to Correction 4**, which records Understat as the single live xG source
following FBref's January 2026 Opta licence loss. As of today Understat's public pages 404 *and*
its robots.txt disallows all crawling, so it is not a source this project may use regardless of
whether the pages return. There is presently **no accessible xG source for the Premier League,
live or archival.**

Understat's pages were readable on 2026-08-16, two days before this entry, so the change is recent.
The `/office` portal suggests access may have moved behind registration or licensing rather than
disappearing.

### Why StatsBomb does not rescue it

The open-data repository would let us **train** an xG model — the research report recommends
exactly that as the hedge for methodological control — but training is not the constraint.
Applying an xG model needs *shot-level events for every match being forecast*, and StatsBomb
publishes two Premier League seasons, neither inside the test decade. Our own corpus carries shot
**counts**, not shot locations, so it cannot support an xG computation. A model with nothing to
score is not a solution.

### The arm is paused, not cancelled, and nothing is wasted

Everything except the input exists and is under test: `reproduce/pooling.py` (log pool, weight
fitting, the profile that distinguishes a genuine boundary solution from an optimiser stopping at a
bound, the simplex extension), `model/shots.py` (the identical machinery on a chance-creation
target, with the league-wide finishing conversion), and the `dixon-coles-sot` arm. Swapping shots
on target for xG changes the input column and nothing else.

**Shots on target remains available as the chance-creation channel** — full coverage from 2000/01,
already ingested, and the substitution Pitcan himself makes in his §5.3. Running the arm that way
was offered and deliberately declined in favour of obtaining real xG, so that the result speaks to
xG specifically rather than to chance-creation channels in general. Recorded so the choice is not
silently revisited later.

### Standing note on collection ethics

`robots.txt` is checked before any automated collection, and a `Disallow` is treated as decisive
regardless of whether the pages happen to respond. That is why Understat was not scraped even
though its match pages were readable two days ago.

---

## 2026-08-18 — UNBLOCKED, Arm 4: an xG source, obtained and validated

The blocker recorded earlier today is resolved. Understat is still not scrapeable — its
`robots.txt` remains `Disallow: /` and its match pages still 404 — so the figures come instead
from a Kaggle dataset published under an explicit licence, which is a different artefact obtained
on different terms. That distinction was put to the project owner and this route chosen
deliberately.

    dataset : yarknyorulmaz/understat-match-team-metrics-dataset-epl-v16-v24
    licence : Open Database License (ODbL)
    source  : understat.com, mirrored
    coverage: 3,420 Premier League matches, 2015/16 -> 2023/24

### Validated against a source we already hold, not taken on trust

Joined to the football-data.co.uk corpus on teams and date:

| check | result |
|---|---|
| join on date + teams | 3,394 of 3,420 exact; **all 26 remaining resolve at exactly ±1 day** |
| **goals, both sides** | **3,394 / 3,394 identical** |
| shots on target | 98.0% / 98.2% identical |
| xG calibration | mean home xG 1.567 vs 1.555 goals scored; away 1.260 vs 1.266 |

The goals agreement is the load-bearing one: a mirror that had been mangled, misaligned or
fabricated would fail it instantly. It is therefore enforced on **every** joined row at load time
rather than checked once here — `attach()` raises if any match's score disagrees. The ±1 day
tolerance exists because Understat timestamps by kickoff and football-data.co.uk by calendar date;
exact dates are tried first, so a fixture cannot steal another meeting's xG.

The xG calibration is a second, independent reassurance: unbiased against realised goals to within
0.012, which is what a competent xG model should look like and what a corrupted feed would not.

Six team names needed aliases (`Manchester United` -> `Man United` and five siblings) — derived by
set difference against the roster rather than guessed, and then confirmed by the score check.

### The coverage cliff, and what it does to the arm

xG runs 2015/16 to 2023/24. Against the test span that is **8 of 10 seasons, 3,040 of 3,800
matches**, with nothing for 2024-25, 2025-26, or the 2026/27 season the model must score live.

This does **not** stop the arm running on the full span. The channel model estimates team strengths
*from* xG and then predicts goals, so prediction needs only team identities — a 2025-26 barrier
still yields a forecast, just one whose xG information stops in May 2024. Both framings are
therefore reported, and they answer different questions:

* **covered subset (2016-17..2023-24)** — does the xG channel carry information *when it exists*?
  This is the scientific question and the primary result.
* **full test span** — what would production get from this source *today*? This is the
  operational question, and it is expected to be worse because the channel goes stale.

Even a win on both would not license wiring xG into production for 2026/27: there is no live feed.
An adopted-but-unrunnable arm would be a reason to buy one, not a reason to claim an improvement.

### Pre-registration, Arm 4

**Hypothesis.** A chance-creation channel built on xG lowers pooled RPS against the goals-only
production model on the xG-covered subset.

**Arm.** `dixon-coles-xg`: identical machinery to the baseline, fitted to xG counts instead of
goals, converted back to goal rates by a league-wide finishing factor, with the τ correction
dropped because it would have been estimated against a different quantity. This is Pitcan's §5.3
specification with xG substituted for shots on target, and it shares one code path with
`dixon-coles-sot` so the two differ only in which columns they read.

**Bar.** The standing rule — paired-bootstrap delta favourable (95% CI excludes 0 or
P(better) ≥ 0.95) and no degradation against the market — with `dixon-coles-sot` run alongside as a
third arm so BH-FDR applies across the family and the two channels can be compared directly.

**Prediction, from the reproduction.** It should **pass gate 1 and leave the market gap untouched**:
the reproduction found chance-creation carries information the goals model lacks (pool weight
0.17–0.63 depending on decay) while earning 0.000 against the price. I expect xG to beat shots on
target, since xG is the refinement shots-on-target proxies for. Magnitude −0.001 to −0.004 on the
covered subset.

**Two known handicaps, recorded before the run so they are not excuses afterwards.**
The xG arm trains on 2015/16 onward where the goals model trains from 1993, so it has roughly a
fifth of the history at every barrier. And xG is continuous, so applying a Poisson likelihood to it
is an approximation — routine in this literature, but an approximation. Both bias *against* the
arm, so a win would be despite them.

**What a null will mean.** Given the reproduction measured real incremental information in the
chance-creation signal, a null here would most likely indict the short training history or the
staleness rather than the signal — and the covered-subset-versus-full-span split is what will
separate those two explanations.

---

## 2026-08-18 — RESULT, Arm 4: NULL, and a qualification of the paper's own headline diagnostic

**Primary — the xG-covered subset (2016-17..2023-24, 3,040 matches, 930 barriers):**

```
arm                   RPS  log loss   skill  draw res  vs market   vs baseline
dixon-coles        0.1979    0.9596   17.6%   0.00221    +0.0074   (baseline)
dc+xg              0.1975    0.9578   17.8%   0.00199    +0.0071   -0.0004 [-0.0012, +0.0003] P=0.860
dc+sot             0.1979    0.9597   17.6%   0.00206    +0.0074   +0.0001 [-0.0001, +0.0002] P=0.276
```

**Secondary — the full test span (3,800 matches), where xG is stale after May 2024:**

```
dc+xg              0.2004    0.9712   16.2%   +0.0084   -0.0001 [-0.0008, +0.0005] P=0.630
dc+sot             0.2005    0.9719   16.1%   +0.0082   +0.0000 [-0.0001, +0.0002] P=0.370
```

Both arms rejected on both pools; Benjamini–Hochberg rejects 0 of 2.

### The arm ran the right comparison the second time

The first run tested each channel as a **replacement** for the goals model, which is not Arm 4 and
is a much weaker proposition: `dixon-coles-xg` alone scores 0.2020 against the baseline's 0.2005
(+0.0015, P=0.024 — decisively worse), and shots on target worse still at 0.2031. Pitcan's shots
model loses to his goals model outright too. The question was never whether the channel wins alone;
it is whether it carries information the goals model lacks *beside* it. Corrected to a logarithmic
pool before anything was concluded.

The pool weight is fitted **online, on forecasts the walk has already made and whose results have
since landed** — at each barrier the weight minimises log loss over every earlier barrier's
out-of-sample predictions. Leak-free by construction, and it avoids surrendering a season of the
test pool to a validation window, which mattered because xG begins with 2015/16 and there is no
earlier xG history to fit a weight on.

### The pool is emphatically not degenerate

| arm | mean weight on channel | max | share of barriers above 0.05 |
|---|---|---|---|
| `dc+xg` | **0.457** | 0.618 | 91% |
| `dc+sot` | 0.103 | 0.305 | 80% |

**xG earns 4.4× the weight of shots on target**, confirming the pre-registered expectation that xG
would beat the proxy it refines. The shots weight of 0.103 also lands near the reproduction's 0.170
at this half-life, so the two independent estimates agree.

### The finding: a 0.46 pool weight buys 0.0004 RPS

This is the part worth keeping. By the predictive-likelihood criterion the xG channel carries
**substantial** information the goals model lacks — nearly half the pooled forecast. The resulting
improvement is **−0.0004 RPS (P=0.860) and −0.0017 log loss (P=0.910)**, neither reaching the bar.

Pitcan proposes the pooling weight as the quantity that "should be the headline diagnostic in this
literature", in place of accuracy quoted beside a market figure. Measured here, the weight answers
*"does this channel carry information the other lacks?"* — and answers it well. It does **not**
answer *"does using it improve the forecast?"*, and the gap between those two questions is an order
of magnitude. Two forecasts that are highly correlated can pool near 50/50 and barely move any
score, because the pool mostly re-expresses what both already agreed on.

That is a qualification of the paper's central methodological proposal, found by running it rather
than by reading it, and it is the kind of thing only a reproduction surfaces.

### Staleness is real, and the pre-registered split separated it

The pre-registration said the covered-subset-versus-full-span contrast would distinguish "the
signal is weak" from "the signal is stale". It did: `dc+xg` strengthens from **P=0.630 on the full
span to P=0.860 on the covered subset**, and its market gap moves from degrading (+0.0084 against
+0.0082) to improving (+0.0071 against +0.0074). The channel does more when its data is current,
which is exactly what a staleness explanation predicts and what a weak-signal explanation would
not.

`dc+xg` is on that basis the strongest arm this project has tested — and it still fails.

### Against the reproduction's prediction

The reproduction predicted the channel would pass gate 1 and leave the market gap untouched. Both
halves are wrong in an interesting way: it **failed** gate 1 (P=0.860 < 0.95), and on the covered
subset it **improved** the market gap rather than leaving it flat. The pooling weight transferred
from Serie A to the Premier League; the consequence for forecast quality did not.

### Scoring the prediction

I predicted −0.001 to −0.004 on the covered subset with xG beating shots on target. Direction and
ranking right, magnitude wrong by roughly an order of magnitude (−0.0004). Three arms now, three
magnitude misses, and the two directional calls I got wrong were both on arms I expected to win.
The pattern is consistent: **I over-predict effect sizes in this problem**, which is worth
carrying into the remaining pre-registrations.

### Two handicaps, as recorded before the run

Both were declared in advance so they could not become excuses, and both still stand: the xG arm
trains on 2015/16 onward against the goals model's 1993, roughly a fifth of the history at every
barrier; and xG is continuous, so the Poisson likelihood is an approximation. Both bias against
the arm, so the true effect is plausibly a little larger than measured — but not, on this evidence,
by the order of magnitude that would change the verdict.

### Production status

Unchanged. Even had it passed, the xG channel could not be wired in for 2026/27: the source stops
in May 2024 and there is no live feed. An adopted-but-unrunnable arm would have been an argument
for buying one.

---

## 2026-08-18 — PRE-REGISTRATION, Arm 3: score-driven dynamic team states

*Written before any comparison was run on either evaluation span. The implementation existed, and
two things had already been seen and are declared below so they cannot be quietly forgotten: an
untuned single-season smoke run, and the opening cells of the tuning sweep.*

### Hypothesis

Letting each team's attack and defence move **within** the training window — a state updated after
every match by the score of the conditional likelihood — lowers pooled RPS against the production
model, which holds both constant and handles time variation only by down-weighting old matches.

### The bar is −0.003, and this is the arm where that correction bites

Correction 1 of this project's founding entry exists for this arm specifically. The research
report's TL;DR headlines a **−0.008** gain for dynamic models, but its own §B table gives three
rows on the same 2,660 EPL matches:

| specification | ARPS |
|---|---|
| static bivariate Poisson | 0.2062 |
| **semi-dynamic, DC-weighted** | **0.2014** |
| dynamic (Koopman & Lit) | 0.1982 |

Our baseline **is** the middle row — a per-team Dixon–Coles with exponential decay is exactly that
specification, and it landed at 0.2005, right where that predicts. So the honest bar against the
thing we actually have is **0.2014 → 0.1982 ≈ −0.003**, and the −0.008 is the distance from a
*static* model nobody here is running. Half of the headline gain has already been collected by the
decay. This number must not appear anywhere in this project as −0.008.

That still makes it, on paper, the largest single effect left on the arm list.

### What the arm does

Each team carries a deviation from its fitted level, `a_i` on attack and `d_i` on defence, both
starting at zero:

```
log lam = [c + h + A_H - D_A] + a_H - d_A
log mu  = [c     + A_A - D_H] + a_A - d_H
```

and after the match the four states involved move by the scaled score, `s = (x - lam + dlog tau) / lam**e`:

```
a_H <- B*a_H + K*s_home      d_A <- B*d_A - K*s_home
a_A <- B*a_A + K*s_away      d_H <- B*d_H - K*s_away
```

For a Poisson log-rate the score is `goals - expected goals`, so this reads as "a team that beats
its expectation gets better, in proportion to how much it beat it". The mirror on defence is not a
second assumption: `a_H` and `d_A` enter the same rate with opposite signs, so their derivatives
are equal and opposite. This is the Generalised Autoregressive Score family (Creal, Koopman & Lucas
2013) — the observation-driven counterpart to the latent-state model Koopman & Lit fit, which is
what makes it affordable at 1,153 barriers.

### Four design decisions, declared before the result

Each could reasonably have gone the other way, so each is recorded now rather than defended later.

1. **Estimation is two-stage.** Levels come from the ordinary weighted MLE, unaware that any state
   exists; the filter runs afterwards. Koopman & Lit estimate jointly. Two-stage is less efficient
   and it **understates the arm** — the level fit has already absorbed into a constant some of the
   variation the states exist to explain. The bias runs against acceptance, which is the safe
   direction, and it is what keeps the walk affordable.
2. **The clock is team-match time, not calendar time.** A state decays when its team plays. This is
   the Elo convention and the natural one for a rating, but it means a team idle through an
   international break carries an undecayed state into its next fixture.
3. **The filter re-runs at every barrier, using that barrier's level fit.** So a 1998 match is
   scored against a level a forecaster standing in 1998 would not have had. Nothing dated at or
   after the barrier enters anything, so the acceptance instrument sees no leakage; states are
   filtered, levels are smoothed. Standard practice, and stated because it looks like leakage at a
   glance.
4. **The arm gets its own likelihood half-life, tuned separately.** Decay and dynamics are
   *substitutes* — both exist to track time variation — so handing this arm the production 730
   days would make the comparison a test of whose hyperparameter happened to suit whom. Exactly the
   argument that gave `elo-dc` its own half-life. If the tuning wants a much longer half-life here,
   that is itself the finding: the states would be doing the forgetting instead.

### Tuning protocol

`K`, `B`, `e` and the half-life are selected on `backtest.tuning_span` (1996-97..2005-06), the
window neither evaluation span touches, by one shallow coordinate pass — the same protocol the
production half-life and `elo-dc`'s K came from. `K = 0` is included as a grid point and must
reproduce the baseline's tuning-span RPS exactly, which makes the sweep self-checking. A winner at
a grid edge is a red flag to widen on, not a value to adopt.

### Pre-registered bar

The standing rule: paired-bootstrap RPS delta favourable (95% CI excludes 0 **or** P(better) ≥ 0.95)
**and** no degradation against the Shin de-vigged market on the odds-covered subset. DM as
corroboration. One arm, so no FDR correction applies within this family.

Two confirmatory runs, both fixed now: the **test span** (2016-17..2025-26) is the gate, and the
**sensitivity span** (2006-07..2015-16) is reported alongside it. A dynamics parameter tuned on
2003-2006 that only works on one of two later decades is a tuning artefact, and the second run is
what would show it.

### A pre-registered sub-analysis the literature does not have

The research report's §B says this outright: the theoretical case for dynamic models is that a
state-space update *localises* information — A beating B updates A and B, where a weighted
likelihood lets that result contaminate unrelated team C — and that this predicts the dynamic edge
is concentrated exactly where a fitted-then-frozen strength is most out of date. It then notes that
**no paper it found isolates the effect size by regime**, and calls that a gap worth a
pre-registered sub-analysis.

So the pool is partitioned three ways, by keys that are known before kickoff and never read off a
result:

| regime | definition | why |
|---|---|---|
| `early_season` | either side inside its first **6** matches of the season | the model is carrying last summer's squad into this season |
| `post_january_window` | any match in **February** | the month after the window shuts, when mid-season signings first play |
| `settled` | everything else | the regime where the static fit is at its freshest |

Neither threshold is tuned and neither gates anything — this is a diagnostic, reported with a
paired delta per regime. It lives in `eval/slices.py` so every future arm gets it too.

**The directional prediction:** the delta is more favourable in `early_season` than in `settled`.
If dynamics help *uniformly*, the localisation story is not the mechanism and something else is
driving whatever gain appears. If they help *only* in `settled`, the mechanism is backwards and I
should not believe the arm even if it passes.

### What each outcome will mean

* **Accept** — the first arm this project has accepted, and it would say the production model's
  single mechanism for time variation is genuinely too blunt: form exists within a season and a
  decayed constant cannot represent it.
* **Null with visible states** — the reportable one. If the state dispersion is materially non-zero
  and the forecast still does not improve, the finding is that the 730-day decay already captures
  what is capturable, and that the extra structure is re-expressing rather than adding. That is the
  same shape as Arm 4's result, where a 0.46 pool weight bought 0.0004 RPS, and it would make two
  independent demonstrations of the same thing.
* **Null with states near zero** — a broken experiment, not a result. The `dynamics` block in the
  report exists to tell these two apart, and it is this arm's analogue of Arm 4's pool weight.

### Prediction, recorded to be scored later

**−0.0010 to −0.0025 on the test span, and I put 60% on it clearing gate 1.**

Higher than my usual because the mechanism is the one thing the baseline structurally *cannot* do —
Arms 1, 2 and 4 all added parameters to a structure that could already express something close, and
all three were null. A within-season form state is a genuinely new degree of freedom.

Lower than the literature's −0.003 for three reasons, one of them about me: the two-stage estimator
gives up efficiency the joint one has; the level fit is refit at every one of 1,153 barriers where
Koopman & Lit's is not; and **three arms have now produced three magnitude misses, all in the same
direction — I over-predict effect sizes in this problem.** Shading a −0.003 literature figure down
to −0.002 is that correction applied deliberately rather than after the fact.

**Two things already seen, declared so they are not mistaken for foresight.** An untuned smoke run
on 2024-25 alone (K = 0.01, B = 0.95, 380 matches) scored 0.2077 against the baseline's 0.2110 — a
−0.0033 that is one season with no significance attached and no tuning behind it. And the first
three cells of the tuning sweep came in at −0.0002 to −0.0004 against the inert reference, an order
of magnitude smaller. The prediction above sits between them, closer to the sweep, because the
sweep is 342 barriers and the smoke run is one season.

**Also predicted:** the tuning will want a **longer** half-life than 730 days for this arm, because
the states take over the job the decay was doing. If it instead wants a shorter one, the two
mechanisms are complements rather than substitutes and I have the mechanism wrong.

---

## 2026-08-18 — RESULT, Arm 3: ACCEPT — the first arm to pass, and its mechanism is not the one predicted

**Test span (the gate), 2016-17..2025-26, 3,800 matches over 1,153 barriers:**

```
arm                   RPS  log loss   skill  vs market   vs baseline
dixon-coles        0.2005    0.9718   16.1%    +0.0082   (baseline)
dc-gas             0.1986    0.9667   16.9%    +0.0061   -0.0019 [-0.0032, -0.0005] P=0.995
```

**Gate 1 passes**: the 95% CI excludes zero. **Gate 2 passes**: the market gap improves from +0.0082
to +0.0061. Diebold–Mariano corroborates independently (HLN −2.53, p=0.011), and log loss moves the
same way (−0.0051, P=0.988). No state was ever clipped and tau stayed valid at every one of the
~10 million filter updates.

Tuned recursion: `K = 0.03`, `B = 0.99`, `e = 1.0` (inverse-information scaling), likelihood
half-life 2555 days. Mean state dispersion 0.129 in log-goal units — the states are saying
something, which is the precondition for the result meaning anything either way.

### The pre-registered mechanism test failed, and then failed again

This is the part that matters more than the verdict.

| regime | test span delta | P | sensitivity span delta | P |
|---|---|---|---|---|
| `early_season` | **−0.00000** | 0.495 | +0.00047 | 0.379 |
| `post_january_window` | +0.00118 | 0.288 | −0.00054 | 0.616 |
| `settled` | **−0.00270** | 0.999 | **−0.00114** | 0.957 |

The report's §B argues that a state-space model wins by *localising* information, and that this
predicts the dynamic edge is concentrated where a fitted-then-frozen strength is most out of date —
early season, post-transfer-window. It notes no paper isolates this, and calls it a gap worth a
pre-registered sub-analysis. So it was pre-registered.

**The gain is entirely in `settled` and exactly zero in `early_season`, on both decades
independently.** The localisation story, as a story about *when* the edge appears, is wrong here.

I also pre-registered this sentence: *"If they help only in `settled`, the mechanism is backwards
and I should not believe the arm even if it passes."* That sentence has fired, and walking it back
is exactly the move the pre-registration exists to prevent, so it gets argued rather than dropped.

**The sentence was wrong, and the tuning said so before the result did.** It assumed the states
would be a *form* state — fast-moving, most valuable when the level is stale. The tuning chose
`B = 0.99`, a memory of about 69 team-matches, or nearly two seasons. A state that takes two
seasons to move **cannot** express itself inside a team's first six matches; in the opening weeks
it is still carrying what it learned last season, which the level fit already knows. The regime
result is what that parameter *predicts*, and I failed to notice the implication when I read the
tuning output.

So the mechanism is not "catch form the decay missed". It is: **the level becomes a long-run
quality estimate over seven years of history, and the state carries where a club has drifted
relative to it.** That is a different claim from the report's, it is consistent with everything
else measured here, and it was reached by running the sub-analysis the report said nobody had run.

### The half-life crossover, which is the strongest evidence for that reading

The two mechanisms are substitutes, so the arm was tuned with its own likelihood decay. Scoring the
**static baseline at the same half-lives** turns a hyperparameter into a finding:

| half-life | `dc-gas` | static baseline |
|---|---|---|
| 730 (production) | — | **0.20048** |
| 1825 | 0.19849 | 0.20138 |
| **2555** | **0.19825** | 0.20156 |
| 3650 | 0.19837 | 0.20212 |
| 36500 (no decay) | 0.19875 | 0.20362 |

**A longer memory hurts the static model monotonically and helps the dynamic one.** Without states,
distant matches are noise about a club that no longer exists; with them, the level can be estimated
on everything while the state carries what changed. The states are what make old data usable, and
that is a statement about *what* the dynamics do rather than *when* they do it.

This is also the second time this project has watched a model with its own recency mechanism ask
the layer above it to stop forgetting. `elo-dc` did the same and ran monotone to no decay at all.

### The arm moves two axes, so the two axes were separated

`dc-gas` differs from the production model in two ways: it has states, and it fits its level with a
2555-day half-life rather than 730. That is deliberate and precedented — decay and dynamics are
substitutes, so each model is tuned on its own terms, exactly as `elo-dc` was — but it means the
headline is a two-axis change and the split has to be measured rather than asserted. A failing
inertness test is what forced the issue: zeroing the loading alone does *not* reproduce the
baseline, because the level fits still differ.

Three configurations on the identical walk, identical splits, identical matches:

```
configuration               RPS  vs market   vs production baseline
dixon-coles @ 730       0.20047   +0.00824   (baseline)
dixon-coles @ 2555      0.20414   +0.01218   +0.00367 [+0.00227, +0.00506] P=0.000
dc-gas      @ 2555      0.19860   +0.00607   -0.00187 [-0.00322, -0.00048] P=0.995

states alone (dc-gas @ 2555 vs dixon-coles @ 2555): -0.00555 [-0.00751, -0.00354] P=1.000
```

**The longer memory is a handicap the arm carries, not an advantage it borrows.** Stripped of its
states, a model fitted on seven years is worse than the production baseline by +0.0037 — nearly
twice the size of Arm 3's entire net gain, and decisively so. The tuning window showed the same
sign at +0.0011; on the test decade it is three times larger.

So the honest decomposition of the −0.0019 headline is: **−0.0056 from the states, +0.0037 given
back to the half-life.** Measured against a static model fitted the same way, the dynamics are
worth about three times what the arm's gate number shows.

That also sharpens the mechanism claim. It is not merely that a long memory becomes *tolerable*
once states exist — on this decade a seven-year memory is actively bad without them, and the states
turn it into the better configuration. The level and the state are doing genuinely different jobs:
long-run quality, and drift away from it.

**A question this raises and deliberately does not answer:** would `dc-gas` do better at 730 days,
given how much larger the static half-life penalty is on the test decade than on the tuning window?
That question cannot be answered here. Selecting a half-life on the test span is tuning against the
acceptance instrument, which is the one thing this project's protocol forbids outright, and knowing
the number is most of the way to being influenced by it. It is therefore recorded as the first item
for the retune the acceptance rule already mandates — to be run on `tuning_span`, where the arm's
half-life was selected against a grid that stopped at 1825 and never saw the interaction.

### Robustness, and the caveat that belongs next to the headline

**The sensitivity span (2006-07..2015-16) does not clear gate 1**: −0.0008 [−0.0020, +0.0003],
P=0.927, DM p=0.117. Same sign, half the magnitude, not significant. Gate 2 is not assessable there
— `avg_closing` begins in 2019/20.

**A season jackknife on the test span** (point estimates; the CI is not recomputed):

| dropped season | pooled delta over the other nine |
|---|---|
| none | −0.00187 |
| 2022-23 (the arm's best) | **−0.00109** |
| 2024-25 | −0.00154 |
| 2020-21 (the arm's worst) | −0.00226 |

One season carries about half the pooled effect. Drop it and what remains — −0.0011 — is
essentially the sensitivity span's −0.0008.

**Sign test across all twenty scored seasons**: the arm is ahead in 14 of 20 (8 of 10 on the test
decade, 6 of 10 on the earlier one), one-sided p = 0.058. Suggestive, not decisive, and
non-parametric, so it does not lean on the bootstrap.

Putting those three together, the honest estimate is that **the true effect is nearer −0.001 than
−0.002**, and that the test span's −0.0019 sits at the optimistic end of its own confidence
interval's implication. The arm passes the rule the project committed to in advance. It is not a
large effect and the entry should not be read as claiming one.

### Leakage was hunted before the result was believed

A single-season dry run came in at −0.0075, which is the signature this project treats as a bug
until proven otherwise. Three checks, all before the confirmatory run:

* **It does not beat the market.** Market 0.1962, `dc-gas` 0.2035 on 2024-25.
* **The baseline was unusually stale that season** — +0.0148 against the market versus its decade
  norm of +0.0082. 2024-25 is where a frozen strength should fail, and it did.
* **The placebo.** The filter was fed the same fixtures with the results randomly permuted, so
  every team kept its schedule and the scores carried no information about who played. Real results
  gain 0.0075; **scrambled results lose 0.0061.** A gain that survived scrambling would have been
  structural and the arm would have been wrong however good its numbers looked.

The placebo is now a standing test rather than a one-off.

### The tuning hit a grid edge and was not adopted until it did not

The first pass put the half-life at 1825, the top of the standard grid. The project's own rule is
that a grid-edge winner means the search never bracketed the optimum. Widened to 36500 the curve
turns over at 2555, so the adopted value is a real interior optimum. `K` and `B` are interior too.
`K` is weakly identified between 0.02 and 0.03 — they differ by 0.00001 RPS — so read it as "about
0.025"; 0.03 is the grid winner and is taken for that reason alone.

### Scoring the prediction

I predicted **−0.0010 to −0.0025 with 60% on clearing gate 1**. The result is **−0.0019, and it
cleared**. First magnitude hit after three consecutive misses, and it came directly from the
correction recorded at the end of Arm 4 — the literature said −0.003, I shaded it down because I
had over-predicted three times running, and the shading was right. That correction has now paid
for itself once.

I also predicted the tuning would want a **longer** half-life because the states take over the
forgetting. It wanted the longest available and then some. Right for the right reason.

And I got the regime prediction **wrong**, in the specific way described above. Two right, one
wrong, and the wrong one is the more interesting.

### Production status: NOT wired

The acceptance rule's own last line: *"An accepted variant earns a hyperparameter retune BEFORE
production wiring."* So `model.seams.dynamics.enabled` stays `false`. The retune is the next piece
of work, and it is a joint one — the production half-life was selected against a model with no
states, and this arm has just demonstrated that the presence of states changes which half-life is
right by a factor of three and a half.

### Three changes to the standing instrument, all additive

* **`slice_staleness`** in `eval/slices.py` — the regime partition above, defined causally from the
  calendar and from how many matches each side has played. Every future arm gets it.
* **`eval/cache.py`** — a content-addressed store for a completed walk, keyed on the arm, every
  split boundary, the identity of every match in the pool, and the **effective** configuration. A
  mismatched entry is not stale, it is wrong, and it is ignored rather than repaired. It lets the
  walk and the statistics run as separate processes, and it makes a sub-analysis thought of the
  next day cost seconds instead of minutes. The fingerprint covers configuration, not source, so
  the standing discipline is: **delete the cache when model or eval code changes.**
* **`tests/conftest.py`** — pins the BLAS thread count. See below; this is the one that mattered.

Neither the slice nor the cache changes a verdict. What was *not* done to make the runs fit:
`n_boot` stays at 10,000 and `paired_delta` stays byte-identical to the WC2026 port. Halving the
bootstrap would have made everything fit and would have quietly made this arm's interval
incomparable to every other arm's — buying a green result by degrading the instrument that decides
it.

### The memory failures had nothing to do with the model, and my first two diagnoses were wrong

Four runs died with `MemoryError` inside the paired bootstrap, which needs two contiguous ~300 MB
arrays. Three explanations were tried in order, and only the last one was right:

1. *"The corpus is 193 columns wide and the splitter copies it per barrier."* True, and worth
   fixing — but trimming a frame **after** loading it reclaims almost nothing, because pandas has
   already materialised every block. The projection has to happen in the read. Real, not the cause.
2. *"The process grows during the walk, so give the statistics a clean heap."* Measurably false.
   Instrumenting commit at each stage showed the process was already ~1.1 GB heavy **at import**,
   before a single fit had run. The walk itself was nearly free.
3. **OpenBLAS.** It allocates per-thread scratch buffers when it initialises, sized for the machine
   rather than for the work. On this 16-core host, importing numpy/pandas/scipy/plmodel costs
   **1,106 MB of commit**; with the thread count pinned to 1 it costs **84 MB**. A gigabyte of
   scratch space for matrix operations this project never performs — the largest array the model
   ever touches is the ~50x50 of a league's attack and defence parameters, far below the size where
   threading pays, and the walk is a long sequence of tiny problems rather than a few large ones.

Pinning is therefore not a workaround, it is the right configuration for this workload, and it
removes a source of run-to-run variation in floating-point summation order as a side effect — worth
having in a project that asserts byte-identical output in six places. Set as environment variables
in `tests/conftest.py` rather than through `threadpoolctl`, because the allocation happens at import
time: by the time Python can call a library function the memory is already committed.

The whole suite, integration included, is green with the pin in place, and every Arm 3 number above
was computed **without** it — so the byte-identity and determinism tests are also confirming the pin
changed no arithmetic.

**The lesson worth keeping:** two plausible diagnoses cost an hour, and both were plausible enough
to act on without measuring. The third took one instrumented run to find. Measure the resource
before optimising the code that appears to consume it.

---

## 2026-08-19 — RETUNE, Arm 3: a more thorough search won the tuning window and lost the test span

The acceptance rule's last line: *"An accepted variant earns a hyperparameter retune BEFORE
production wiring."* Two reasons it was not a formality here — the first values came from **one**
shallow coordinate pass, and the half-life was never re-opened after `K` and `B` moved underneath
it.

So: coordinate cycles run to convergence on `backtest.tuning_span` and nowhere else, every axis
reported for grid-edge status.

### What moved

| | in force | retuned |
|---|---|---|
| `score_loading` K | 0.03 | **0.025** |
| `persistence` B | 0.99 | **0.995** |
| `scaling_exponent` e | 1.0 | **1.5** |
| half-life | 2555 | **7300** |
| tuning-window RPS | 0.19806 | **0.19780** |

Converged in two cycles. All four axes interior at the end.

**The exponent needed the grid widened, and the widening paid.** The first grid was `{0, 0.5, 1}` —
the range the Creal–Koopman–Lucas family is conventionally *written* in — and 1.0 won at its edge.
Extended, **1.5 wins outright (0.19780) and 2.0 is worse (0.19791)**, so it is a genuine interior
optimum the conventional grid could not see. The grid had encoded a notational convention as if it
were a constraint on the model. This project's edge rule has now caught the same class of error
twice in one arm.

A unit test had encoded the same convention (`assert exponent in (0, 0.5, 1)`) and failed the
moment the grid widened. It has been replaced with the real requirement — non-negative, since a
negative exponent would make a surprise in a high-scoring match count for *more*, inverting the
scaling the family exists to apply.

### And then the retuned configuration scored WORSE on the test span

```
configuration                          test-span delta vs baseline        DM p
first values  (K.03 B.99 e1.0 2555d)   -0.0019 [-0.0032, -0.0005] P=0.995  0.011
retuned       (K.025 B.995 e1.5 7300d) -0.0013 [-0.0026, -0.0000] P=0.976  0.065
```

The retune improved the tuning window by **−0.00027** and degraded the test span by **+0.0006**.

That is not a contradiction, it is what a flat plateau does. The half-life curve reads 0.19781 at
2555, 0.19780 at 5475, 0.19780 at 7300 — a **three-way tie inside 0.00001 RPS**. A more thorough
search over a surface that flat is not extracting more signal, it is being given more opportunity
to fit the tuning window's own noise, and the half-life it picks then has to survive on a different
decade where the same axis is expensive. The decomposition already showed why the axis is
dangerous: a static model at 2555 days is +0.0037 worse on the test decade than at 730, and the
penalty grows with the memory.

**The retuned values ship anyway, and that is the point.** Both configurations were selected on the
tuning window and both are protocol-legal. Choosing between them on test-span evidence would be
selecting a hyperparameter against the acceptance instrument, which is the one thing this project
forbids outright — and it would be far more corrosive than the 0.0006 it would buy. The rule
decides, not the number it produces.

Recorded so the number is not quietly improved later: **the shipping configuration's gate-1 margin
is thin.** The 95% CI upper bound is −0.000008, P(better) = 0.976, DM p = 0.065. It passes, and it
passes by less than the configuration it replaced.

### What survives both configurations

The things that do not depend on which of the two ships are the things worth trusting:

* **both pass both gates** — acceptance is robust to the choice;
* **gate 2 improves either way** — the market gap goes +0.0082 → +0.0061 or +0.0068;
* **the regime pattern replicates a third time** — `settled` −0.0020 (P=0.994), `early_season`
  +0.0001, `post_january_window` +0.0014. Three independent runs across two decades now put the
  entire gain outside the regimes where the literature predicts it;
* **2022-23 still carries about half the effect** (−0.0085 of a −0.0013 pooled mean);
* **the states are still worth several times the headline** against a static model fitted at the
  same half-life: −0.0049 on the tuning window at 7300 days.

### The generalisable lesson

More hyperparameter search is not free, and on a flat surface it is worse than free. The first,
cruder search happened to land on a better point for the era being forecast; the thorough one
climbed 0.00027 of tuning-window noise and gave back twice that on the test decade. **The defence
is not to search less — it is to read a plateau as a plateau.** The half-life is documented in
`config.yaml` as "long, and barely identified above 2555", not as 7300, for exactly this reason,
in the same spirit as the production half-life's "somewhere in 240-1095".

### Production wiring is now permitted, and has not been done

`model.seams.dynamics.enabled` remains `false`. The rule's precondition is satisfied — the retune
has happened — so wiring is now a decision rather than a blocker. It is left open deliberately
because it is not a small change: promoting `dc-gas` to production changes the baseline every
remaining arm is measured against, and the `seams_are_inert()` contract plus the seam-inertness
tests are written around a production model with every seam off. That restructuring should be a
deliberate step, not a side effect of a retune.
