# Premier League Forecasting System

A Premier League match-outcome and season-simulation forecaster. The primary output is a
**falsification machine**: a walk-forward acceptance harness sharp enough to kill bad ideas
honestly, with a production Dixon–Coles model attached. The model is the by-product; the harness
is the deliverable.

## The acceptance rule

Every candidate change is judged by one standing instrument, stated in `config.yaml` and embedded
verbatim in every harness report:

```
Accept a candidate change iff BOTH:
  (1) its paired-bootstrap RPS delta vs baseline on the test pool is favourable
      (95% CI excludes 0, OR P(better) >= 0.95), AND
  (2) its delta vs the Shin de-vigged market on the odds-covered subset does not degrade.
An accepted variant earns a hyperparameter retune BEFORE production wiring.
```

Two gates, not one: a change can improve the measuring instrument's own distribution while
degrading the target distribution. Gate 2 exists to catch that.

## Ground rules

- **Leak-freedom is structural.** A hard `LeakageError` fires on every split when
  `max(train_date) >= min(test_date)`. Feature providers may use only information dated strictly
  before their `asof` barrier. No k-fold anywhere, ever — it leaks future into past for time series.
- **No magic numbers in code.** Every tunable lives in `config.yaml` with an inline justification.
  Enforced by `tests/test_no_magic_numbers.py`, not by discipline.
- **Every seam proves it is inert** — byte-identical output when switched off.
- **The harness proves an arm does something** — a broken experiment returning "no effect" is
  otherwise indistinguishable from a correct one.
- **Missing data is a value, not a hole.** Never imputed; counted and reported per feature.
- **`NOTES.md` is a dated decision ledger** — hypothesis, pre-registered bar, result with CI,
  verdict. No phase labels.

## Commands

```
pl config      show the loaded config and the acceptance rule
pl ingest      fetch + cache + schema-validate all sources; emit coverage report
pl fit         fit the production model; dump params, ratings, fixture probabilities
pl backtest    rolling-origin walk-forward; metrics + calibration
pl compare     paired A/B of candidate arms vs baseline, with the market gate
pl simulate    season Monte Carlo -> title / top-4 / relegation / points distribution
               --validate scores those probabilities against every completed season
               --uncertainty point|drift overrides how strength uncertainty propagates
pl reproduce   re-run an external claim on our data
pl audit       calibration slices: promoted-team, big-six-vs-rest, favourite, by season
pl live        freeze a matchweek's forecasts before kickoff; score them after
```

Every command listed above is built.

## Development

```
uv sync --extra dev
uv run pytest -q
```

## Data

Match results, odds and match statistics come from
[football-data.co.uk](https://www.football-data.co.uk/) (E0–E3, 1993/94→present), free for
personal and research use with attribution. xG comes from Understat; player availability from the
official Fantasy Premier League API.

## Expectations

Parity with the market is the goal, not beating it — the PL 1X2 closing line is among the most
efficient markets in the world. Treat any apparent betting edge as a bug until proven otherwise,
and prefer a scoring-rule improvement to a backtest ROI as evidence, always.
