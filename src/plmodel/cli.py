"""The `pl` command-line surface.

A command declared but not yet built exits with a message saying so — an honest stub is better
than a missing command, because `pl --help` then documents the intended surface without pretending
it works. Nothing is stubbed now: every command this project set out to build exists.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from plmodel import __version__
from plmodel.config import ConfigError, load_config

# The intended command surface. Entries are removed from here as they are built, so this mapping
# is both the roadmap and the single place an unbuilt command is documented. Empty means every
# command the project set out to build exists.
_PLANNED: dict[str, str] = {}

# Accepted spellings of "against" in a fixture typed at the command line.
_FIXTURE_SEPARATORS = (" v ", " vs ", " V ", " - ")

# Back-off from the tau validity boundary, matching the model's own prediction-time clamp.
_PREDICT_RHO_MARGIN = 0.01

# A club above the cold-start floor but below this multiple of it has parameters that look
# confident and are not. Hull's 2017 rating is the case this exists to surface.
_STALE_MULTIPLE = 3.0

# External claims this project has re-run on its own data, by paper id.
_REPRODUCIBLE: tuple[str, ...] = ("pitcan2026",)


def cmd_config(args: argparse.Namespace) -> int:
    """Print the loaded config's identity — the smoke test that config.yaml parses."""
    cfg = load_config(args.config)
    print(f"config      : {args.config or 'config.yaml'}")
    print(f"seed        : {cfg.seed}")
    print(f"divisions   : {', '.join(cfg.data.divisions)} from {cfg.data.first_season}")
    print(f"cache_dir   : {cfg.cache_dir}")
    print("\nacceptance rule:")
    for line in cfg.acceptance_rule.splitlines():
        print(f"  {line}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Fetch, validate and cache the match corpus; write the coverage report."""
    from plmodel.data.coverage import build_report
    from plmodel.data.football_data import load_matches

    cfg = load_config(args.config)
    divisions = tuple(args.divisions.split(",")) if args.divisions else None
    corpus, metas = load_matches(cfg, divisions=divisions, refresh=args.refresh)

    # Expected goals is a separate source with its own licence, coverage and provenance, so it is
    # attached only when its archive is present. `--with-xg` fetches it; otherwise an existing
    # cache is reused and a missing one simply leaves the columns absent.
    from plmodel.data import xg as xg_source

    xg_coverage = None
    if args.with_xg or xg_source.cache_path(cfg).exists():
        if args.with_xg:
            xg_source.download(cfg, refresh=args.refresh)
        corpus = xg_source.attach(corpus, xg_source.load_raw(cfg))
        xg_coverage = xg_source.coverage_summary(corpus)

    report = build_report(
        corpus, metas, skipped=corpus.attrs.get("skipped_seasons"), cfg=cfg
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if xg_coverage is not None:
        report["expected_goals"] = xg_coverage
    report_path = cfg.output_dir / "coverage.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    parquet_path = cfg.cache_dir / "matches.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(parquet_path, index=False)

    totals = report["totals"]
    print(f"matches   : {totals['n_played']:,} played of {totals['n_rows']:,} rows")
    print(f"span      : {totals['date_min']} -> {totals['date_max']} "
          f"({totals['n_seasons']} seasons, {totals['n_teams']} teams)")
    for row in report["by_division"]:
        print(f"  {row['division']}: {row['n_matches']:,} matches over {row['n_seasons']} seasons")
    if report["skipped_seasons"]:
        print(f"\nskipped {len(report['skipped_seasons'])} season file(s):")
        for line in report["skipped_seasons"]:
            print(f"  {line}")
    market = report["market_benchmark"]
    print(f"\nmarket benchmark: {market['gate_benchmark']} "
          f"(de-vig {market['devig_primary']}, sensitivity {market['devig_sensitivity']})")
    for name, block in market["families"].items():
        role = "GATE" if name == market["gate_benchmark"] else (
            "diag" if name in market["diagnostic_benchmarks"] else "    ")
        span = f"{block['first_priced']} -> {block['last_priced']}" if block["n_priced"] else "-"
        print(f"  {role} {name:18} {block['n_priced']:>6,} priced  {block['settlement']:<10} {span}")

    print("\nknown discontinuities:")
    for d in report["known_discontinuities"]:
        print(f"  {d['date']}  {d['source']}: {d['field']}")
        print(f"            {d['effect']}")
    print(f"\ncoverage  : {report_path}")
    print(f"corpus    : {parquet_path}")
    return 0


def _load_corpus(cfg, *, division: str | None = None):
    """The cached corpus, filtered to the prediction division and played matches."""
    import pandas as pd

    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        raise SystemExit("no cached corpus — run `pl ingest` first")
    corpus = pd.read_parquet(path)
    div = division or cfg.backtest.prediction_division
    played = corpus[(corpus["division"] == div) & corpus["played"]]
    return corpus, played.sort_values("date", kind="stable").reset_index(drop=True)


def _build_splits(cfg, matches, *, span):
    from plmodel.eval.backtest import walk_forward

    return walk_forward(
        matches,
        first_season=span.first_season,
        last_season=span.last_season,
        refit_every=cfg.backtest.refit_every,
        min_train_matches=cfg.backtest.min_train_matches,
    )


def cmd_compare(args: argparse.Namespace) -> int:
    """Paired A/B of candidate arms against the baseline, with both acceptance gates."""
    from plmodel.eval.compare import registered_arms, report_json, run_compare

    cfg = load_config(args.config)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in registered_arms()]
    if unknown:
        print(f"unknown arm(s) {unknown}; registered: {list(registered_arms())}", file=sys.stderr)
        return 2

    corpus, matches = _load_corpus(cfg)
    span = cfg.backtest.sensitivity_span if args.sensitivity else cfg.backtest.test_span
    if args.first_season or args.last_season:
        # Narrowing the pool is sometimes necessary — a channel that only covers part of the span
        # cannot be judged where it does not exist — so it is an explicit flag that lands in the
        # report rather than a quiet edit to config.
        from plmodel.config import SeasonSpan

        span = SeasonSpan(args.first_season or span.first_season,
                          args.last_season or span.last_season)
    splits = _build_splits(cfg, matches, span=span)
    report = run_compare(
        matches, splits, cfg, arm_names,
        history=corpus[corpus["division"] == cfg.backtest.prediction_division],
        n_bins=cfg.audit.calibration_bins,
        big_six=cfg.audit.big_six,
    )

    payload = report_json(report)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / (args.out or "compare.json")
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    _print_compare(payload, report)
    print(f"\nreport    : {out_path}")
    return 0


def _print_compare(payload: dict, report) -> None:
    splits = payload["splits"]
    print(f"walk      : {splits['n_splits']} barriers, {splits['first_barrier']} -> "
          f"{splits['last_barrier']} (refit every {splits['refit_every']})")
    print(f"pool      : {len(report.outcomes):,} matches")

    market = payload["market"] or {}
    if market.get("rps_market") is not None:
        print(f"market    : {market['benchmark']} ({market['devig']}) "
              f"RPS {market['rps_market']:.4f} on {market['n_covered']:,} covered")

    print(f"\n{'arm':16}{'RPS':>9}{'log loss':>10}{'skill':>8}{'draw res':>10}"
          f"{'vs market':>11}   {'vs baseline'}")
    for name, block in payload["arms"].items():
        pooled = block["pooled"]
        draw = block["calibration"]["draw"]["resolution"]
        delta = block["vs_baseline"]
        vs_base = (
            f"{delta['delta_rps']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] "
            f"P={delta['p_a_better']:.3f}" if delta else "(baseline)"
        )
        vs_mkt = f"{block['vs_market']['delta_rps']:+.4f}" if block["vs_market"] else "-"
        print(f"{name:16}{pooled['rps']:>9.4f}{pooled['log_loss']:>10.4f}"
              f"{pooled['skill']:>8.1%}{draw:>10.5f}{vs_mkt:>11}   {vs_base}")

    if payload["verdicts"]:
        print("\nacceptance rule:")
        for line in payload["acceptance_rule"].splitlines():
            print(f"  {line}")
        print()
        for name, verdict in payload["verdicts"].items():
            gate2 = {True: "pass", False: "FAIL", None: "n/a"}[verdict["gate2_vs_market"]]
            mark = "ACCEPT" if verdict["accepted"] else "reject"
            print(f"  {mark:7} {name:16} gate1 "
                  f"{'pass' if verdict['gate1_vs_baseline'] else 'FAIL'} "
                  f"({verdict['gate1_reason']}), gate2 {gate2}")


def cmd_audit(args: argparse.Namespace) -> int:
    """Calibration slices for one arm: promoted-team, big-six, favourite, by season."""
    from plmodel.eval.compare import registered_arms, run_compare

    cfg = load_config(args.config)
    if args.arm not in registered_arms():
        print(f"unknown arm {args.arm!r}; registered: {list(registered_arms())}", file=sys.stderr)
        return 2

    corpus, matches = _load_corpus(cfg)
    splits = _build_splits(cfg, matches, span=cfg.backtest.test_span)
    report = run_compare(
        matches, splits, cfg, [args.arm],
        history=corpus[corpus["division"] == cfg.backtest.prediction_division],
        n_bins=cfg.audit.calibration_bins,
        big_six=cfg.audit.big_six,
    )
    arm = report.arms[0]

    print(f"arm       : {arm.name}   pooled RPS {arm.pooled['rps']:.4f} "
          f"on {arm.pooled['n']:,} matches\n")
    print(arm.slices.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\ncalibration by outcome (Murphy decomposition):")
    print(f"  {'outcome':9}{'base rate':>11}{'reliability':>13}{'resolution':>12}{'brier':>9}")
    for outcome, block in arm.calibration.items():
        d = block["decomposition"]
        print(f"  {outcome:9}{d['base_rate']:>11.4f}{d['reliability']:>13.5f}"
              f"{d['resolution']:>12.5f}{d['brier']:>9.4f}")
    print("\n  Draw resolution is expected to stay flat: nothing in the literature moves it.")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / f"audit_{arm.name}.json"
    out_path.write_text(
        json.dumps(
            {
                "arm": arm.name,
                "pooled": arm.pooled,
                "calibration": arm.calibration,
                "slices": arm.slices.to_dict("records"),
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nreport    : {out_path}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit the production model on everything available, and dump what it believes."""
    import pandas as pd

    from plmodel.model.dixon_coles import fit_dixon_coles, fit_summary

    cfg = load_config(args.config)
    _, matches = _load_corpus(cfg)
    ref = pd.Timestamp(args.asof) if args.asof else matches["date"].max() + pd.Timedelta(days=1)
    train = matches[matches["date"] < ref]
    if train.empty:
        print(f"no matches before {ref.date()}", file=sys.stderr)
        return 2

    fit = fit_dixon_coles(
        train,
        half_life_days=args.half_life or cfg.model.decay_half_life_days,
        ref_date=ref,
        max_goals=cfg.model.max_goals,
        param_bounds=cfg.model.param_bounds,
        min_effective_share=cfg.model.min_effective_share,
        max_iter=cfg.model.max_iter,
    )
    summary = fit_summary(fit)

    print(f"as of     : {ref.date()}   ({fit.n_obs:,} matches, effective {fit.effective_n:.0f})")
    print(f"half-life : {fit.half_life_days:.0f} days")
    print(f"intercept : {fit.intercept:+.4f}   home advantage {fit.home_advantage:+.4f} "
          f"(x{math.exp(fit.home_advantage):.3f} on the home rate)")
    print(f"rho       : {fit.rho:+.4f} ({summary['rho_sign']})")
    print(f"converged : {fit.converged} in {fit.n_iterations} iterations")
    if fit.cold_start_teams:
        print(f"cold start: {len(fit.cold_start_teams)} team(s) pinned at league average")

    table = fit.team_table()
    current = set(matches[matches["season"] == matches["season"].max()]["home_team"])
    table = table[table["team"].isin(current)]
    print(f"\n{'team':18}{'attack':>9}{'defence':>9}{'exp goals for':>15}{'against':>9}")
    for row in table.itertuples(index=False):
        lam = math.exp(fit.intercept + fit.home_advantage + row.attack)
        mu = math.exp(fit.intercept - row.defence)
        print(f"{row.team:18}{row.attack:>+9.3f}{row.defence:>+9.3f}{lam:>15.2f}{mu:>9.2f}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / "fit.json"
    out_path.write_text(
        json.dumps({"summary": summary, "teams": fit.team_table().to_dict("records")},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nparams    : {out_path}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Walk-forward the production model and report against the baseline and the market."""
    from plmodel.eval.compare import report_json, run_compare

    cfg = load_config(args.config)
    corpus, matches = _load_corpus(cfg)
    span = cfg.backtest.sensitivity_span if args.sensitivity else cfg.backtest.test_span
    splits = _build_splits(cfg, matches, span=span)
    report = run_compare(
        matches, splits, cfg, ["home-rate", "dixon-coles"],
        history=corpus[corpus["division"] == cfg.backtest.prediction_division],
        n_bins=cfg.audit.calibration_bins, big_six=cfg.audit.big_six,
    )
    payload = report_json(report)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / (args.out or "backtest.json")
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    _print_compare(payload, report)
    dc = payload["arms"].get("dixon-coles", {})
    if dc.get("fit"):
        f = dc["fit"]
        print(f"\nfits      : {f['n_fits']} ({f['n_converged']} converged, "
              f"{f['mean_iterations']:.1f} mean iterations), half-life {f['half_life_days']:.0f}d")
        print(f"rho       : mean {f['rho']['mean']:+.4f}, "
              f"{f['rho']['share_negative']:.0%} of fits negative "
              f"[{f['rho']['min']:+.4f}, {f['rho']['max']:+.4f}]")
        print(f"home adv  : mean {f['home_advantage']['mean']:+.4f} "
              f"({f['home_advantage']['first']:+.4f} -> {f['home_advantage']['last']:+.4f})")
        print(f"cold start: {f['cold_start']['mean_per_fit']:.1f} teams/fit, "
              f"rho clamped {f['rho_clamped']} time(s)")
    if payload["market"] and payload["market"].get("rps_market") is not None:
        gap = dc.get("vs_market", {}).get("delta_rps")
        if gap is not None:
            print(f"\nmarket gap: {gap:+.4f} "
                  f"(model {dc['pooled']['rps']:.4f} vs market "
                  f"{payload['market']['rps_market']:.4f})")
            print("  A materially SMALLER gap than +0.006 is a leakage signal, not a triumph.")
    print(f"\nreport    : {out_path}")
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Re-run an external claim on our own data."""
    from plmodel.reproduce import pitcan2026

    if args.paper != pitcan2026.PAPER_ID:
        print(f"unknown paper {args.paper!r}; available: {list(_REPRODUCIBLE)}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    _, matches = _load_corpus(cfg)
    result = pitcan2026.run(matches, cfg, market_family=args.market or pitcan2026.MARKET_FAMILY)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / f"reproduce_{args.paper}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    paper = result["paper_results"]
    windows, verdict = result["windows"], result["verdict"]
    print(f"{result['citation']}\n")
    print(f"market    : {result['market_family']} ({result['devig']}), "
          f"half-life {result['half_life_days']:.0f}d")
    for name in ("validation", "test"):
        w = windows[name]
        print(f"{name:10}: {w['span'][0]}..{w['span'][1]}  n={w['n']:,} "
              f"({w['n_covered']:,} priced)   paper n={w['paper_n']:,}")

    print(f"\npool weights fitted on validation{'':6}{'ours':>10}{'paper':>10}")
    for label, key, model in (
        ("market + goals  (weight on goals)", "goals_vs_market", "goals"),
        ("market + shots  (weight on shots)", "shots_vs_market", "shots"),
        ("goals  + shots  (weight on shots)", "shots_vs_goals", "shots"),
    ):
        ours = verdict["weights"][key]
        theirs = verdict["paper_weights"][key]
        print(f"  {label:38}{ours:>8.3f}{theirs:>10.2f}")

    three = result["validation_pools"]["three_way"]
    print(f"  {'market + goals + shots (simplex)':38}"
          f"{three['market']:>8.3f}/{three['goals']:.3f}/{three['shots']:.3f}"
          f"{'1.00/0.00/0.00':>16}")

    profile = result["validation_pools"]["goals_profile_admissible"]
    print(f"\nboundary  : argmin w = {profile['argmin_weight']:.3f}, "
          f"log loss monotone increasing over [0,1] = {profile['monotone_increasing']}")
    wide = result["validation_pools"]["goals_profile"]
    print(f"            unconstrained argmin over [-1,1] = {wide['argmin_weight']:+.3f} "
          f"(paper {paper['unconstrained_weight']:+.3f})")

    scores = result["test_scores"]
    print(f"\ntest scores on {scores['n_covered']:,} priced matches"
          f"{'':10}{'ours':>10}{'paper':>10}")
    print(f"  {'market RPS':46}{scores['market']['rps']:>10.4f}{paper['rps_market']:>10.4f}")
    print(f"  {'goals model RPS':46}{scores['goals']['rps']:>10.4f}{paper['rps_goals']:>10.4f}")
    print(f"  {'shots model RPS':46}{scores['shots']['rps']:>10.4f}{'-':>10}")
    gap = scores["goals_vs_market"]
    print(f"  {'gap (goals - market)':46}{gap['delta_rps']:>+10.4f}{paper['gap']:>+10.4f}")
    print(f"  {'  95% CI':46}[{gap['ci_low']:+.4f}, {gap['ci_high']:+.4f}]"
          f"   [{paper['gap_ci'][0]:+.4f}, {paper['gap_ci'][1]:+.4f}]")

    if result.get("half_life_sensitivity"):
        print(f"\ngoals+shots weight on SHOTS, by decay rate (the market is not in this pool):")
        print(f"  {'half-life':>10}{'w_shots':>10}{'goals RPS':>12}{'shots RPS':>12}")
        for row in result["half_life_sensitivity"]:
            print(f"  {row['half_life_days']:>10.0f}{row['weight_on_shots']:>10.3f}"
                  f"{row['rps_goals']:>12.4f}{row['rps_shots']:>12.4f}")
        if verdict.get("paper_weight_inside_our_range"):
            print(f"  The paper's {paper['goals_plus_shots']['shots']:.2f} lies inside this range: "
                  "the chance-creation channel simply wants a shorter memory than the goals one.")

    print(f"\nreproduces the Serie A pattern: {verdict['reproduces']}")
    print(f"  goals adds nothing to the price     : {verdict['goals_earns_zero_against_market']}")
    print(f"  shots informative vs the goals model: {verdict['shots_informative_against_goals']}")
    print(f"  zero is a genuine minimum, not a bound: {verdict['boundary_is_genuine']}")
    print(f"\nxG arm gate: {verdict['xg_arm_gate']}")
    print(f"\nreport    : {out_path}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Freeze a matchday's forecasts before kickoff, or score previously frozen ones."""
    import pandas as pd

    from plmodel.data.football_data import upcoming_fixtures
    from plmodel.data.teams import TeamNameError
    from plmodel.eval.live import freeze_matchday, next_barrier, score_ledger

    cfg = load_config(args.config)
    corpus, played = _load_corpus(cfg)
    ledger_dir = cfg.output_dir / "live"

    if args.score:
        scored = score_ledger(ledger_dir, played)
        if scored.empty:
            print("nothing to score yet: no frozen forecast has a result.")
            return 0
        print(scored.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return 0

    try:
        fixtures = upcoming_fixtures(cfg, corpus, use_feed=not args.no_feed)
    except TeamNameError as exc:
        # The roster guard, doing its job. Named clubs, not a stack trace, because the fix is one
        # line in team_aliases.yaml and whoever is standing here has a match about to kick off.
        print(f"the fixture feed names a club the roster does not know:\n  {exc}", file=sys.stderr)
        print("\nAdd the mapping to data/static/team_aliases.yaml and run again, or pass "
              "--no-feed to freeze only from the season file.", file=sys.stderr)
        return 2

    if fixtures.empty:
        print(f"no unplayed {cfg.backtest.prediction_division} fixtures known.")
        print("  the season file carries fixtures only once it has results in it, and the "
              "rolling feed lags by a day or two.")
        print("  re-run closer to kickoff, or check https://www.football-data.co.uk/fixtures.csv")
        return 0

    barrier = next_barrier(fixtures)
    day = fixtures[fixtures["date"] == barrier]
    if args.dry_run:
        print(f"next barrier : {pd.Timestamp(barrier).date()}   {len(day)} fixture(s)")
        print(f"source       : {'season file only' if args.no_feed else 'season file + feed'}")
        for row in day.itertuples(index=False):
            print(f"  {row.home_team} v {row.away_team}")
        target = ledger_dir / f"{pd.Timestamp(barrier).date()}.json"
        print(f"\nwould write  : {target}")
        if target.exists():
            print("  ! that file already exists and would NOT be overwritten")
        print("nothing was written (--dry-run).")
        return 0

    frozen = freeze_matchday(
        fixtures, played, cfg, ledger_dir,
        arm_names=[a.strip() for a in args.arms.split(",") if a.strip()],
    )
    if frozen is None:
        print("no unplayed fixtures in the corpus - nothing to freeze.")
        return 0
    path, block = frozen
    print(f"froze {block['n_fixtures']} fixture(s) for {block['barrier']}")
    for row in block["fixtures"]:
        probs = "  ".join(
            f"{arm}: {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}" for arm, p in row["forecasts"].items()
        )
        print(f"  {row['home_team']} v {row['away_team']}   {probs}")
    print(f"\nledger    : {path}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Monte Carlo a season's remaining fixtures, or validate the simulator against history."""
    import pandas as pd

    from plmodel.season.simulate import simulate_season
    from plmodel.season.validate import matchweek_barriers

    cfg = load_config(args.config)
    corpus, matches = _load_corpus(cfg)
    if args.validate:
        return _run_validation(args, cfg, matches)

    season = args.season or str(matches["season"].max())
    division = cfg.backtest.prediction_division
    rows = corpus[(corpus["division"] == division) & (corpus["season"] == season)]
    rows = rows.sort_values("date", kind="stable").reset_index(drop=True)
    if rows.empty:
        print(f"no {division} fixtures for {season}", file=sys.stderr)
        print("The source publishes a season's fixtures shortly before it starts.", file=sys.stderr)
        return 2

    barrier = (pd.Timestamp(args.asof) if args.asof
               else matchweek_barriers(rows, weeks=(args.week,),
                                       fixtures_per_week=cfg.season.fixtures_per_week)[0].date)
    played = rows[(rows["date"] < barrier) & rows["played"]]
    remaining = rows[rows["date"] >= barrier]
    fit = _season_fit(cfg, matches, barrier)
    spec = cfg.season.spec(
        uncertainty=args.uncertainty,
        n_replicates=args.replicates or cfg.season.n_replicates,
    )
    forecast = simulate_season(
        fit, played, remaining, spec=spec, seed=cfg.seed, season=season, barrier=barrier,
        deductions=cfg.season.points_deductions.get(season, {}),
    )
    _print_simulation(forecast, cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / (args.out or f"simulate_{season}.json")
    out_path.write_text(json.dumps(forecast.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"\nreport    : {out_path}")
    return 0


def _season_fit(cfg, matches, barrier):
    """The production fit at a barrier, with the standing strictly-before rule enforced."""
    from plmodel.eval.backtest import training_frame
    from plmodel.model.dixon_coles import fit_dixon_coles

    train = training_frame(matches, barrier)
    if train.empty:
        raise SystemExit(f"no matches before {barrier.date()} to fit on")
    return fit_dixon_coles(
        train, half_life_days=cfg.model.decay_half_life_days, ref_date=barrier,
        max_goals=cfg.model.max_goals, param_bounds=cfg.model.param_bounds,
        min_effective_share=cfg.model.min_effective_share, max_iter=cfg.model.max_iter,
    )


def _print_simulation(forecast, cfg) -> None:
    import pandas as pd

    probabilities = forecast.probabilities
    index = {team: i for i, team in enumerate(forecast.teams)}
    order = [index[t] for t in probabilities["team"]]
    low, high = forecast.points_quantile(0.05)[order], forecast.points_quantile(0.95)[order]
    questions = list(cfg.season.questions)
    banked = forecast.table.set_index("team")["points"]

    print(f"season    : {forecast.season}   barrier {pd.Timestamp(forecast.barrier).date()}")
    print(f"table     : {forecast.n_played} played, {forecast.n_remaining} to play "
          f"(horizon {forecast.horizon:.2f})")
    print(f"replicates: {forecast.n_replicates:,}   uncertainty {forecast.uncertainty}")
    if forecast.diagnostics["n_cold_start"]:
        cold = ", ".join(forecast.diagnostics["cold_start_teams"])
        print(f"cold start: {cold} - pinned at league average, so read those rows with care")

    header = f"{'club':16}{'pts':>5}" + "".join(f"{q:>12}" for q in questions)
    print(f"\n{header}{'mean pts':>10}{'90% band':>12}")
    for row, lo, hi in zip(probabilities.itertuples(index=False), low, high):
        cells = "".join(f"{getattr(row, q):>12.3f}" for q in questions)
        band = f"{lo}-{hi}"
        print(f"{row.team:16}{int(banked[row.team]):>5}{cells}{row.mean_points:>10.1f}{band:>12}")

    ties = forecast.diagnostics["boundary_ties"]
    if any(ties.values()):
        detail = ", ".join(f"{name} {count:,}" for name, count in ties.items() if count)
        print(f"\nplayoff clause: {detail} replicate(s) of {forecast.n_replicates:,} ended level "
              "on points, goal difference and goals scored at a question's boundary, and were "
              "split by a coin as the competition's neutral-venue playoff would be.")


def _run_validation(args: argparse.Namespace, cfg, matches) -> int:
    from plmodel.season.simulate import UNCERTAINTY_DRIFT, UNCERTAINTY_POINT
    from plmodel.season.validate import run_span, summarise

    span = (cfg.backtest.sensitivity_span if args.sensitivity else
            cfg.backtest.tuning_span if args.tuning else cfg.backtest.test_span)
    seasons = tuple(sorted(
        s for s in matches["season"].unique() if span.first_season <= s <= span.last_season
    ))
    replicates = args.replicates or cfg.season.validation_replicates
    specs = {
        UNCERTAINTY_POINT: cfg.season.spec(uncertainty=UNCERTAINTY_POINT, n_replicates=replicates),
        UNCERTAINTY_DRIFT: cfg.season.spec(uncertainty=UNCERTAINTY_DRIFT, n_replicates=replicates),
    }
    print(f"span      : {span.first_season}..{span.last_season}, {len(seasons)} season-years")
    print(f"barriers  : matchweeks {list(cfg.season.validation_weeks)}, "
          f"{replicates:,} replicates each")
    scored = run_span(
        matches, cfg, seasons=seasons, specs=specs, weeks=cfg.season.validation_weeks,
        fixtures_per_week=cfg.season.fixtures_per_week, prob_floor=cfg.season.prob_floor,
        deductions=cfg.season.points_deductions, progress=args.progress,
    )
    summary = summarise(scored, n_boot=cfg.backtest.n_boot, seed=cfg.seed,
                        baseline=UNCERTAINTY_POINT)
    _print_validation(summary, seasons)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # The scored rows, not just the summary: the sweep costs minutes and a later question about
    # a slice should not need it run again.
    scored.to_parquet(cfg.output_dir / (args.out or "simulate_validation.json").replace(
        ".json", "_rows.parquet"))
    payload = {
        "span": [span.first_season, span.last_season],
        "n_seasons": len(seasons),
        "weeks": list(cfg.season.validation_weeks),
        "n_replicates": replicates,
        "acceptance_rule": cfg.acceptance_rule,
        "acceptance_rule_applies": False,
        "why_not": (
            "The acceptance rule governs match forecasts scored by RPS against a de-vigged "
            "market. A season forecast is a different quantity on a different unit, and this "
            "corpus carries no outright market, so the rule's second gate has no analogue here. "
            "What is reused is its construction: a paired bootstrap, clustered on seasons, with "
            "a favourable delta needing a 95% interval that excludes zero or P(better) >= 0.95."
        ),
        "summary": summary,
    }
    out_path = cfg.output_dir / (args.out or "simulate_validation.json")
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nreport    : {out_path}")
    return 0


def _print_validation(summary: dict, seasons: tuple[str, ...]) -> None:
    print(f"\n{'spec':10}{'brier':>9}{'log loss':>10}{'PIT ks':>9}{'tail':>8}{'floored':>9}"
          "   vs point (brier, clustered on seasons)")
    for name, block in summary["specs"].items():
        pit = block["points_pit"]
        delta = block.get("vs_baseline")
        against = (f"{delta['delta']:+.5f} [{delta['ci_low']:+.5f}, {delta['ci_high']:+.5f}] "
                   f"P={delta['p_a_better']:.3f}" if delta else "(baseline)")
        print(f"{name:10}{block['brier']:>9.5f}{block['log_loss']:>10.4f}{pit['ks']:>9.4f}"
              f"{pit['tail_mass']:>8.3f}{block['share_floored']:>9.3f}   {against}")

    print(f"\n{'spec':10}{'points log score':>18}   vs point (clustered on seasons)")
    for name, block in summary["specs"].items():
        delta = block.get("vs_baseline_points")
        against = (f"{delta['delta']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] "
                   f"P={delta['p_a_better']:.3f}" if delta else "(baseline)")
        print(f"{name:10}{block['points_log_score']:>18.4f}   {against}")
    print("\n  A calibrated points forecast puts 0.20 of its PIT mass in the outer tenths; "
          "more than that is a distribution too narrow for what happened.")

    weeks = sorted(next(iter(summary["specs"].values()))["by_week"])
    print(f"\nby horizon (Brier / PIT ks)\n{'spec':10}" +
          "".join(f"{'week ' + str(w):>18}" for w in weeks))
    for name, block in summary["specs"].items():
        cells = "".join(f"{block['by_week'][w]['brier']:>10.5f}"
                        f"{block['by_week'][w]['points_pit']['ks']:>8.3f}" for w in weeks)
        print(f"{name:10}{cells}")

    questions = list(next(iter(summary["specs"].values()))["by_question"])
    print(f"\nby question (Brier)\n{'spec':10}" + "".join(f"{q:>14}" for q in questions))
    for name, block in summary["specs"].items():
        print(f"{name:10}" + "".join(
            f"{block['by_question'][q]['brier']:>14.5f}" for q in questions))

    n = len(seasons)
    print(f"\n  {n} season-years is the sample size, not {n * 20}: one club per season wins the "
          "title, so the rows inside a season are anything but independent.")


def cmd_predict(args: argparse.Namespace) -> int:
    """Forecast fixtures typed at the command line, or read from a file."""
    import numpy as np
    import pandas as pd

    from plmodel.data.teams import AmbiguousTeamError, load_aliases, resolve_team
    from plmodel.model.scoreline import (
        both_teams_to_score,
        clamp_rho_for_rates,
        collapse_three_class,
        scoreline_matrix,
        top_scorelines,
        totals_probability,
    )

    cfg = load_config(args.config)
    _, matches = _load_corpus(cfg)
    ref = pd.Timestamp(args.asof) if args.asof else matches["date"].max() + pd.Timedelta(days=1)
    train = matches[matches["date"] < ref]
    if train.empty:
        print(f"no matches before {ref.date()}", file=sys.stderr)
        return 2

    try:
        pairs = _parse_fixtures(args)
    except ValueError as exc:
        print(f"could not read the fixtures: {exc}", file=sys.stderr)
        return 2
    if not pairs:
        print("no fixtures given; use --fixtures, --file, or --home/--away", file=sys.stderr)
        return 2

    fit = _season_fit(cfg, matches, ref)
    known = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    aliases = load_aliases(cfg.static_dir)
    resolved: list[tuple[str, str]] = []
    for home, away in pairs:
        try:
            resolved.append((resolve_team(home, known, aliases=aliases),
                             resolve_team(away, known, aliases=aliases)))
        except AmbiguousTeamError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2
    for (typed_h, typed_a), (h, a) in zip(pairs, resolved):
        if typed_h != h or typed_a != a:
            print(f"read '{typed_h} v {typed_a}' as '{h} v {a}'")

    frame = pd.DataFrame([{"date": ref, "home_team": h, "away_team": a} for h, a in resolved])
    lam, mu = fit.match_rates(frame)
    rho, _ = clamp_rho_for_rates(lam, mu, fit.rho, margin=_PREDICT_RHO_MARGIN)
    grid = scoreline_matrix(lam, mu, rho, cfg.model.max_goals)
    outcome = collapse_three_class(grid)
    over, under = totals_probability(grid, args.line)
    btts = both_teams_to_score(grid)
    scores = top_scorelines(grid, args.top)
    quality = _team_history(train, fit, ref, cfg)

    print(f"as of     : {ref.date()}   ({len(train):,} matches behind the barrier)")
    print(f"half-life : {fit.half_life_days:.0f} days   home advantage {fit.home_advantage:+.4f}")
    for i, (h, a) in enumerate(resolved):
        print(f"\n{h} v {a}")
        print(f"  expected goals   {lam[i]:.2f} - {mu[i]:.2f}")
        print(f"  home / draw / away   {outcome[i, 0]:.3f} / {outcome[i, 1]:.3f} / "
              f"{outcome[i, 2]:.3f}")
        print(f"  over {args.line} / under {args.line}   {over[i]:.3f} / {under[i]:.3f}"
              f"        both to score  {btts[i]:.3f} / {1.0 - btts[i]:.3f}")
        print("  likeliest scores " + ",  ".join(
            f"{x}-{y} {p:.3f}" for x, y, p in scores[i]))
        for club in (h, a):
            note = quality[club]
            if note:
                print(f"  ! {club}: {note}")

    payload = {
        "as_of": str(ref.date()),
        "n_train": int(len(train)),
        "line": args.line,
        "fixtures": [
            {
                "home_team": h, "away_team": a,
                "expected_goals": [float(lam[i]), float(mu[i])],
                "home": float(outcome[i, 0]), "draw": float(outcome[i, 1]),
                "away": float(outcome[i, 2]),
                "over": float(over[i]), "under": float(under[i]),
                "both_teams_to_score": float(btts[i]),
                "top_scorelines": [
                    {"home_goals": x, "away_goals": y, "p": p} for x, y, p in scores[i]
                ],
                "warnings": [quality[c] for c in (h, a) if quality[c]],
            }
            for i, (h, a) in enumerate(resolved)
        ],
        "not_modelled": (
            "Half-time/full-time, first goalscorer, assists, cards and corners are NOT produced "
            "here. Every market above is a reading of one full-time scoreline distribution; "
            "anything needing a half-time state or a player is a model this project has not built."
        ),
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / (args.out or "predict.json")
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nHalf-time/full-time is not modelled — see `pl predict --help`.")
    print(f"report    : {out_path}")
    return 0


def _parse_fixtures(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Fixtures from --home/--away, --fixtures, or a CSV file. Any mix is accepted."""
    import pandas as pd

    pairs: list[tuple[str, str]] = []
    if args.home or args.away:
        if not (args.home and args.away):
            raise ValueError("--home and --away must be given together")
        pairs.append((args.home.strip(), args.away.strip()))
    if args.fixtures:
        for chunk in args.fixtures.split(","):
            text = chunk.strip()
            if not text:
                continue
            for token in _FIXTURE_SEPARATORS:
                if token in f" {text} ":
                    home, _, away = text.partition(token.strip() if token != " - " else " - ")
                    if home.strip() and away.strip():
                        pairs.append((home.strip(), away.strip()))
                        break
            else:
                raise ValueError(f"{text!r} is not 'Home v Away'")
    if args.file:
        frame = pd.read_csv(args.file)
        missing = {"home_team", "away_team"} - set(frame.columns)
        if missing:
            raise ValueError(f"{args.file} is missing columns {sorted(missing)}")
        pairs.extend((str(r.home_team), str(r.away_team)) for r in frame.itertuples(index=False))
    return pairs


def _team_history(train, fit, ref, cfg) -> dict[str, str]:
    """A warning per club whose parameters rest on little or old evidence, keyed by club.

    The season validation put promoted clubs' relegation Brier at three to four times everyone
    else's, and the worst case is not the club the fit has never seen -- that one is visibly pinned
    at the league average -- but the club whose last top-flight match was years ago and which
    therefore carries a confident-looking number fitted to almost nothing.
    """
    import numpy as np

    from plmodel.model.dixon_coles import decay_weights

    weights = decay_weights(train["date"], ref, cfg.model.decay_half_life_days)
    effective: dict[str, float] = {}
    last_seen: dict[str, object] = {}
    for side in ("home_team", "away_team"):
        for club, w, when in zip(train[side], weights, train["date"]):
            effective[club] = effective.get(club, 0.0) + float(w)
            if club not in last_seen or when > last_seen[club]:
                last_seen[club] = when
    median = float(np.median(list(effective.values()))) if effective else 0.0
    floor = median * cfg.model.min_effective_share

    notes: dict[str, str] = {}
    for club in set(fit.teams) | set(effective) | set(fit.cold_start_teams):
        share = effective.get(club, 0.0)
        seen = last_seen.get(club)
        if share == 0.0:
            notes[club] = ("never seen in this division — pinned at the league average, which "
                           "rates it as a typical club of this tier")
        elif share < floor:
            notes[club] = (f"cold-started: {share:.1f} effective matches against a median of "
                           f"{median:.0f}; last seen {seen.date()}")
        elif share < floor * _STALE_MULTIPLE:
            notes[club] = (f"thin evidence: {share:.1f} effective matches, last seen "
                           f"{seen.date()} — the parameters look confident and are not")
        else:
            notes[club] = ""
    return notes


def _planned(name: str) -> int:
    print(f"`pl {name}` is not built yet: {_PLANNED[name]}.", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pl", description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
    sub = ap.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("config", help="show the loaded config and the acceptance rule")
    pc.set_defaults(func=cmd_config)

    pi = sub.add_parser("ingest", help="fetch + cache + validate the corpus; emit coverage report")
    pi.add_argument("--refresh", action="store_true", help="re-download cached season files")
    pi.add_argument("--divisions", default=None, help="comma-separated subset, e.g. E0,E1")
    pi.add_argument("--with-xg", action="store_true",
                    help="also fetch and attach expected goals (needs a Kaggle token)")
    pi.set_defaults(func=cmd_ingest)

    pcmp = sub.add_parser("compare", help="paired A/B of arms vs baseline, with the market gate")
    pcmp.add_argument("--arms", required=True,
                      help="comma-separated arm names; the first is the baseline")
    pcmp.add_argument("--sensitivity", action="store_true",
                      help="run on the earlier decade instead of the test span")
    pcmp.add_argument("--first-season", default=None, help="override the pool's first season")
    pcmp.add_argument("--last-season", default=None, help="override the pool's last season")
    pcmp.add_argument("--out", default=None, help="report filename within the output directory")
    pcmp.set_defaults(func=cmd_compare)

    pf = sub.add_parser("fit", help="fit the production model and dump its parameters")
    pf.add_argument("--asof", default=None, help="fit as of this date (default: after the last match)")
    pf.add_argument("--half-life", type=float, default=None, help="override the configured half-life")
    pf.set_defaults(func=cmd_fit)

    pb = sub.add_parser("backtest", help="walk-forward the production model")
    pb.add_argument("--sensitivity", action="store_true", help="run on the earlier decade")
    pb.add_argument("--out", default=None, help="report filename within the output directory")
    pb.set_defaults(func=cmd_backtest)

    pr = sub.add_parser("reproduce", help="re-run an external claim on our data")
    pr.add_argument("--paper", required=True, help=f"paper id; available: {list(_REPRODUCIBLE)}")
    pr.add_argument("--market", default=None, help="override the market family")
    pr.set_defaults(func=cmd_reproduce)

    pa = sub.add_parser("audit", help="calibration slices for one arm")
    pa.add_argument("--arm", default="uniform", help="arm to audit")
    pa.set_defaults(func=cmd_audit)

    pl_ = sub.add_parser("live", help="freeze a matchday's forecasts, or score frozen ones")
    # The production model leads the list. The model-free arms cost nothing and are kept as the
    # floor every frozen block is read against, but the reason this ledger exists at all is that a
    # FITTED model's belief before kickoff cannot be reconstructed afterwards.
    pl_.add_argument("--arms", default="dixon-coles,home-rate,uniform",
                     help="comma-separated arms to freeze")
    pl_.add_argument("--score", action="store_true",
                     help="score previously frozen forecasts instead of freezing")
    pl_.add_argument("--dry-run", action="store_true",
                     help="show what would be frozen, and write nothing")
    pl_.add_argument("--no-feed", action="store_true",
                     help="use only the cached season file, not the rolling fixtures feed")
    pl_.set_defaults(func=cmd_live)

    ps = sub.add_parser("simulate", help="Monte Carlo a season's remaining fixtures")
    ps.add_argument("--season", default=None,
                    help="season label (default: the latest in the corpus)")
    ps.add_argument("--week", type=int, default=0,
                    help="forecast from after this many matchweeks (default: preseason)")
    ps.add_argument("--asof", default=None, help="forecast from this date instead of a matchweek")
    ps.add_argument("--uncertainty", choices=("point", "drift"), default=None,
                    help="override how parameter uncertainty is propagated (default: config)")
    ps.add_argument("--replicates", type=int, default=None, help="override the replicate count")
    ps.add_argument("--validate", action="store_true",
                    help="score the simulator against every completed season in a span")
    ps.add_argument("--sensitivity", action="store_true", help="validate on the earlier decade")
    ps.add_argument("--tuning", action="store_true", help="validate on the tuning span")
    ps.add_argument("--progress", action="store_true", help="print each barrier as it runs")
    ps.add_argument("--out", default=None, help="report filename within the output directory")
    ps.set_defaults(func=cmd_simulate)

    pp = sub.add_parser(
        "predict",
        help="forecast fixtures given on the command line",
        description=(
            "Full-time markets for any fixture: home/draw/away, over/under, both teams to "
            "score, and the likeliest exact scores. All of it is one scoreline distribution "
            "read different ways. Half-time/full-time, goalscorers, assists, cards and corners "
            "are NOT modelled and this command will not invent them."
        ),
    )
    pp.add_argument("--fixtures", default=None,
                    help="comma-separated, e.g. \"Arsenal v Coventry, Man Utd v Hull\"")
    pp.add_argument("--file", default=None, help="CSV with home_team and away_team columns")
    pp.add_argument("--home", default=None, help="a single fixture's home side")
    pp.add_argument("--away", default=None, help="a single fixture's away side")
    pp.add_argument("--asof", default=None,
                    help="forecast as of this date (default: after the last match in the corpus)")
    pp.add_argument("--line", type=float, default=2.5, help="total-goals line (default 2.5)")
    pp.add_argument("--top", type=int, default=6, help="how many exact scores to list")
    pp.add_argument("--out", default=None, help="report filename within the output directory")
    pp.set_defaults(func=cmd_predict)

    for name, purpose in _PLANNED.items():
        p = sub.add_parser(name, help=f"[not built yet] {purpose}")
        p.set_defaults(func=lambda _a, _n=name: _planned(_n))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
