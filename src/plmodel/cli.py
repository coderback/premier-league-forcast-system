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
from pathlib import Path

from plmodel import __version__
from plmodel.config import ConfigError, load_config

# The intended command surface. Entries are removed from here as they are built, so this mapping
# is both the roadmap and the single place an unbuilt command is documented. Empty means every
# command the project set out to build exists.
_PLANNED: dict[str, str] = {}

# The registered arm the production model is scored as. `pl backtest` reads its block out of the
# report by name, and a lookup that missed used to return an empty dict and skip the entire
# readout without a word — a trap laid directly in the path of ever renaming or replacing it.
PRODUCTION_ARM = "dc+sot-form"

# Accepted spellings of "against" in a fixture typed at the command line.
_FIXTURE_SEPARATORS = (" v ", " vs ", " V ", " - ")

# Back-off from the tau validity boundary, matching the model's own prediction-time clamp.
_PREDICT_RHO_MARGIN = 0.01

# A club above the cold-start floor but below this multiple of it has parameters that look
# confident and are not. Hull's 2017 rating is the case this exists to surface.
_STALE_MULTIPLE = 3.0

# `pl predict`'s derived markets: winning margins are shown up to "by 3 or more", and the
# total-goals table up to "6 or more". Presentation only; every figure is read from the same grid.
_MARGIN_CAP = 3
_TOTAL_GOALS_CAP = 6

# `pl live` flags any frozen outcome below this probability. A warning only; the forecast is
# frozen as fitted.
_LOW_PROBABILITY = 0.02

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
    # A declared family can only be larger than the arms actually run. Refused rather than
    # clamped: asking for a smaller one means somebody has misunderstood which tests are being
    # corrected across, and quietly widening the bar would be the one direction this flag must
    # never move.
    candidates = len(arm_names) - 1
    if args.family_size is not None and args.family_size < candidates:
        print(f"--family-size {args.family_size} is smaller than the {candidates} candidate "
              f"arm(s) in this run; a declared family cannot be smaller than the tests it "
              f"corrects across", file=sys.stderr)
        return 2

    splits = _build_splits(cfg, matches, span=span)
    # Only arms that read other divisions get them, and they get them through run_arm, which
    # truncates every extra row at each barrier exactly as it truncates the prediction division.
    from plmodel.eval.compare import ARMS_NEEDING_TIERS

    tiers = (corpus[corpus["played"]] if any(a in ARMS_NEEDING_TIERS for a in arm_names)
             else None)
    report = run_compare(
        matches, splits, cfg, arm_names,
        history=corpus[corpus["division"] == cfg.backtest.prediction_division],
        n_bins=cfg.audit.calibration_bins,
        big_six=cfg.audit.big_six,
        family_size=args.family_size,
        tiers=tiers,
    )

    sensitivity = _sensitivity_deltas(args.against_span)
    payload = report_json(report, sensitivity=sensitivity)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / (args.out or "compare.json")
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    _print_compare(payload, report)
    if sensitivity is None:
        print("\n  gate 4 was not evaluated: no sensitivity-span report was supplied, so nothing "
              "in this run\n  can be accepted. Run with --sensitivity, then re-run the test span "
              "with\n  --against-span pointing at that report.")
    print(f"\nreport    : {out_path}")
    return 0


def _sensitivity_deltas(path: str | None) -> dict[str, dict[str, float]] | None:
    """Per-arm deltas from a previously written report, for the acceptance rule's fourth gate.

    Read from a file rather than recomputed, because the two spans are two expensive walks and the
    rule only needs one number from the second. The file is a report this project wrote, so a
    missing key is a bug rather than a user error and raises.
    """
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: block["vs_baseline"]
        for name, block in payload["arms"].items()
        if block.get("vs_baseline")
    }


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

    # The other markets, reported for every arm and never gated on (except the closing one for an
    # arm that reads its own match's prices). Keeps the old close-based headline comparable.
    diagnostics = {k: v for k, v in (market.get("diagnostics") or {}).items()
                   if v.get("n_covered")}
    if diagnostics:
        print("\nother markets (reported; gap = arm RPS - market RPS on that market's rows):")
        for family, info in diagnostics.items():
            gaps = "   ".join(
                f"{name} {block['vs_market_diagnostics'][family]['delta_rps']:+.4f}"
                for name, block in payload["arms"].items()
                if family in (block.get("vs_market_diagnostics") or {})
            )
            settled = f"({info['settlement']})"
            print(f"  {family:17} n={info['n_covered']:>5,} {settled:11}   {gaps}")
    tracks = {name: block.get("track") for name, block in payload["arms"].items()}
    informed = [name for name, track in tracks.items() if track != "independent"]
    if informed:
        print(f"\nMARKET-INFORMED (reads bookmaker prices; never production): {', '.join(informed)}")

    if payload["verdicts"]:
        print("\nacceptance rule:")
        for line in payload["acceptance_rule"].splitlines():
            print(f"  {line}")
        print()
        label = {True: "pass", False: "FAIL", None: "n/a"}
        for name, verdict in payload["verdicts"].items():
            mark = "ACCEPT" if verdict["accepted"] else "reject"
            print(f"  {mark:7} {name:16} "
                  f"gate1 {label[verdict['gate1_vs_baseline']]} ({verdict['gate1_reason']})")
            if verdict.get("gate1_by_season"):
                print(f"  {'':7} {'':16}   {verdict['gate1_by_season']}")
            print(f"  {'':7} {'':16} gate2 {label[verdict['gate2_vs_market']]}"
                  f"   gate3 {label[verdict['gate3_family_wise']]}"
                  f"   gate4 {label[verdict['gate4_sensitivity']]}")
            for key in ("gate2_reason", "gate3_reason", "gate4_reason"):
                if verdict.get(key):
                    print(f"  {'':7} {'':16}   {verdict[key]}")


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
    for line in _calibration_lines(arm.calibration):
        print(line)

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


def _calibration_lines(calibration: dict) -> list[str]:
    """The Murphy readout, with the discretisation error beside the score it distorts.

    ``resid`` is ``brier - brier_from_decomposition``. :mod:`plmodel.eval.calibration` has always
    computed both and says in its own docstring that reporting the pair "makes the discretisation
    visible instead of hiding it" — and then this readout printed only one of them, which is where
    the visibility was lost. NOTES.md 2026-09-22 records what that cost: a +0.00054 draw-resolution
    gap that read as a finding, survived into an entry as a lead, and turned out to be the two arms'
    binning residuals differing by 0.00037.

    The residual is a property of the GRID, not of the forecaster. It earns a column because a
    resolution difference is only readable when it is large against the residuals underneath it.
    """
    header = (f"  {'outcome':9}{'base rate':>11}{'reliability':>13}{'resolution':>12}"
              f"{'brier':>9}{'from bins':>11}{'resid':>10}")
    lines = [header]
    for outcome, block in calibration.items():
        d = block["decomposition"]
        implied = d["brier_from_decomposition"]
        lines.append(
            f"  {outcome:9}{d['base_rate']:>11.4f}{d['reliability']:>13.5f}"
            f"{d['resolution']:>12.5f}{d['brier']:>9.4f}{implied:>11.4f}"
            f"{d['brier'] - implied:>+10.5f}"
        )
    lines.append("")
    lines.append("  Draw resolution is expected to stay flat: nothing in the literature moves it.")
    lines.append("  Read `resid` before comparing resolution across arms: where two arms' residuals")
    lines.append("  differ by as much as their resolutions do, the comparison is measuring the bin")
    lines.append("  grid rather than the forecasts. See NOTES.md 2026-09-22.")
    return lines


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit the production model on everything available, and dump what it believes."""
    import pandas as pd

    from plmodel.model.dixon_coles import fit_summary

    cfg = load_config(args.config)
    _, matches = _load_corpus(cfg)
    ref = pd.Timestamp(args.asof) if args.asof else matches["date"].max() + pd.Timedelta(days=1)
    if matches[matches["date"] < ref].empty:
        print(f"no matches before {ref.date()}", file=sys.stderr)
        return 2

    fit = _season_fit(cfg, matches, ref, half_life_days=args.half_life)
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
    # Rates from the fit's own predictor rather than re-derived here. This used to be
    # `exp(intercept + home_advantage + attack)` written out by hand — a second implementation of
    # the linear predictor living in the CLI, and one that silently drops the structural
    # home-advantage terms, the covariate context and the log-rate clip. An unrecognised opponent
    # takes the league average by construction, so this reads as "against an average side".
    against_average = pd.DataFrame({
        "date": ref, "home_team": list(table["team"]), "away_team": "<league average>",
    })
    # The opponent has no form, so the shots term is undefined and switches off for every row:
    # this table is the strengths alone. History is passed because a covariate fit requires it.
    lam, mu = fit.match_rates(against_average, matches[matches["date"] < ref])
    print(f"\n{'team':18}{'attack':>9}{'defence':>9}{'exp goals for':>15}{'against':>9}")
    for row, home_rate, away_rate in zip(table.itertuples(index=False), lam, mu):
        print(f"{row.team:18}{row.attack:>+9.3f}{row.defence:>+9.3f}"
              f"{home_rate:>15.2f}{away_rate:>9.2f}")

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
        matches, splits, cfg, ["home-rate", PRODUCTION_ARM],
        history=corpus[corpus["division"] == cfg.backtest.prediction_division],
        n_bins=cfg.audit.calibration_bins, big_six=cfg.audit.big_six,
    )
    payload = report_json(report)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / (args.out or "backtest.json")
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    _print_compare(payload, report)
    if PRODUCTION_ARM not in payload["arms"]:
        print(f"the backtest report carries no {PRODUCTION_ARM!r} block; the production arm did "
              "not run, so there is nothing to report against.", file=sys.stderr)
        return 2
    dc = payload["arms"][PRODUCTION_ARM]
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
    """Freeze the coming round's forecasts before kickoff, or score previously frozen ones."""
    import pandas as pd

    from plmodel.data.football_data import upcoming_fixtures
    from plmodel.data.teams import TeamNameError
    from plmodel.eval.live import (
        ledger_resolution, round_dates, score_ledger, split_past_fixtures,
    )

    cfg = load_config(args.config)
    corpus, played = _load_corpus(cfg)
    ledger_dir = cfg.output_dir / "live"

    if args.score:
        scored = score_ledger(ledger_dir, played)
        if scored.empty:
            print("nothing to score yet: no frozen forecast has a result.")
        else:
            print(scored.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        fate = ledger_resolution(ledger_dir, played, _today())
        print(f"\nfrozen fixtures: {fate['n_frozen']}   scored: {fate['n_scored']}   "
              f"still to come: {fate['n_to_come']}   past their date, unresolved: "
              f"{len(fate['unresolved'])}")
        for row in fate["unresolved"]:
            print(f"  {row['date']}  {row['home_team']} v {row['away_team']}: {row['status']}")
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

    fixtures, past = split_past_fixtures(fixtures, _today())
    if not past.empty:
        print(f"set aside {len(past)} fixture(s) dated before today and still unplayed in the "
              "corpus (postponed, or played and not yet ingested):", file=sys.stderr)
        for row in past.itertuples(index=False):
            print(f"  {pd.Timestamp(row.date).date()}  {row.home_team} v {row.away_team}",
                  file=sys.stderr)

    if fixtures.empty:
        print(f"no unplayed {cfg.backtest.prediction_division} fixtures known.")
        print("  the season file carries fixtures only once it has results in it, and the "
              "rolling feed lags by a day or two.")
        print("  re-run closer to kickoff, or check https://www.football-data.co.uk/fixtures.csv")
        return 0

    dates = round_dates(fixtures)
    check, stale = _corpus_check(cfg, played, offline=args.no_feed)
    if args.dry_run:
        print(f"coming round : {len(dates)} date(s), "
              f"{int(fixtures['date'].isin(dates).sum())} fixture(s)")
        print(f"source       : {'season file only' if args.no_feed else 'season file + feed'}")
        for date in dates:
            target = ledger_dir / f"{pd.Timestamp(date).date()}.json"
            day = fixtures[fixtures["date"] == date]
            note = "   ! already frozen, would NOT be overwritten" if target.exists() else ""
            print(f"\n  {pd.Timestamp(date).date()}  -> {target.name}{note}")
            for row in day.itertuples(index=False):
                print(f"    {row.home_team} v {row.away_team}")
        print("\nwould commit : yes, each file alone")
        print(f"corpus check : {check['summary']}")
        if stale:
            print(f"  ! would REFUSE to freeze: {stale}")
        print("nothing was written (--dry-run).")
        return 0

    if stale:
        print(f"refusing to freeze: {stale}", file=sys.stderr)
        print("  a forecast trained on a corpus behind the source is not the forecast the "
              "backtest validated.", file=sys.stderr)
        return 2

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for date in dates:
        _freeze_one_date(fixtures, played, cfg, ledger_dir, arms, check, date)
    return 0


def _freeze_one_date(fixtures, played, cfg, ledger_dir, arms, check, date) -> None:
    """Freeze and commit one date of the round, reporting as it goes. Never raises on a date that
    is already frozen: most runs of a daily cadence meet some of those."""
    import pandas as pd

    from plmodel.eval.live import commit_frozen, freeze_matchday

    try:
        path, block = freeze_matchday(
            fixtures, played, cfg, ledger_dir, arm_names=arms, staleness_check=check,
            barrier=date,
        )
    except FileExistsError:
        # The refusal to overwrite is the whole guarantee and stays, but it is a normal outcome
        # rather than a fault.
        print(f"already frozen for {pd.Timestamp(date).date()} - left as it is "
              "(a frozen forecast is never rewritten).")
        return
    for arm, error in (block.get("arm_errors") or {}).items():
        print(f"  ! {arm} (market-informed) not frozen: {error}", file=sys.stderr)
    pending = block["n_pending_before_barrier"]
    print(f"\nfroze {block['n_fixtures']} fixture(s) for {block['barrier']}"
          + (f"  ({pending} earlier fixture(s) of the round still unplayed)" if pending else ""))
    for row in block["fixtures"]:
        probs = "  ".join(
            f"{arm}: {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}" for arm, p in row["forecasts"].items()
        )
        print(f"  {row['home_team']} v {row['away_team']}   {probs}")
    for line in _low_probability_warnings(block["fixtures"]):
        print(line, file=sys.stderr)
    print(f"ledger    : {path}")

    # Committing is part of freezing rather than an afterthought: an untracked file in a
    # gitignored directory cannot support the claim that it existed before the result did.
    receipt = commit_frozen(path)
    if not receipt["committed"]:
        print(f"NOT COMMITTED: {receipt['reason']}", file=sys.stderr)
        print("  the forecast is on disk and still usable, but nothing yet records WHEN it was "
              "written.", file=sys.stderr)
        print("  commit it by hand before kickoff if the ledger is to mean what it claims.",
              file=sys.stderr)
        return

    print(f"committed : {receipt['commit']}  {receipt['message']}")
    unpushed = receipt["unpushed_commits"]
    if unpushed is None:
        print("  no upstream branch, so nothing off this machine has witnessed it; "
              "`git push` is what completes the guarantee.")
    elif unpushed:
        print(f"  {unpushed} commit(s) unpushed. A local commit date can be set to anything, so "
              "pushing is what records when a third party first saw this.")


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
        history=matches[matches["date"] < barrier],
        deductions=cfg.season.points_deductions.get(season, {}),
    )
    _print_simulation(forecast, cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / (args.out or f"simulate_{season}.json")
    out_path.write_text(json.dumps(forecast.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"\nreport    : {out_path}")
    return 0


def _season_fit(cfg, matches, barrier, *, half_life_days=None):
    """The production fit at a barrier. See plmodel.model.production for why it lives there."""
    from plmodel.model.production import production_fit

    try:
        return production_fit(cfg, matches, barrier, half_life_days=half_life_days)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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
        handicap_probability,
        margin_distribution,
        scoreline_matrix,
        scoring_markets,
        top_scorelines,
        total_goals_distribution,
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
    lam, mu = fit.match_rates(frame, train)
    rho, _ = clamp_rho_for_rates(lam, mu, fit.rho, margin=_PREDICT_RHO_MARGIN)
    grid = scoreline_matrix(lam, mu, rho, cfg.model.max_goals)
    outcome = collapse_three_class(grid)
    over, under = totals_probability(grid, args.line)
    btts = both_teams_to_score(grid)
    scores = top_scorelines(grid, args.top)
    scoring = scoring_markets(grid)
    margins = margin_distribution(grid, _MARGIN_CAP)
    totals = total_goals_distribution(grid, _TOTAL_GOALS_CAP)
    # The handicap goes to whichever side the model favours: "home -1.5" means little when the
    # away side is the stronger one. Read from the home side, a favourite away side is +line.
    home_favoured = lam >= mu
    handicaps = [
        handicap_probability(grid[i:i + 1], -args.handicap if home_favoured[i] else args.handicap)
        for i in range(len(resolved))
    ]
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
        home, draw, away = outcome[i]
        print(f"  double chance   home or draw {home + draw:.3f}   draw or away {draw + away:.3f}"
              f"   no draw {home + away:.3f}")
        print(f"  draw no bet     {h} {home / (home + away):.3f}   {a} {away / (home + away):.3f}")
        print(f"  to score        {h} {scoring['home_scores'][i]:.3f}   "
              f"{a} {scoring['away_scores'][i]:.3f}")
        print(f"  clean sheet     {h} {scoring['home_clean_sheet'][i]:.3f}   "
              f"{a} {scoring['away_clean_sheet'][i]:.3f}")
        print(f"  win to nil      {h} {scoring['home_win_to_nil'][i]:.3f}   "
              f"{a} {scoring['away_win_to_nil'][i]:.3f}")
        print("  winning margin  " + "  ".join(
            f"{_margin_label(m, h, a)} {margins[i, m + _MARGIN_CAP]:.3f}"
            for m in range(_MARGIN_CAP, -_MARGIN_CAP - 1, -1)))
        covers, push, against = (float(v[0]) for v in handicaps[i])
        fav, dog = (h, a) if home_favoured[i] else (a, h)
        fav_p, dog_p = (covers, against) if home_favoured[i] else (against, covers)
        print(f"  handicap        {fav} -{args.handicap:g} {fav_p:.3f}   {dog} +{args.handicap:g} "
              f"{dog_p:.3f}" + (f"   push {push:.3f}" if push > 0 else ""))
        print("  total goals     " + "  ".join(
            f"{t}{'+' if t == _TOTAL_GOALS_CAP else ''} {totals[i, t]:.3f}"
            for t in range(_TOTAL_GOALS_CAP + 1)))
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
                "double_chance": {
                    "home_or_draw": float(outcome[i, 0] + outcome[i, 1]),
                    "draw_or_away": float(outcome[i, 1] + outcome[i, 2]),
                    "no_draw": float(outcome[i, 0] + outcome[i, 2]),
                },
                "draw_no_bet": {
                    "home": float(outcome[i, 0] / (outcome[i, 0] + outcome[i, 2])),
                    "away": float(outcome[i, 2] / (outcome[i, 0] + outcome[i, 2])),
                },
                **{key: float(values[i]) for key, values in scoring.items()},
                "winning_margin": {
                    _margin_label(m, "home", "away"): float(margins[i, m + _MARGIN_CAP])
                    for m in range(-_MARGIN_CAP, _MARGIN_CAP + 1)
                },
                "handicap": {
                    "favourite": h if home_favoured[i] else a,
                    "line": args.handicap,
                    "favourite_covers": float(handicaps[i][0][0] if home_favoured[i]
                                              else handicaps[i][2][0]),
                    "push": float(handicaps[i][1][0]),
                    "underdog_covers": float(handicaps[i][2][0] if home_favoured[i]
                                             else handicaps[i][0][0]),
                },
                "total_goals": {
                    f"{t}{'+' if t == _TOTAL_GOALS_CAP else ''}": float(totals[i, t])
                    for t in range(_TOTAL_GOALS_CAP + 1)
                },
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


def _margin_label(margin: int, home: str, away: str) -> str:
    """'draw', '<home> by 2', '<away> by 3+' for a home margin capped at _MARGIN_CAP."""
    if margin == 0:
        return "draw"
    side = home if margin > 0 else away
    size = abs(margin)
    return f"{side} by {size}{'+' if size == _MARGIN_CAP else ''}"


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
                    # Partition on the token WHOLE. Stripping " v " to "v" splits inside any home
                    # side whose name contains one -- Everton, Aston Villa, Coventry, Wolves --
                    # and the guard above has already established the spaced token is present, so
                    # there was never anything for the strip to buy. The " - " special case that
                    # used to sit here was this bug half-noticed: one separator was exempted from
                    # the stripping instead of all four.
                    home, _, away = text.partition(token)
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


def _today():
    """Today's date, as a function so tests can fix it."""
    import pandas as pd

    return pd.Timestamp.today().normalize()


def _corpus_check(cfg, played, *, offline: bool) -> tuple[dict, str | None]:
    """Whether the cached corpus is as current as the source: (record for the ledger, refusal).

    `pl live` reads the cached corpus and fetches only the fixture feed, so a run without `pl
    ingest` first would freeze on a history missing the last round. One request to the season file
    `pl ingest` reads settles it. If the source cannot be reached, it refuses rather than assumes:
    `--no-feed` is the explicit way to freeze offline, and the file then says it was not checked.
    """
    import pandas as pd

    from plmodel.data.football_data import IngestError, latest_published_result
    from plmodel.eval.live import corpus_staleness

    corpus_last = str(pd.Timestamp(played["date"].max()).date()) if len(played) else None
    if offline:
        return {"checked": False, "corpus_last_result": corpus_last,
                "summary": f"NOT checked against the source (--no-feed); corpus ends {corpus_last}"}, None
    try:
        latest = latest_published_result(cfg, cfg.backtest.prediction_division, today=_today())
    except IngestError as exc:
        return {"checked": False, "corpus_last_result": corpus_last, "error": str(exc),
                "summary": f"could not reach the source: {exc}"}, (
            f"could not confirm the corpus is current ({exc}); run `pl ingest`, or pass "
            "--no-feed to freeze offline without the check")
    stale = corpus_staleness(played, latest)
    published = None if latest is None else str(pd.Timestamp(latest).date())
    return {"checked": True, "corpus_last_result": corpus_last, "source_last_result": published,
            "summary": (f"source has results through {published}, corpus through {corpus_last}"
                        + ("" if stale else " - current"))}, stale


def _low_probability_warnings(fixtures) -> list[str]:
    """One line per frozen outcome below :data:`_LOW_PROBABILITY`. Nothing is changed.

    An outcome the model calls nearly impossible is either a real mismatch or a fit that ran out of
    evidence. In 2026-27 one did (Coventry, 0.4% to win at Forest, and won). The warning puts the
    fixture in front of a person before kickoff, not in a post-mortem.
    """
    lines = []
    for row in fixtures:
        for arm, p in row["forecasts"].items():
            low = [f"{name} {q:.3f}" for name, q in zip(("home", "draw", "away"), p)
                   if q < _LOW_PROBABILITY]
            if low:
                lines.append(f"  ! {row['home_team']} v {row['away_team']} ({arm}): "
                             f"{', '.join(low)} - below {_LOW_PROBABILITY:.0%}; check the clubs' "
                             f"evidence before kickoff")
    return lines


def _team_history(train, fit, ref, cfg) -> dict[str, str]:
    """A warning per club whose parameters rest on little or old evidence, keyed by club.

    The season validation put promoted clubs' relegation Brier at three to four times everyone
    else's, and the worst case is not the club the fit has never seen -- that one is visibly pinned
    at the league average -- but the club whose last top-flight match was years ago and which
    therefore carries a confident-looking number fitted to almost nothing.
    """
    from plmodel.model.dixon_coles import cold_start_bar, decay_weights

    weights = decay_weights(train["date"], ref, cfg.model.decay_half_life_days)
    last_seen: dict[str, object] = {}
    for side in ("home_team", "away_team"):
        for club, when in zip(train[side], train["date"]):
            if club not in last_seen or when > last_seen[club]:
                last_seen[club] = when
    effective: dict[str, float] = {}
    median = floor = 0.0
    if len(train):
        # The same bar the fit uses, so a club this report calls cold is one the fit pinned.
        clubs, values, median, floor = cold_start_bar(
            train, weights, ref, cfg.model.min_effective_share
        )
        effective = dict(zip(clubs, (float(v) for v in values)))

    notes: dict[str, str] = {}
    for club in set(fit.teams) | set(effective) | set(fit.cold_start_teams):
        share = effective.get(club, 0.0)
        seen = last_seen.get(club)
        if club in set(getattr(fit, "boxed_teams", ())):
            notes[club] = (f"pinned at the league average: its fitted strength hit the edge of "
                           f"the allowed range on {share:.1f} effective matches, which is no "
                           f"estimate at all")
        elif share == 0.0:
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
    pcmp.add_argument("--against-span", default=None,
                      help="a report from the other span, supplying the rule's fourth gate")
    pcmp.add_argument("--out", default=None, help="report filename within the output directory")
    pcmp.add_argument("--family-size", type=int, default=None,
                      help="declare a pre-registered family LARGER than this run's arms, for an "
                           "arm re-scored serially across configurations; tightens gate 3 only")
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

    pl_ = sub.add_parser("live", help="freeze the coming round's forecasts, or score frozen ones")
    # The production model leads the list. The model-free arms cost nothing and are kept as the
    # floor every frozen block is read against, but the reason this ledger exists at all is that a
    # FITTED model's belief before kickoff cannot be reconstructed afterwards.
    # `dixon-coles` stays on the list after `dc+sot-form` replaced it in production, so the frozen
    # ledger carries the old and new production models head to head on matches neither has seen.
    # `odds-ratings` is the market-informed track's model (accepted 2026-11-18): frozen beside
    # production under its own name, never in its place.
    pl_.add_argument("--arms", default="dc+sot-form,odds-ratings,dixon-coles,home-rate,uniform",
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
    pp.add_argument("--handicap", type=float, default=1.5,
                    help="goal handicap given to the favourite (default 1.5)")
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
