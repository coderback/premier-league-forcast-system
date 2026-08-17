"""The `pl` command-line surface.

Commands are registered as each phase lands. A command declared here but not yet built exits with
a message naming the phase that builds it — an honest stub is better than a missing command,
because `pl --help` then documents the intended surface without pretending it works.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from plmodel import __version__
from plmodel.config import ConfigError, load_config

# The intended command surface. Entries are removed from here as they are built, so this mapping
# is both the roadmap and the single place an unbuilt command is documented.
_PLANNED: dict[str, str] = {
    "simulate": "season Monte Carlo -> title / top-4 / relegation / points distribution",
    "reproduce": "re-run an external claim on our data",
}


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
    report = build_report(
        corpus, metas, skipped=corpus.attrs.get("skipped_seasons"), cfg=cfg
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
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


def cmd_live(args: argparse.Namespace) -> int:
    """Freeze a matchday's forecasts before kickoff, or score previously frozen ones."""
    from plmodel.eval.live import freeze_matchday, score_ledger

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

    fixtures = corpus[
        (corpus["division"] == cfg.backtest.prediction_division) & ~corpus["played"]
    ]
    frozen = freeze_matchday(
        fixtures, played, cfg, ledger_dir,
        arm_names=[a.strip() for a in args.arms.split(",") if a.strip()],
    )
    if frozen is None:
        print("no unplayed fixtures in the corpus — nothing to freeze.")
        print("The source publishes a season's fixtures shortly before it starts.")
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
    pi.set_defaults(func=cmd_ingest)

    pcmp = sub.add_parser("compare", help="paired A/B of arms vs baseline, with the market gate")
    pcmp.add_argument("--arms", required=True,
                      help="comma-separated arm names; the first is the baseline")
    pcmp.add_argument("--sensitivity", action="store_true",
                      help="run on the earlier decade instead of the test span")
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

    pa = sub.add_parser("audit", help="calibration slices for one arm")
    pa.add_argument("--arm", default="uniform", help="arm to audit")
    pa.set_defaults(func=cmd_audit)

    pl_ = sub.add_parser("live", help="freeze a matchday's forecasts, or score frozen ones")
    pl_.add_argument("--arms", default="uniform,home-always,home-rate",
                     help="comma-separated arms to freeze")
    pl_.add_argument("--score", action="store_true",
                     help="score previously frozen forecasts instead of freezing")
    pl_.set_defaults(func=cmd_live)

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
