"""The `pl` command-line surface.

Commands are registered as each phase lands. A command declared here but not yet built exits with
a message naming the phase that builds it — an honest stub is better than a missing command,
because `pl --help` then documents the intended surface without pretending it works.
"""
from __future__ import annotations

import argparse
import json
import sys

from plmodel import __version__
from plmodel.config import ConfigError, load_config

# The intended command surface. Entries are removed from here as they are built, so this mapping
# is both the roadmap and the single place an unbuilt command is documented.
_PLANNED: dict[str, str] = {
    "compare": "paired A/B of candidate arms vs baseline, with the market gate",
    "audit": "calibration slices: promoted-team, big-six-vs-rest, favourite, by season",
    "live": "freeze a matchweek's forecasts before kickoff; score them after",
    "fit": "fit the production model; dump params, ratings, fixture probabilities",
    "backtest": "rolling-origin walk-forward; metrics + calibration",
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
