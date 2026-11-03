"""Assemble demo/data/site-data.json from the repo's saved outputs (gitignored output/ folder).

Re-shapes saved numbers only. The one derived quantity is the actual league table, computed
from the cached football-data.co.uk results (data/cache/matches.parquet).

    python demo/data/build_site_data.py --output-dir output --matches data/cache/matches.parquet
"""
import argparse, json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True); ap.add_argument("--matches", required=True)
a = ap.parse_args(); O = Path(a.output_dir); D = Path(__file__).parent
J = lambda n: json.load(open(O / n, encoding="utf8"))

def table(df):
    r = {}
    for x in df.itertuples():
        if not x.played: continue
        for t, gf, ga in ((x.home_team, x.home_goals, x.away_goals), (x.away_team, x.away_goals, x.home_goals)):
            d = r.setdefault(t, [0, 0, 0, 0]); d[0] += 1; d[1] += 3 if gf > ga else (1 if gf == ga else 0); d[2] += gf; d[3] += ga
    t = pd.DataFrame([(k, *v) for k, v in r.items()], columns=["team", "p", "pts", "gf", "ga"]); t["gd"] = t.gf - t.ga
    t = t.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True); t["pos"] = t.index + 1
    return t.set_index("team")

m = pd.read_parquet(a.matches); e = m[m.division == "E0"]
def season(s, simfile, early=False):
    sim = J(simfile); act = table(e[e.season == s]); rows = []
    for p in sim["probabilities"]:
        t = p["team"]; r = {"team": t, "title": round(p["title"], 5), "top4": round(p["top_four"], 5), "rel": round(p["relegation"], 5),
                            "mpts": round(p["mean_points"], 2), "mpos": round(p["mean_position"], 2),
                            "apts": int(act.loc[t, "pts"]), "apos": int(act.loc[t, "pos"]), "agd": int(act.loc[t, "gd"]), "played": int(act.loc[t, "p"])}
        rows.append(r)
    return {"season": s, "barrier": sim["barrier"], "n_replicates": sim["n_replicates"], "n_remaining": sim["n_remaining"], "teams": rows,
            "uncertainty": sim["uncertainty"], "seed": sim["diagnostics"]["seed"]}

# arms tested: test decade (2016-17 to 2025-26, 2,660 odds-covered matches) paired with sensitivity decade (2006-07 to 2015-16)
PAIRS = [
         ("arm2_home_advantage", None, None, None), ("arm3_dynamics", "arm3_sensitivity", "dc-gas", "Score-driven (GAS) dynamics"),
         ("arm4_pooled", None, None, None), ("arm8_tiers", "arm8_sensitivity", None, None), ("decay_test", "decay_sens", None, None),
         ("gas2555_test", "gas2555_sens", "dc-gas", "GAS dynamics, variant 2"), ("gbm_test", "gbm_sensitivity", "gbm", "LightGBM"),
         ("gate_sot_form_test", "gate_sot_form_sens", "dc+sot-form", "Dixon-Coles + shots-on-target form")]
LABEL = {"elo-dc": "Elo rating instead of per-team strengths", "dc-gas": "Score-driven (GAS) strengths", "ha-trend": "Home advantage: time trend",
         "ha-empty": "Home advantage: empty-stadium term", "ha-both": "Home advantage: trend + empty stadiums", "dc+xg": "Dixon-Coles + xG channel",
         "dc+sot": "Dixon-Coles + shots-on-target channel", "dc-tiers": "Joint fit with lower divisions", "dc-tiers3": "Joint fit, three tiers",
         "dc-decay": "Alternative time-decay", "dc-gas-decay": "GAS dynamics with decay", "gbm": "LightGBM goals model",
         "dc+sot-form": "Trailing shots-on-target form term", "dixon-coles-sot": "Dixon-Coles + shots-on-target (full span)", "dixon-coles-xg": "Dixon-Coles + xG (full span)"}
RUN = {"arm3_dynamics/dc-gas": "Score-driven (GAS) strengths", "decay_test/dc-gas": "GAS strengths, decay-test run", "gas2555_test/dc-gas": "GAS strengths, 2,555-day variant",
       "decay_test/dc-gas-decay": "GAS strengths with decay", "arm4_pooled/dc+xg": "xG as a pooled channel", "arm4_pooled/dc+sot": "Shots on target as a pooled channel"}
arms = []; seen = set()
for test, sens, _, _ in PAIRS:
    t = J(test + ".json"); s = J(sens + ".json") if sens else None
    for k, v in t["arms"].items():
        if k == "dixon-coles" or (test, k) in seen: continue
        seen.add((test, k)); ver = t.get("verdicts", {}).get(k)
        vb, vm = v["vs_baseline"], v["vs_market"]


        sp = None
        if s and k in s["arms"] and s["arms"][k]["vs_baseline"]:
            sp = round(s["arms"][k]["vs_baseline"]["p_a_better"], 3)
        arms.append({"id": k, "label": RUN.get(test + "/" + k, LABEL.get(k, k)), "src": test, "rps": round(v["pooled"]["rps"], 5), "n": v["pooled"]["n"],
                     "d_base": round(vb["delta_rps"], 5) if vb else None, "d_lo": round(vb["ci_low"], 5) if vb else None, "d_hi": round(vb["ci_high"], 5) if vb else None,
                     "d_mkt": round(vm["delta_rps"], 5) if vm else None, "m_lo": round(vm["ci_low"], 5) if vm else None, "m_hi": round(vm["ci_high"], 5) if vm else None,
                     "g": None if not ver else [ver.get("gate1_vs_baseline"), ver.get("gate2_vs_market"), ver.get("gate3_family_wise"), ver.get("gate4_sensitivity")],
                     "accepted": None if not ver else ver["accepted"], "rule": None if not ver else (4 if ver.get("gate3_family_wise") is not None else 2),
                     "d_dc_point": round(v["pooled"]["rps"] - t["arms"]["dixon-coles"]["pooled"]["rps"], 5), "sens_p": sp})
g = J("gate_sot_form_test.json")
dc = g["arms"]["dixon-coles"]; sf = g["arms"]["dc+sot-form"]
seasons = []
dcs = {x["group"]: x for x in dc["slices"] if x["slice"] == "season"}
for x in sf["slices"]:
    if x["slice"] == "season":
        seasons.append({"season": x["group"], "n": x["n"], "dc": round(dcs[x["group"]]["rps"], 5), "sot": round(x["rps"], 5), "uniform": round(x["rps_uniform"], 5)})
def seasonval(n):
    s = J(n)["summary"]["specs"]; out = {}
    for k in ("point", "drift"):
        v = s[k]; out[k] = {"brier": v["brier"], "log_loss": v["log_loss"], "pit_ks": v["points_pit"]["ks"], "by_q": {q: v["by_question"][q]["brier"] for q in v["by_question"]},
                           "promoted_brier": v["by_promotion"]["promoted"]["brier"], "established_brier": v["by_promotion"]["established"]["brier"]}
    return out
pr = J("predict.json")
out = {"s2526": season("2025-26", "simulate_2025-26.json"), "s2627": season("2026-27", "simulate_2026-27.json"),
       "arms": arms, "market": {"benchmark": g["market"]["benchmark"], "devig": g["market"]["devig"], "n_cov": g["market"]["n_covered"], "n_total": g["market"]["n_total"], "rps_market": round(g["market"]["rps_market"], 5)},
       "prod": {"dc": {"rps": round(dc["pooled"]["rps"], 5), "skill": round(dc["pooled"]["skill"], 4), "vm": dc["vs_market"]},
                "sot": {"rps": round(sf["pooled"]["rps"], 5), "skill": round(sf["pooled"]["skill"], 4), "vm": sf["vs_market"], "vb": sf["vs_baseline"]},
                "uniform": round(dc["pooled"]["rps_uniform"], 5), "n": dc["pooled"]["n"], "verdict": g["verdicts"]["dc+sot-form"]},
       "seasons": seasons, "seasonval_test": seasonval("season_test.json"), "seasonval_old": seasonval("simulate_validation.json"),
       "predict": pr}
(D / "site-data.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False, default=float), encoding="utf8")
print("ok", len(json.dumps(out, default=float)) // 1024, "KB; arms", len(arms))
for r in arms: print(r["id"], r["rps"], r["d_base"], r["d_mkt"], r["accepted"], r["sens_p"])
