"""Generate the paper's LaTeX tables, key numbers and figure files from a results directory.

  python make_paper_tables.py --results results --out paper
Writes paper/tables/*.tex, paper/tables/key_numbers.json and paper/figures/Fig1-5 (.tif + .png).
Run analyze.py on the results directory first.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DISTS = ["gaussian", "exponential", "lognormal", "pareto", "bimodal"]
DLABEL = {"gaussian": "Gaussian", "exponential": "Exponential", "lognormal": "Lognormal",
          "pareto": "Pareto", "bimodal": "Bimodal"}


def cell(r, digits=1):
    if r is None:
        return "--"
    ns = "" if r.p_holm < 0.05 else "$^{\\dagger}$"
    return (f"${r.rel_pct:+.{digits}f}$ {{\\scriptsize[${r.rel_ci_lo:+.{digits}f}$, "
            f"${r.rel_ci_hi:+.{digits}f}$]}}{ns}")


def short(r, digits=1):
    if r is None:
        return "--"
    ns = "" if r.p_holm < 0.05 else "$^{\\dagger}$"
    return f"${r.rel_pct:+.{digits}f}${ns}"


def pick(pt, **kw):
    s = pt
    for k, v in kw.items():
        s = s[np.isclose(s[k], v)] if isinstance(v, float) else s[s[k] == v]
    return s.iloc[0] if len(s) else None


def main(results, out):
    out.mkdir(parents=True, exist_ok=True)
    pt = pd.read_csv(results / "analysis/tables/paired_EED-RR.csv")
    pt15 = pd.read_csv(results / "analysis/tables/paired_EED-RR_sf1.5.csv")
    summ = pd.read_csv(results / "analysis/tables/summary_by_cell.csv")
    pt = pt[pt.switch_cost_frac == 0]
    trace_name = [d for d in pt[pt.family == "trace"].dist.unique()]
    trace_name = trace_name[0] if trace_name else None
    rows = [("synthetic", d, 0.9, 1000, DLABEL[d]) for d in DISTS]
    if trace_name:
        rows.append(("trace", trace_name, 0.9, 1000, "SDSC-SP2 trace"))
    nums = {}

    # ---- Table: EED-RR vs RR (same base quantum)
    mets = [("AWT", "Mean wait"), ("resp_mean", "Mean response"), ("W_short", "Wait, short third"),
            ("SD_mean", "Mean slowdown"), ("SD_max", "Max slowdown"), ("JFI_t_mean", "Jain $\\bar J(t)$")]
    L = ["\\begin{tabular}{l" + "r" * len(mets) + "}", "\\toprule",
         "Workload & " + " & ".join(m for _, m in mets) + " \\\\", "\\midrule"]
    for fam, d, load, n, lab in rows:
        cs = [short(pick(pt, family=fam, dist=d, load=load, n=n, control="RR", metric=m)) for m, _ in mets]
        L.append(f"{lab} & " + " & ".join(cs) + " \\\\")
        for m, _ in mets:
            r = pick(pt, family=fam, dist=d, load=load, n=n, control="RR", metric=m)
            if r is not None:
                nums[f"vsRR_{d}_{m}"] = [round(r.rel_pct, 1), round(r.rel_ci_lo, 1), round(r.rel_ci_hi, 1), float(r.p_holm)]
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "tab_vs_rr.tex").write_text("\n".join(L), encoding="utf-8")

    # ---- Table: targeting vs equal budget, ablations and definitional variants (AWT / SD_mean)
    ctr = [("RR-matched", "Equal-budget RR"), ("RandomBoost", "Random boost"),
           ("EED[JFI-only]", "JFI only"), ("EED[deficit-only]", "Deficit only"),
           ("EED[all,dispatch]", "Original state def.")]
    L = ["\\begin{tabular}{l" + "c" * len(ctr) + "}", "\\toprule",
         "Workload & " + " & ".join(c for _, c in ctr) + " \\\\", "\\midrule"]
    for fam, d, load, n, lab in rows:
        cs = []
        for c, _ in ctr:
            a = pick(pt, family=fam, dist=d, load=load, n=n, control=c, metric="AWT")
            s = pick(pt, family=fam, dist=d, load=load, n=n, control=c, metric="SD_mean")
            cs.append(f"{short(a)} / {short(s)}")
            if a is not None:
                nums[f"vs_{c}_{d}_AWT"] = [round(a.rel_pct, 1), round(a.rel_ci_lo, 1), round(a.rel_ci_hi, 1), float(a.p_holm)]
                nums[f"vs_{c}_{d}_SD"] = [round(s.rel_pct, 1), float(s.p_holm)]
        L.append(f"{lab} & " + " & ".join(cs) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "tab_controls.tex").write_text("\n".join(L), encoding="utf-8")

    # ---- Table: all policies, AWT and mean slowdown relative to RR (means over seeds)
    pols = [("EED-RR", "EED-RR ($sf=0.25$)"), ("EED-RR(sf=1.5)", "EED-RR ($sf=1.5$)"),
            ("RR(x1.5)", "RR, $1.5\\times$ quantum"), ("MARR", "MARR"),
            ("MARR(upper-median)", "MARR, upper median"), ("EImHLVQTRR", "EImHLVQTRR (per round)"),
            ("EImHLVQTRR(per-dispatch)", "EImHLVQTRR (per dispatch)"), ("ADRR*", "ADRR$^{a}$"),
            ("SARR", "SARR"), ("STS-RR", "STS-RR"), ("FCFS", "FCFS"), ("SJF", "SJF$^{b}$"),
            ("SRTF", "SRTF$^{b}$")]
    cols = [(f, d, l) for f, d, ld, n, l in rows]
    L = ["\\begin{tabular}{l" + "rr" * len(cols) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{l}}}" for _, _, l in cols) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
         "Policy & " + " & ".join("Wait & SD" for _ in cols) + " \\\\", "\\midrule"]
    for p, lab in pols:
        cs = []
        for f, d, _ in cols:
            s = summ[(summ.family == f) & (summ.dist == d) & np.isclose(summ.load, 0.9) &
                     (summ.n == 1000) & (summ.switch_cost_frac == 0) & (summ.policy == p)]
            if len(s):
                cs += [f"{s.AWT_vs_RR.iloc[0]:.2f}", f"{s.SD_mean_vs_RR.iloc[0]:.2f}"]
                nums[f"ratio_{p}_{d}"] = [round(s.AWT_vs_RR.iloc[0], 3), round(s.SD_mean_vs_RR.iloc[0], 3),
                                          round(s.resp_mean_vs_RR.iloc[0], 3)]
            else:
                cs += ["--", "--"]
        L.append(f"{lab} & " + " & ".join(cs) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "tab_baselines.tex").write_text("\n".join(L), encoding="utf-8")

    # ---- Table: batch / overload scenario (AWT relative to RR, response relative to RR)
    bp = [("RR(x1.25)", "RR, $1.25\\times$"), ("RR(x1.5)", "RR, $1.5\\times$"), ("RR(x2)", "RR, $2\\times$"),
          ("FCFS", "FCFS"), ("EED-RR", "EED-RR ($sf=0.25$)"), ("EED-RR(sf=1.5)", "EED-RR ($sf=1.5$)"),
          ("EED-RR-original(sf=1.5)", "EED-RR, original definition"), ("RR-matched(sf=1.5)",
                                                                       "Equal-budget RR ($sf=1.5$)")]
    L = ["\\begin{tabular}{l" + "r" * len(DISTS) + "}", "\\toprule",
         "Policy & " + " & ".join(DLABEL[d] for d in DISTS) + " \\\\", "\\midrule"]
    for p, lab in bp:
        cs = []
        for d in DISTS:
            s = summ[(summ.family == "batch") & (summ.dist == d) & (summ.policy == p)]
            cs.append(f"{s.AWT_vs_RR.iloc[0]:.2f}" if len(s) else "--")
            if len(s):
                nums[f"batch_{p}_{d}"] = round(s.AWT_vs_RR.iloc[0], 3)
        L.append(f"{lab} & " + " & ".join(cs) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "tab_batch.tex").write_text("\n".join(L), encoding="utf-8")

    # ---- Robustness over load and n (EED-RR vs RR and vs equal-budget, AWT)
    syn = pt[(pt.family == "synthetic") & (pt.metric == "AWT")]
    rob = []
    for d in DISTS:
        for c in ("RR", "RR-matched"):
            s = syn[(syn.dist == d) & (syn.control == c)]
            rob.append(dict(dist=d, control=c, min=round(s.rel_pct.min(), 1), max=round(s.rel_pct.max(), 1),
                            n_cells=len(s), n_sig=int((s.p_holm < 0.05).sum()),
                            n_better=int(((s.p_holm < 0.05) & (s.rel_pct < 0)).sum())))
    pd.DataFrame(rob).to_csv(out / "robustness_load_n.csv", index=False)
    nums["robustness"] = rob

    # ---- Switch cost
    sc = pd.read_csv(results / "analysis/tables/paired_EED-RR.csv")
    sc = sc[(sc.switch_cost_frac > 0) & (sc.metric == "AWT") & sc.control.isin(["RR", "RR-matched"])]
    nums["switchcost"] = sc[["dist", "switch_cost_frac", "control", "rel_pct", "p_holm"]].round(3).to_dict("records")

    # ---- sf = 1.5 vs its controls (secondary analysis)
    s15 = pt15[(pt15.switch_cost_frac == 0) & (pt15.family == "synthetic") & np.isclose(pt15.load, 0.9) &
               (pt15.n == 1000) & (pt15.metric.isin(["AWT", "SD_mean", "W_short"]))]
    nums["sf15"] = s15[["dist", "control", "metric", "rel_pct", "p_holm"]].round(3).to_dict("records")

    # ---- validation table and sf
    sel = json.loads((results / "sf_selection.json").read_text())
    nums["sf_selection"] = sel
    # ---- legacy reproduction
    lg = pd.read_csv(results / "legacy.csv")
    rep = lg[lg.stage == "legacy_repro"].groupby("policy")[["AWT", "dispatches"]].mean()
    nums["legacy_repro"] = rep.round(2).to_dict("index")
    leg = summ[(summ.family == "legacy")].set_index("policy")
    nums["legacy_ratio"] = {p: round(leg.loc[p, "AWT_vs_RR"], 3) for p in
                            ["EED-RR", "EED-RR-original(sf=1.5)", "RR(x1.5)", "RR(x2)", "FCFS", "RR-matched(sf=1.5)"]
                            if p in leg.index}
    (out / "key_numbers.json").write_text(json.dumps(nums, indent=1, default=float), encoding="utf-8")
    print("tables written to", out)


FIGURES = {"Fig2": "fig_heatmap_relative", "Fig3": "fig_waiting_by_job_class",
           "Fig4": "fig_jain_over_time", "Fig5": "fig_sensitivity_validation"}


def export_figures(results, out):
    """Write Fig1 (flowchart) and copy the analysis figures under their manuscript names."""
    import shutil

    from eedrr_bench.flowchart import draw
    out.mkdir(parents=True, exist_ok=True)
    draw(out, "Fig1")
    for name, src in FIGURES.items():
        for ext in (".tif", ".png"):
            shutil.copyfile(results / "analysis" / "figures" / (src + ext), out / (name + ext))
    print("figures written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    main(Path(a.results), Path(a.out) / "tables")
    export_figures(Path(a.results), Path(a.out) / "figures")
