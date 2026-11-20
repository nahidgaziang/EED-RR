"""Tables and PLOS-ready figures from the experiment CSVs.

Outputs (under <out>/):
  tables/summary_by_cell.csv     mean of every metric per (family, dist, load, n, policy)
  tables/paired_<proposed>.csv   paired comparisons proposed vs every control, every cell
  headline.md                    readable digest of the paired comparisons
  figures/*.tif (PLOS: RGB, LZW, 300 dpi) + *.png previews + *_data.csv (values plotted)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import holm, paired_summary

KEY = ["family", "dist", "load", "n", "switch_cost_frac"]
METRICS = ["AWT", "resp_mean", "W_short", "W_long", "SD_mean", "SD_max", "W_p95", "W_p99", "W_max",
           "W_var", "JFI_final", "JFI_t_mean", "preemptions", "switches"]
CONTROLS = ["RR", "RR(mean)", "RR(x1.25)", "RR(x1.5)", "RR(x2)", "RR-matched", "RandomBoost",
            "EED[JFI-only]", "EED[deficit-only]", "EED[all,dispatch]", "EED[all,continuous]",
            "EED[active,dispatch]", "EED-RR-original(sf=1.5)", "MARR", "EImHLVQTRR", "ADRR*",
            "SARR", "STS-RR", "FCFS", "SJF", "SRTF"]
CONTROLS_15 = ["RR", "RR-matched(sf=1.5)", "RandomBoost(sf=1.5)", "EED-RR-original(sf=1.5)",
               "MARR", "EImHLVQTRR", "FCFS", "SRTF"]
LABEL = {"AWT": "Mean waiting time", "resp_mean": "Mean response time",
         "W_short": "Waiting, short third", "W_long": "Waiting, long third",
         "SD_mean": "Mean slowdown", "SD_max": "Max slowdown", "W_p95": "p95 waiting",
         "W_p99": "p99 waiting", "W_max": "Max waiting", "W_var": "Waiting variance",
         "JFI_final": "Jain index (final)",
         "JFI_t_mean": "Jain index (time-avg)", "preemptions": "Preemptions",
         "switches": "Process switches"}

# Colour-blind-safe categorical colours, a neutral reference colour and a diverging pair.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REF = "#898781"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIV_LO, DIV_MID, DIV_HI = "#2a78d6", "#f0efec", "#e34948"

DIST_ORDER = ["gaussian", "exponential", "lognormal", "pareto", "bimodal"]
DIST_NAME = {"gaussian": "Gaussian", "exponential": "Exponential", "lognormal": "Lognormal",
             "pareto": "Pareto", "bimodal": "Bimodal"}
POLICY_NAME = {"RR-matched": "Equal-budget RR", "RandomBoost": "Random boost",
               "EED[JFI-only]": "JFI only", "EED[deficit-only]": "Deficit only",
               "EED[all,dispatch]": "Original state def.", "RR(x1.5)": "RR, 1.5x quantum"}


def ordered(dists):
    """Distributions in a fixed order (the order of the paper's tables)."""
    present = list(dict.fromkeys(dists))
    return [d for d in DIST_ORDER if d in present] + [d for d in present if d not in DIST_ORDER]


def dname(d):
    return DIST_NAME.get(d, d)


def pname(p):
    return POLICY_NAME.get(p, p)


# ------------------------------------------------------------------------- loading
def load(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(results_dir)
    out = {}
    for name in ("validation", "test", "batch", "legacy", "trace", "switchcost", "timeseries"):
        p = d / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["load"] = df["load"].fillna(-1.0)
            df["switch_cost_frac"] = df["switch_cost_frac"].fillna(0.0)
            out[name] = df
    p = d / "timeseries_J.csv"
    if p.exists():
        out["series"] = pd.read_csv(p)
    return out


def summary_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(KEY + ["policy"], dropna=False)[METRICS + ["AWT", "ATT"]].mean()
    g = g.loc[:, ~g.columns.duplicated()].reset_index()
    rr = g[g.policy == "RR"].set_index(KEY)
    for m in ("AWT", "resp_mean", "SD_mean", "W_p99", "W_short"):
        g[f"{m}_vs_RR"] = g.apply(lambda r: r[m] / rr.loc[tuple(r[k] for k in KEY), m]
                                  if tuple(r[k] for k in KEY) in rr.index else np.nan, axis=1)
    return g


def paired(df: pd.DataFrame, proposed: str, controls: list[str], metrics=METRICS,
           n_boot: int = 10000) -> pd.DataFrame:
    rows = []
    for cell, g in df.groupby(KEY, dropna=False):
        if proposed not in set(g.policy):
            continue
        for m in metrics:
            piv = g.pivot_table(index="seed", columns="policy", values=m, aggfunc="first")
            for c in controls:
                if c not in piv.columns or c == proposed:
                    continue
                ok = piv[[proposed, c]].dropna()
                if len(ok) < 2:
                    continue
                s = paired_summary(ok[proposed].values, ok[c].values, metric=m, n_boot=n_boot)
                rows.append({**dict(zip(KEY, cell)), "proposed": proposed, "control": c,
                             "metric": m, **s})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("metric")["p"].transform(lambda p: holm(p.values))
    return out


# ------------------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": BASE,
        "axes.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.labelcolor": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.5,
        "lines.markersize": 4.5, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _save(fig, out: Path, name: str, data: pd.DataFrame | None = None):
    import io

    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight", format="png")
    buf.seek(0)
    Image.open(buf).convert("RGB").save(out / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    if data is not None:
        data.to_csv(out / f"{name}_data.csv", index=False)
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_heatmaps(pt: pd.DataFrame, out: Path, n: int = 1000, proposed: str = "EED-RR"):
    """Relative difference (%) of the proposed method vs controls, dist x load."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    panels = [("RR", "AWT"), ("RR", "SD_mean"), ("RR-matched", "AWT")]
    titles = ["A  Mean waiting vs RR", "B  Mean slowdown vs RR",
              "C  Mean waiting vs equal-budget RR"]
    sub = pt[(pt.family == "synthetic") & (pt.n == n) & (pt.proposed == proposed)]
    if sub.empty:
        return
    dists = ordered(sub.dist)
    loads = sorted(sub.load.unique())
    cmap = LinearSegmentedColormap.from_list("div", [DIV_LO, DIV_MID, DIV_HI])
    mats, data = [], []
    for ctrl, m in panels:
        M = np.full((len(dists), len(loads)), np.nan)
        P = np.full_like(M, np.nan)
        for i, d in enumerate(dists):
            for j, L in enumerate(loads):
                r = sub[(sub.control == ctrl) & (sub.metric == m) & (sub.dist == d) & (sub.load == L)]
                if len(r):
                    M[i, j], P[i, j] = r.rel_pct.iloc[0], r.p_holm.iloc[0]
                    data.append(dict(control=ctrl, metric=m, dist=d, load=L,
                                     rel_pct=M[i, j], p_holm=P[i, j]))
        mats.append((M, P))
    vmax = max(5.0, np.nanmax([np.nanmax(np.abs(M)) for M, _ in mats]))
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharey=True)
    for ax, (M, P), title in zip(axes, mats, titles):
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    continue
                star = "*" if P[i, j] < 0.05 else ""
                col = "white" if abs(M[i, j]) > 0.6 * vmax else INK
                ax.text(j, i, f"{M[i, j]:+.1f}{star}", ha="center", va="center", fontsize=6.5,
                        color=col)
        ax.set_xticks(range(len(loads)), [f"{L:g}" for L in loads])
        ax.set_xlabel("Offered load ρ")
        ax.set_title(title, loc="left")
    axes[0].set_yticks(range(len(dists)), [dname(d) for d in dists])
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Difference, % (negative = proposed better)")
    cb.outline.set_visible(False)
    fig.text(0.0, -0.06, f"n = {n}; paired over independent seeds; * Holm-adjusted p < 0.05 "
             "(Wilcoxon signed-rank). For Jain/fairness metrics higher is better.",
             fontsize=6.5, color=INK2)
    _save(fig, out, "fig_heatmap_relative", pd.DataFrame(data))


def fig_sensitivity(val: pd.DataFrame, out: Path):
    """Validation-set trade-off of the scaling factor (geometric-mean ratio to RR)."""
    import matplotlib.pyplot as plt
    key = ["dist", "load", "n", "seed"]
    rr = val[val.policy == "RR"].set_index(key)
    eed = val[val.policy == "EED-RR"]
    series = [("AWT", "Mean waiting", BLUE, "o"), ("W_short", "Short-job waiting", ORANGE, "s"),
              ("SD_mean", "Mean slowdown", AQUA, "^")]
    dists = ordered(val.dist)
    fig, axes = plt.subplots(1, len(dists), figsize=(7.5, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, d in zip(axes, dists):
        for m, lab, col, mk in series:
            xs, ys = [], []
            for sf, g in eed[eed.dist == d].groupby("sf"):
                g = g.set_index(key)
                common = g.index.intersection(rr.index)
                r = np.exp(np.log(g.loc[common, m].astype(float) / rr.loc[common, m].astype(float)).mean())
                xs.append(float(sf))
                ys.append(r)
                rows.append(dict(dist=d, metric=m, sf=float(sf), gm_ratio_vs_RR=r))
            ax.plot(xs, ys, color=col, marker=mk, label=lab)
        ax.axhline(1.0, color=REF, lw=1, ls="--")
        ax.set_title(dname(d), loc="left")
        ax.set_xlabel("Scaling factor sf")
    axes[0].set_ylabel("Ratio to RR (validation)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    _save(fig, out, "fig_sensitivity_validation", pd.DataFrame(rows))


def fig_jfi_time(series: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    show = [("EED-RR", BLUE, "-"), ("RR", REF, "--"), ("MARR", ORANGE, "-")]
    dists = ordered(series.dist)
    fig, axes = plt.subplots(1, len(dists), figsize=(7.5, 2.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dists):
        s = series[series.dist == d]
        tmax = s.t.max()
        for pol, col, ls in show:
            g = s[s.policy == pol].sort_values("t")
            if len(g):
                sm = g.J.rolling(15, center=True, min_periods=5).mean()
                ax.plot(g.t / tmax, sm, color=col, ls=ls, lw=1.3, label=pname(pol))
        ax.set_title(dname(d), loc="left")
        ax.set_xlabel("Time / makespan")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Jain index of x(t)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.0, -0.14, "Rolling mean over 15 of 200 sampling instants; one workload per panel "
             "(ρ = 0.9, n = 1000). Raw samples are in the data file.", fontsize=6.5, color=INK2)
    _save(fig, out, "fig_jain_over_time", series[series.policy.isin([p for p, _, _ in show])])


def fig_class_waiting(test: pd.DataFrame, out: Path, n: int = 1000, load: float = 0.9):
    import matplotlib.pyplot as plt
    sub = test[(test.family == "synthetic") & (test.n == n) & np.isclose(test.load, load)]
    if sub.empty:
        return
    pols = [("EED-RR", BLUE), ("RR-matched", ORANGE), ("RandomBoost", AQUA)]
    classes = [("W_short", "short"), ("W_medium", "medium"), ("W_long", "long")]
    dists = ordered(sub.dist)
    fig, axes = plt.subplots(1, len(dists), figsize=(7.5, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    w = 0.26
    for ax, d in zip(axes, dists):
        g = sub[sub.dist == d].groupby("policy")[[c for c, _ in classes]].mean()
        for k, (pol, col) in enumerate(pols):
            if pol not in g.index:
                continue
            vals = [100.0 * (g.loc[pol, c] / g.loc["RR", c] - 1.0) for c, _ in classes]
            x = np.arange(len(classes)) + (k - 1) * w
            ax.bar(x, vals, width=w * 0.9, color=col, label=pname(pol), edgecolor="white", linewidth=0.8)
            rows += [dict(dist=d, policy=pol, job_class=cl, pct_change_vs_RR=v)
                     for (_, cl), v in zip(classes, vals)]
        ax.axhline(0.0, color=BASE, lw=0.8)
        ax.set_xticks(range(len(classes)), [cl for _, cl in classes])
        ax.set_title(dname(d), loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"Mean waiting, % vs RR\n(ρ={load:g}, n={n}; burst terciles)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    _save(fig, out, "fig_waiting_by_job_class", pd.DataFrame(rows))


def fig_forest(pt: pd.DataFrame, out: Path, family: str, n: int | None, load: float | None,
               proposed: str = "EED-RR", tag: str = ""):
    import matplotlib.pyplot as plt
    sub = pt[(pt.family == family) & (pt.proposed == proposed)]
    if n is not None:
        sub = sub[sub.n == n]
    if load is not None:
        sub = sub[np.isclose(sub.load, load)]
    if sub.empty:
        return
    metrics = ["AWT", "resp_mean", "SD_mean", "W_p99"]
    # Oracle references (FCFS/SJF/SRTF) differ by up to 10-20x and would flatten every
    # other row; they are reported in the tables instead.
    ctrls = [c for c in ["RR", "RR(x1.5)", "RR-matched", "RandomBoost", "EED[JFI-only]",
                         "EED[deficit-only]", "EED[all,dispatch]", "MARR", "EImHLVQTRR"]
             if c in set(sub.control)]
    for (d, Lg), s in sub.groupby(["dist", "load"], sort=False):
        if s.n.nunique() > 1:  # one figure per workload size
            s = s[s.n == s.n.max()]
        ltag = f"_rho{Lg:g}" if (load is None and Lg >= 0) else ""
        fig, axes = plt.subplots(1, len(metrics), figsize=(7.5, 0.28 * len(ctrls) + 0.9), sharey=True)
        for ax, m in zip(axes, metrics):
            r = s[s.metric == m].set_index("control").reindex(ctrls)
            y = np.arange(len(ctrls))[::-1]
            ax.errorbar(r.rel_pct, y, xerr=[r.rel_pct - r.rel_ci_lo, r.rel_ci_hi - r.rel_pct],
                        fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1, capsize=0, ms=3.5)
            ax.axvline(0, color=REF, lw=1, ls="--")
            ax.set_title(LABEL[m], loc="left")
            ax.set_xlabel("% vs control")
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(np.arange(len(ctrls))[::-1], [pname(c) for c in ctrls])
        _save(fig, out, f"supp_forest_{family}_{d}{ltag}{tag}",
              s[s.metric.isin(metrics)][["control", "metric", "rel_pct", "rel_ci_lo",
                                         "rel_ci_hi", "p_holm", "wins", "n_pairs"]])


# -------------------------------------------------------------------------- digest
def _fmt(r):
    sig = "*" if r.p_holm < 0.05 else ""
    return f"{r.rel_pct:+.1f}% [{r.rel_ci_lo:+.1f}, {r.rel_ci_hi:+.1f}]{sig}"


def headline(pt_all: pd.DataFrame, path: Path, sel: dict | None, n_main: int = 1000,
             load_main: float = 0.9):
    pt = pt_all[pt_all.switch_cost_frac == 0]
    lines = ["# EED-RR: results digest", ""]
    if sel:
        lines += [f"Scaling factor (validation rule: {sel.get('rule')}): **sf = {sel.get('chosen_sf')}**", ""]
    lines += ["Cells show EED-RR vs control: relative difference of means "
              "[95% bootstrap CI]; * = Holm-adjusted Wilcoxon p < 0.05. Negative = EED-RR lower "
              "(better for time metrics; worse for Jain index).", ""]
    metrics = ["AWT", "resp_mean", "W_short", "SD_mean", "W_p99", "JFI_final"]
    ctrls = ["RR", "RR-matched", "RandomBoost", "EED[JFI-only]", "EED[deficit-only]",
             "EED[all,dispatch]", "MARR", "EImHLVQTRR", "FCFS", "SRTF"]
    def table(g, title, ctrl_list):
        out = [f"## {title}", "",
               "| control | " + " | ".join(LABEL[m] for m in metrics) + " |",
               "|---|" + "---|" * len(metrics)]
        for c in ctrl_list:
            cells = []
            for m in metrics:
                r = g[(g.control == c) & (g.metric == m)]
                cells.append(_fmt(r.iloc[0]) if len(r) else "")
            if any(cells):
                out.append(f"| {c} | " + " | ".join(cells) + " |")
        return out + [""]

    for (fam, d, L, n), g in pt[pt.proposed == "EED-RR"].groupby(["family", "dist", "load", "n"]):
        if fam == "synthetic" and not (n == n_main and np.isclose(L, load_main)):
            continue
        lines += table(g, f"{fam} / {d}" + (f" / rho={L:g}" if L >= 0 else "") + f" / n={n}", ctrls)
    sc = pt_all[(pt_all.switch_cost_frac > 0) & (pt_all.proposed == "EED-RR")]
    if len(sc):
        lines += ["# Switch-cost scenario (cost = fraction x TQ_base per process switch)", ""]
        for (d, f), g in sc.groupby(["dist", "switch_cost_frac"]):
            lines += table(g, f"{d} / switch cost {f:g} x TQ_base", ["RR", "RR-matched", "RandomBoost",
                                                                    "MARR", "EImHLVQTRR", "FCFS", "SRTF"])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def analyze(results_dir: str | Path, out_dir: str | Path | None = None, n_boot: int = 10000):
    import json
    R = load(results_dir)
    out = Path(out_dir or Path(results_dir) / "analysis")
    tabs, figs = out / "tables", out / "figures"
    tabs.mkdir(parents=True, exist_ok=True)
    _style()
    frames = [R[k] for k in ("test", "batch", "legacy", "trace", "switchcost") if k in R]
    allres = pd.concat(frames, ignore_index=True)
    allres = allres[allres.stage != "legacy_repro"]
    summary_by_cell(allres).to_csv(tabs / "summary_by_cell.csv", index=False)
    pt = paired(allres, "EED-RR", CONTROLS, n_boot=n_boot)
    pt.to_csv(tabs / "paired_EED-RR.csv", index=False)
    pt15 = paired(allres, "EED-RR(sf=1.5)", CONTROLS_15, n_boot=n_boot)
    pt15.to_csv(tabs / "paired_EED-RR_sf1.5.csv", index=False)
    if "legacy" in R:
        lg = R["legacy"]
        lg[lg.stage == "legacy_repro"].to_csv(tabs / "legacy_reproduction.csv", index=False)
    sel_p = Path(results_dir) / "sf_selection.json"
    sel = json.loads(sel_p.read_text()) if sel_p.exists() else None
    pt0 = pt[pt.switch_cost_frac == 0]
    n_main, L = 1000, 0.9
    if "test" in R:
        ns = sorted(R["test"].n.unique())
        n_main = 1000 if 1000 in ns else ns[0]
        loads = sorted(R["test"].load.unique())
        L = 0.9 if any(np.isclose(loads, 0.9)) else loads[-1]
        fig_heatmaps(pt0, figs, n=n_main)
        fig_class_waiting(R["test"], figs, n=n_main, load=L)
        fig_forest(pt0, figs, "synthetic", n_main, L)
    headline(pt, out / "headline.md", sel, n_main=n_main, load_main=L)
    if "validation" in R:
        fig_sensitivity(R["validation"], figs)
    if "series" in R:
        fig_jfi_time(R["series"], figs)
    for fam in ("batch", "legacy", "trace"):
        fig_forest(pt0, figs, fam, None, None)
    print(f"analysis written to {out}")
    return out
