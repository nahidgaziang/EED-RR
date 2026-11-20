"""Flowchart of the EED-RR scheduling loop (Fig 1 of the paper).

  python -m eedrr_bench.flowchart out_dir      # writes Fig1.tif and Fig1.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from .analysis import AQUA, BLUE, INK, INK2, ORANGE, _save, _style

PROC_FILL, EED_FILL, DEC_FILL, TERM_FILL = "#f3f7fd", "#dbe8fa", "#fff3ea", "#e6f6ef"


def draw(out_dir: str | Path, name: str = "Fig1"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon

    _style()
    fig, ax = plt.subplots(figsize=(5.6, 8.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 17)
    ax.axis("off")

    def box(x, y, w, h, text, fill=PROC_FILL, edge=BLUE, rounded=0.08, size=7.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={rounded}",
                                    fc=fill, ec=edge, lw=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.35)

    def diamond(x, y, w, h, text):
        ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                             closed=True, fc=DEC_FILL, ec=ORANGE, lw=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.2)

    def arrow(points, label=None, label_at=None, ha="left"):
        xs, ys = zip(*points)
        if len(points) > 2:
            ax.plot(xs[:-1], ys[:-1], color=INK2, lw=0.9, solid_capstyle="butt")
        ax.annotate("", xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0,
                                    mutation_scale=9))
        if label:
            ax.text(*label_at, label, fontsize=7, color=INK2, ha=ha, va="center", style="italic")

    def line(points):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=INK2, lw=0.9)

    X, W, H = 5.0, 6.0, 0.8
    y_start, y_init, y_admit, y_empty = 16.35, 15.15, 13.85, 12.45
    y_pop, y_fair, y_beta, y_q, y_run = 11.15, 9.95, 8.85, 7.75, 6.45
    y_rem, y_done_box, y_all, y_end = 5.05, 3.75, 2.35, 0.75

    box(X, y_start, 4.6, 0.7, r"Start: processes $(a_i, b_i)$, scaling factor $sf$",
        fill=TERM_FILL, edge=AQUA, rounded=0.3)
    box(X, y_init, W, H, r"$TQ_{\mathrm{base}} = (\mathrm{mean}(b) + \mathrm{median}(b))\,/\,2$"
        "\n" r"$t = 0$;  ready queue $Q = \emptyset$;  active set $A = \emptyset$")
    box(X, y_admit, W, H, "Admit arrivals: append every process with $a_k \\leq t$\n"
                          "to the tail of $Q$ and to $A$")
    diamond(X, y_empty, 2.6, 1.0, "$Q$ empty?")
    box(8.45, y_empty, 2.3, 0.7, "$t \\leftarrow$ next\narrival time")
    box(X, y_pop, W, 0.65, "$P_i \\leftarrow$ process at the head of $Q$")
    box(X, y_fair, W, H, r"$x_k = \mathrm{cpu}_k\,/\,(t - a_k)$  for every $k \in A$"
        "\n" r"$J$ = Jain's index of $x$ over $A$;   $\bar{x}$ = mean of $x$",
        fill=EED_FILL)
    box(X, y_beta, W, 0.65, r"$\beta_i = \max\left(0,\ (\bar{x} - x_i)\,/\,\bar{x}\right)$", fill=EED_FILL)
    box(X, y_q, W, 0.65, r"$q = \min\left(TQ_{\mathrm{base}}\,(1 + (1 - J)\,\beta_i\,sf),\ r_i\right)$",
        fill=EED_FILL)
    box(X, y_run, W, H, "Run $P_i$ for $q$:  $t \\leftarrow t + q$,  $\\mathrm{cpu}_i \\leftarrow \\mathrm{cpu}_i + q$,"
                        "  $r_i \\leftarrow r_i - q$\nAdmit processes that arrived during the run")
    diamond(X, y_rem, 2.6, 1.0, "$r_i > 0$?")
    box(1.75, y_rem, 2.3, 0.7, "Append $P_i$ to\nthe tail of $Q$")
    box(X, y_done_box, W, 0.75, "Record completion time of $P_i$\nand remove $P_i$ from $A$")
    diamond(X, y_all, 3.2, 1.15, "All processes\ncompleted?")
    box(X, y_end, 5.2, 0.75, "End: report waiting, response time, slowdown,\nJain's index and context switches",
        fill=TERM_FILL, edge=AQUA, rounded=0.3)

    # main column
    for y0, h0, y1, h1 in [(y_start, 0.7, y_init, H), (y_init, H, y_admit, H), (y_admit, H, y_empty, 1.0),
                           (y_pop, 0.65, y_fair, H), (y_fair, H, y_beta, 0.65), (y_beta, 0.65, y_q, 0.65),
                           (y_q, 0.65, y_run, H), (y_run, H, y_rem, 1.0)]:
        arrow([(X, y0 - h0 / 2), (X, y1 + h1 / 2)])
    arrow([(X, y_empty - 0.5), (X, y_pop + 0.325)], "No", (X + 0.15, y_empty - 0.8))
    arrow([(X, y_rem - 0.5), (X, y_done_box + 0.375)], "No", (X + 0.15, y_rem - 0.8))
    arrow([(X, y_done_box - 0.375), (X, y_all + 0.575)])
    arrow([(X, y_all - 0.575), (X, y_end + 0.375)], "Yes", (X + 0.15, y_all - 0.85))

    # Q empty -> advance clock -> admit again
    arrow([(X + 1.3, y_empty), (8.45 - 1.15, y_empty)], "Yes", (X + 1.4, y_empty + 0.22))
    arrow([(8.45, y_empty + 0.35), (8.45, y_admit), (X + W / 2, y_admit)])

    # r_i > 0 -> re-queue -> admit again;  not all completed -> admit again
    arrow([(X - 1.3, y_rem), (1.75 + 1.15, y_rem)], "Yes", (X - 1.4, y_rem + 0.22), ha="right")
    line([(1.75, y_rem + 0.35), (1.75, y_admit - 0.25)])
    arrow([(X - 1.6, y_all), (0.45, y_all), (0.45, y_admit - 0.25), (X - W / 2, y_admit - 0.25)],
          "No", (X - 1.75, y_all + 0.22), ha="right")
    ax.plot([1.75], [y_admit - 0.25], "o", ms=2.5, color=INK2)

    # bracket marking the EED-RR-specific steps
    top, bot = y_fair + H / 2, y_q - 0.325
    ax.plot([8.15, 8.3, 8.3, 8.15], [top, top, bot, bot], color=BLUE, lw=1.0)
    ax.text(8.45, (top + bot) / 2, "EED-RR\nelastic\nquantum", fontsize=7.5, color=BLUE,
            ha="left", va="center", linespacing=1.3)

    _save(fig, Path(out_dir), name)


if __name__ == "__main__":
    draw(sys.argv[1] if len(sys.argv) > 1 else ".")
