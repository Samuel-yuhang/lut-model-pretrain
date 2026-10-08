"""Per-layer error curves (teacher-forced vs free-running) for several eval_layerwise.py result files.

usage: python scripts/plot_layerwise.py --out results/layerwise.png results/a.json:label results/b.json:label ...
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # fixed order
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="nmse", choices=["nmse", "rel_mse", "cos"])
    ap.add_argument("--linear", action="store_true", help="linear y axis (default log)")
    ap.add_argument("--noise_floor", type=float, default=1e-3, help="hide values below this (fp32-vs-bf16 noise)")
    ap.add_argument("runs", nargs="+", help="path.json:label")
    args = ap.parse_args()
    runs = []
    for spec in args.runs:
        path, label = spec.split(":", 1)
        with open(path) as f:
            runs.append((label, json.load(f)))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, facecolor=SURFACE)
    titles = {"teacher_forced": "Teacher-forced (layer input = teacher hidden state)",
              "free_running": "Free-running (error accumulated through the student)"}
    nlayers = len(runs[0][1]["free_running"])
    fp_sets = [set(range(nlayers)) - set(r["lut_layers"]) if r.get("lut_layers") is not None else set() for _, r in runs]
    nonempty = [f for f in fp_sets if f]
    shared = bool(nonempty) and all(f == nonempty[0] for f in nonempty)
    fp_layers = nonempty[0] if shared else set()  # one common placement -> shaded bands; otherwise per-series squares
    if args.linear:  # fix the shared range up front so direct labels are placed against final limits
        ymax = max(row[args.metric] for _, r in runs for key in ("teacher_forced", "free_running") for row in r[key])
        axes[0].set_ylim(0, ymax * 1.08)
        axes[0].set_autoscaley_on(False)
    for ax, key in zip(axes, ["teacher_forced", "free_running"], strict=True):
        ax.set_facecolor(SURFACE)
        for i in sorted(fp_layers):
            ax.axvspan(i - 0.5, i + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)
        ends = []
        for k, (label, r) in enumerate(runs):
            lut = set(range(nlayers)) if r.get("lut_layers") is None else set(r["lut_layers"])
            # teacher-forced error of a full-precision layer is only fp32-vs-bf16 noise: leave it out
            y = [row[args.metric] if (key == "free_running" or i in lut) else float("nan") for i, row in enumerate(r[key])]
            if args.metric != "cos":
                y = [v if v >= args.noise_floor else float("nan") for v in y]
            ax.plot(range(nlayers), y, color=SERIES[k], lw=2, marker="o", ms=4, mec=SURFACE, mew=1, label=label, zorder=3)
            if not shared and key == "free_running" and fp_sets[k]:
                xs = sorted(fp_sets[k])
                ax.plot(xs, [y[i] for i in xs], ls="none", marker="s", ms=7, mfc=SURFACE, mec=SERIES[k], mew=1.8, zorder=4)
            ends.append((y[-1], label))
        if key == "free_running":  # direct labels, nudged apart so they never overlap
            fig.canvas.draw()
            pts = sorted((ax.transData.transform((nlayers - 1, v))[1], lab) for v, lab in ends if v == v)
            min_gap, placed = 13.0, []
            for py, lab in pts:
                py = max(py, placed[-1][0] + min_gap) if placed else py
                placed.append((py, lab))
            for py, lab in placed:
                x_disp = ax.transData.transform((nlayers - 1, 1))[0] + 8
                xd, yd = ax.transData.inverted().transform((x_disp, py))
                ax.text(xd, yd, lab, fontsize=9, color=INK2, va="center")
        ax.set_title(titles[key], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("decoder layer (" + ("shaded = full-precision layer" if shared else
                      "hollow square = full-precision layer; gaps on the left") + ")")
        if args.metric != "cos" and not args.linear:
            ax.set_yscale("log")

        ax.grid(True, color=GRID, lw=0.6, zorder=1)
        ax.set_xlim(-0.7, nlayers - 0.3 + (7.5 if key == "free_running" else 0))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel({"nmse": "normalized MSE vs teacher" + ("" if args.linear else " (log)"), "rel_mse": "relative MSE (log)", "cos": "cosine"}[args.metric])
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
