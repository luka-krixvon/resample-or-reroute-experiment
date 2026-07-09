"""Generate paper-quality assets from the experiment CSVs.

Outputs (into ../paper/):
  figs/fig_pareto.pdf    3-panel cost-quality Pareto (gsm8k / math500 / gpqa)
  figs/fig_verifier.pdf  verifier-quality ablation (method acc @ mid budget vs q)
  tab_results.tex        matched-cost comparison table (generated, not hand-typed)

House style: serif/STIX, grey #8A8F94 baselines, blue #3B6EA5 method,
amber #C0873E accents, opaque legend background, vector PDF.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper_canonical")
FIGS = os.path.join(PAPER, "figs")
os.makedirs(FIGS, exist_ok=True)

GREY, BLUE, AMB = "#8A8F94", "#3B6EA5", "#C0873E"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 9, "axes.titlesize": 9.5})

BENCHES = [("gsm8k", "GSM8K (saturated)"),
           ("math500", "MATH-500 (intermediate)"),
           ("gpqa", "GPQA-Diamond (hard, heterogeneous)"),
           ("humanevalplus", "HumanEval+ (code)")]

# display name, color, linestyle, marker, is_method
STYLES = {
    "resample_or_reroute":     ("Resample-or-Reroute (ours)", BLUE, "-", "o", True),
    "resample_or_reroute_ucb": ("ours (UCB variant)",         BLUE, ":", "^", True),
    "oracle_alloc":            ("oracle allocation (ceiling)", AMB, "--", "", False),
    "fixed_best_of_k":         ("budget-aware best-of-$K$",   "#555a60", "-", "s", False),
    "frugal_cascade":          ("cascade (FrugalGPT-style)",  GREY, "-", "D", False),
    "random_alloc":            ("random allocation",          "#B7BCC1", "-", "v", False),
    "single_route":            ("single-route (best model)",  "#555a60", "", "*", False),
    "learned_router":          ("router (one commit)",        GREY, "", "P", False),
}


def load(bench, q="1.0"):
    path = os.path.join(HERE, "results", f"pareto_{bench}_q{q}.csv")
    by = {}
    for r in csv.DictReader(open(path)):
        by.setdefault(r["policy"], []).append(
            (float(r["mean_cost"]), float(r["accuracy"])))
    return {k: sorted(v) for k, v in by.items()}


def nearest(pts, target):
    return min(pts, key=lambda p: abs(p[0] - target))


def fig_pareto():
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.1))
    for ax, (bench, title) in zip(axes, BENCHES):
        by = load(bench)
        # oracle ceiling as a horizontal reference line (its cost saturates ~7,
        # so as a curve it collapses to an invisible point)
        ceil = by["oracle_alloc"][-1][1]
        ax.axhline(ceil, color=AMB, ls="--", lw=1.2,
                   label="oracle allocation (ceiling)", zorder=2)
        for pol in ["random_alloc", "frugal_cascade", "fixed_best_of_k",
                    "resample_or_reroute_ucb", "resample_or_reroute"]:
            if pol not in by:
                continue
            name, color, ls, mk, is_m = STYLES[pol]
            xs, ys = zip(*by[pol])
            ax.plot(xs, ys, ls=ls or "-", marker=mk or None, color=color,
                    lw=2.2 if is_m and ls == "-" else 1.2,
                    ms=4.5 if is_m else 3.5, label=name,
                    zorder=5 if is_m else 3, alpha=1.0 if is_m else 0.9)
        for pol in ["single_route", "learned_router"]:
            name, color, _, mk, _ = STYLES[pol]
            c, a = by[pol][0]
            ax.plot([c], [a], marker=mk, color=color, ms=9 if mk == "*" else 6,
                    ls="", label=name, zorder=4)
        ax.set_title(title)
        ax.set_xlabel("mean cost per query (size proxy)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5,
               frameon=True, facecolor="white", framealpha=1.0,
               edgecolor="#D9DCDF", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_pareto.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote figs/fig_pareto.pdf")


def fig_verifier():
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    qs = ["0.6", "0.8", "1.0"]
    colors = {"gsm8k": GREY, "math500": BLUE, "gpqa": AMB, "humanevalplus": "#4E7D57"}
    for bench, title in BENCHES:
        accs = []
        for q in qs:
            by = load(bench, q)
            accs.append(nearest(by["resample_or_reroute"], 26)[1])
        ax.plot([float(q) for q in qs], accs, "-o", color=colors[bench],
                lw=1.8, ms=4.5, label=title.split(" (")[0])
    ax.set_xlabel("verifier quality $q$")
    ax.set_ylabel("accuracy @ matched budget")
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, frameon=True, facecolor="white", framealpha=1.0,
              edgecolor="#D9DCDF")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_verifier.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote figs/fig_verifier.pdf")


def tab_results():
    """Matched-cost table: accuracy at the point nearest mean cost ~26 (mid budget),
    plus each policy's actual mean cost there. Single-commit baselines are shown
    at their (budget-independent) operating point."""
    lines = [
        "% AUTO-GENERATED by experiments/make_paper_assets.py -- do not hand-edit.",
        "\\begin{table}[t]\\centering",
        "\\caption{Accuracy at a matched mid budget (point nearest mean cost $26$;"
        " cost = parameter-size proxy). Single-commit baselines are budget-independent"
        " and shown at their fixed operating point. Oracle allocation is a ceiling,"
        " not a deployable policy.}",
        "\\label{tab:results}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccccc}\\toprule",
        " & \\multicolumn{2}{c}{GSM8K} & \\multicolumn{2}{c}{MATH-500} & \\multicolumn{2}{c}{GPQA} & \\multicolumn{2}{c}{HumanEval+} \\\\",
        "Policy & cost & acc & cost & acc & cost & acc & cost & acc \\\\ \\midrule",
    ]
    order = ["resample_or_reroute", "resample_or_reroute_ucb", "fixed_best_of_k",
             "frugal_cascade", "random_alloc", "learned_router", "single_route",
             "oracle_alloc"]
    for pol in order:
        name = STYLES[pol][0].replace("$K$", "$K$")
        row = [name]
        for bench, _ in BENCHES:
            by = load(bench)
            c, a = nearest(by[pol], 26)
            row += [f"{c:.1f}", f"{a:.3f}"]
        bold = pol == "resample_or_reroute"
        cells = row[1:]
        if bold:
            cells = [f"\\textbf{{{x}}}" for x in cells]
            row = [f"\\textbf{{{row[0]}}}"] + cells
        else:
            row = [row[0]] + cells
        sep = " \\\\ \\midrule" if pol == "random_alloc" else " \\\\"
        lines.append(" & ".join(row) + sep)
    lines += ["\\bottomrule\\end{tabular}}\\end{table}"]
    out = os.path.join(PAPER, "tab_results.tex")
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote tab_results.tex")


if __name__ == "__main__":
    fig_pareto()
    fig_verifier()
    tab_results()
