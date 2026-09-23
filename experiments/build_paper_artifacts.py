"""Build every LaTeX artifact of the manuscript from experimental outputs.

Reads results/{oulad,ednet}_results.json, results/preds_*.csv,
results/attention_*.npy and results/shap_oulad.json, and emits:
  manuscript/numbers.tex                       -- numeric macros used by main.tex
  manuscript/table_leakage.tex                 -- headline contemporaneous-vs-prospective table
  manuscript/table_oulad_prospective.tex       -- full metric suite, OULAD
  manuscript/table_ednet_prospective.tex       -- full metric suite, EdNet-KT1
  figures/fig1..fig6 (PDF)

No reported value is transcribed by hand; edit this script, not the .tex,
to change how numbers are presented.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
MAN = os.path.join(ROOT, "manuscript")
FIG = os.path.join(ROOT, "manuscript", "figures")

BLUE, VIOLET, RED, AQUA = "#2a78d6", "#4a3aa7", "#e34948", "#1baf7a"
DISP = {"LogisticRegression": "Logistic Regression", "RandomForest": "Random Forest",
        "GradientBoosting": "Gradient Boosting", "XGBoost": "XGBoost",
        "LSTM": "LSTM", "BiLSTM-Attention": "BiLSTM-Attention"}
GRAY = "#6b6a63"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150,
})
MODELS = ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost",
          "LSTM", "BiLSTM-Attention"]
TREE = ["RandomForest", "GradientBoosting", "XGBoost"]


def load(dataset):
    with open(os.path.join(RES, f"{dataset}_results.json")) as fh:
        return json.load(fh)


def get(res, dataset, regime, model):
    for r in res["results"]:
        if r["dataset"] == dataset and r["regime"] == regime and r["model"] == model:
            return r
    raise KeyError((dataset, regime, model))


def conditional_severe_over(dataset, regime, model):
    """P(y_pred >= y_true + 2 | y_true <= 1): false-alarm side of the
    two-direction severe-error analysis."""
    df = pd.read_csv(os.path.join(RES, f"preds_{dataset}_{regime}_{model}.csv"))
    low = df[df.y_true <= 1]
    return float((low.y_pred >= low.y_true + 2).mean())


def conditional_severe_under(dataset, regime, model):
    """P(y_pred <= y_true - 2 | y_true >= 2) from saved test predictions."""
    df = pd.read_csv(os.path.join(RES, f"preds_{dataset}_{regime}_{model}.csv"))
    sev = df[df.y_true >= 2]
    return float((sev.y_pred <= sev.y_true - 2).mean())


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def ci_str(ci, nd=3):
    return f"[{ci['lo']:.{nd}f},\\,{ci['hi']:.{nd}f}]"


# ── numbers.tex ──────────────────────────────────────────────────────────────
def build_numbers(ou, ed):
    s_ou, s_ed = ou["summary"], ed["summary"]
    n = {}

    def mac(k, v):
        n[k] = v

    mac("nOULADenrol", f"{s_ou['enrolments']:,}".replace(",", "\\,"))
    mac("nOULADstudents", f"{s_ou['students']:,}".replace(",", "\\,"))
    mac("nEdNetusers", f"{s_ed['users']:,}".replace(",", "\\,"))
    mac("nEdNetBins", "10")
    tot_ou = sum(s_ou["label_distribution"].values())
    tot_ed = sum(s_ed["label_distribution"].values())
    for k in range(4):
        mac(f"ouladL{k}".replace("0", "zero").replace("1", "one").replace("2", "two").replace("3", "three"),
            f"{100*s_ou['label_distribution'][str(k)]/tot_ou:.1f}\\%")
        mac(f"ednetL{k}".replace("0", "zero").replace("1", "one").replace("2", "two").replace("3", "three"),
            f"{100*s_ed['label_distribution'][str(k)]/tot_ed:.1f}\\%")

    bilm = get(ou, "oulad", "prospective", "BiLSTM-Attention")
    mac("bilstmParams", f"{bilm['params']:,}".replace(",", "\\,"))

    gaps = {"oulad": {"acc": [], "qwk": []}, "ednet": {"acc": [], "qwk": []}}
    for ds, res in (("oulad", ou), ("ednet", ed)):
        for m in MODELS:
            pro, con = get(res, ds, "prospective", m), get(res, ds, "contemporaneous", m)
            gaps[ds]["acc"].append(con["metrics"]["accuracy"] - pro["metrics"]["accuracy"])
            gaps[ds]["qwk"].append(con["metrics"]["qwk"] - pro["metrics"]["qwk"])
    for ds, tag in (("oulad", "OULAD"), ("ednet", "EdNet")):
        a, q = gaps[ds]["acc"], gaps[ds]["qwk"]
        mac(f"gap{tag}accRange", f"{100*min(a):.1f}--{100*max(a):.1f}")
        mac(f"gap{tag}qwkRange", f"{min(q):.2f}--{max(q):.2f}")
        mac(f"gap{tag}qwkMax", f"{max(q):.2f}")

    xgb = get(ou, "oulad", "prospective", "XGBoost")
    mac("ouladProXGBFone", fmt(xgb["metrics"]["macro_f1"]))
    mac("ouladProXGBqwk", fmt(xgb["metrics"]["qwk"]))
    mac("ouladProBiLSTMFone", fmt(bilm["metrics"]["macro_f1"]))
    mac("ouladProXGBoff", f"{100*xgb['metrics']['one_off_accuracy']:.1f}\\%")
    mac("ouladProXGBsev", f"{100*conditional_severe_under('oulad', 'prospective', 'XGBoost'):.1f}\\%")
    mac("ouladProXGBsevover", f"{100*conditional_severe_over('oulad', 'prospective', 'XGBoost'):.1f}\\%")

    best = max(TREE, key=lambda m: get(ed, "ednet", "prospective", m)["metrics"]["macro_f1"])
    bt = get(ed, "ednet", "prospective", best)
    disp = {"RandomForest": "Random Forest", "GradientBoosting": "Gradient Boosting",
            "XGBoost": "XGBoost"}[best]
    mac("ednetBestSummary",
        f"{disp} macro-F1 ${fmt(bt['metrics']['macro_f1'])}$, QWK ${fmt(bt['metrics']['qwk'])}$")
    mac("ednetProBiLSTMFone", fmt(get(ed, "ednet", "prospective", "BiLSTM-Attention")["metrics"]["macro_f1"]))
    mac("ednetProBestoff", f"{100*bt['metrics']['one_off_accuracy']:.1f}\\%")
    mac("ednetProBestsev", f"{100*conditional_severe_under('ednet', 'prospective', best):.1f}\\%")
    mac("ednetProBestsevover", f"{100*conditional_severe_over('ednet', 'prospective', best):.1f}\\%")
    mac("ednetProLSTMsev", f"{100*conditional_severe_under('ednet', 'prospective', 'LSTM'):.1f}\\%")
    mac("ednetProLSTMsevover", f"{100*conditional_severe_over('ednet', 'prospective', 'LSTM'):.1f}\\%")

    # neural hyperparameter-sensitivity macros (robustness check)
    if os.path.exists(os.path.join(RES, "hp_sweep.json")):
        with open(os.path.join(RES, "hp_sweep.json")) as fh:
            hp = json.load(fh)
        for ds, tag, res in (("oulad", "OULAD", ou), ("ednet", "EdNet", ed)):
            rows = [dict(r, model=name) for name, blk in hp[ds].items() for r in blk["rows"]]
            bestrow = max(rows, key=lambda r: r["val_macro_f1"])
            mac(f"tuned{tag}model", DISP.get(bestrow["model"], bestrow["model"]))
            mac(f"tuned{tag}Fone", fmt(bestrow["macro_f1"]))
            mac(f"tuned{tag}qwk", fmt(bestrow["qwk"]))
            ap = get(res, ds, "prospective", bestrow["model"])["metrics"]
            mac(f"apriori{tag}Fone", fmt(ap["macro_f1"]))
            mac(f"apriori{tag}qwk", fmt(ap["qwk"]))

    lines = ["%% AUTO-GENERATED by experiments/build_paper_artifacts.py — do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in n.items()]
    with open(os.path.join(MAN, "numbers.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"numbers.tex: {len(n)} macros")


# ── tables ───────────────────────────────────────────────────────────────────
def build_tables(ou, ed):
    # headline leakage table
    rows = []
    for m in MODELS:
        cells = [DISP.get(m, m)]
        for res, ds in ((ou, "oulad"), (ed, "ednet")):
            pro, con = get(res, ds, "prospective", m)["metrics"], get(res, ds, "contemporaneous", m)["metrics"]
            cells += [fmt(pro["accuracy"]), fmt(con["accuracy"]), f"$+{100*(con['accuracy']-pro['accuracy']):.1f}$",
                      fmt(pro["qwk"]), fmt(con["qwk"]), f"$+{con['qwk']-pro['qwk']:.2f}$"]
        rows.append(" & ".join(cells) + r" \\")
    table = r"""% AUTO-GENERATED by experiments/build_paper_artifacts.py
\begin{table}[ht]
\caption{The cost of a contemporaneous feature window: identical models, targets, and protocol, differing only in whether the predictor window covers the labelled period. $\Delta$ columns are the contemporaneous-minus-prospective gap.\label{tab:leakage}}
\centering\footnotesize
\setlength{\tabcolsep}{2.8pt}
\begin{tabular}{@{}l ccc ccc ccc ccc@{}}
\toprule
& \multicolumn{6}{c}{\textbf{OULAD}} & \multicolumn{6}{c}{\textbf{EdNet-KT1}} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-13}
& \multicolumn{3}{c}{Accuracy} & \multicolumn{3}{c}{QWK} & \multicolumn{3}{c}{Accuracy} & \multicolumn{3}{c}{QWK} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
\textbf{Model} & Pros. & Cont. & $\Delta$ & Pros. & Cont. & $\Delta$ & Pros. & Cont. & $\Delta$ & Pros. & Cont. & $\Delta$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(MAN, "table_leakage.tex"), "w").write(table)

    # per-dataset prospective tables
    template = r"""% AUTO-GENERATED by experiments/build_paper_artifacts.py
\begin{table}[ht]
\caption{__CAPTION__
\label{tab:__LABEL__}}
\centering\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrr}
\toprule
\textbf{Model} & \textbf{Accuracy [CI]} & \textbf{Macro-F1 [CI]} & \textbf{MAE} & \textbf{QWK [CI]} & \textbf{1-off} & \textbf{Sev.\ und.} \\
\midrule
__ROWS__
\bottomrule
\end{tabular*}
__FOOT__
\end{table}
"""
    for res, ds, label, caption, foot in (
        (ou, "oulad", "oulad_prospective",
         r"""Prospective-protocol performance on OULAD (test split; bootstrap 95\% CIs resample students). Sev.\ und.\ is the share of severe cases (levels 2--3) predicted as low risk (levels 0--1).""",
         r"""Test split: \nOULADenrol-enrolment equivalents from \nOULADstudents{} students."""),  # replaced below
        (ed, "ednet", "ednet_prospective",
         r"""Prospective-protocol performance on EdNet-KT1 (test split; bootstrap 95\% CIs resample users).""",
         None),
    ):
        rows = []
        for m in MODELS:
            r = get(res, ds, "prospective", m)
            mt, ci = r["metrics"], r["bootstrap_ci"]
            sev = 100 * conditional_severe_under(ds, "prospective", m)
            name = DISP.get(m, m)
            rows.append(" & ".join([
                name, f"{fmt(mt['accuracy'])} {ci_str(ci['accuracy'])}",
                f"{fmt(mt['macro_f1'])} {ci_str(ci['macro_f1'])}",
                fmt(mt["mae"]), f"{fmt(mt['qwk'])} {ci_str(ci['qwk'])}",
                fmt(mt["one_off_accuracy"]), f"{sev:.1f}\\%",
            ]) + r" \\")
        if ds == "oulad":
            pv = pd.read_csv(os.path.join(RES, f"preds_{ds}_prospective_XGBoost.csv"))
            n_enrol, n_stu = len(pv), pv.student.nunique()
            foot = (r"\footnotetext{Test split: " + f"{n_enrol:,}".replace(",", "{,}") +
                    r"{} enrolments from " + f"{n_stu:,}".replace(",", "{,}") +
                    r"{} distinct students; preprocessing fitted on the training split only; "
                    r"severe underestimation conditioned on true levels 2--3.}")
        else:
            n_te = int(round(0.15 * ed["summary"]["users"]))
            foot = (r"\footnotetext{Test split: $\approx$" + f"{n_te:,}".replace(",", "{,}") +
                    r"{} users; preprocessing fitted on the training split only; severe "
                    r"underestimation conditioned on true levels 2--3.}")
        tbl = (template.replace("__CAPTION__", caption).replace("__LABEL__", label)
                       .replace("__ROWS__", "\n".join(rows)).replace("__FOOT__", foot))
        open(os.path.join(MAN, f"table_{label}.tex"), "w").write(tbl)
    print("tables written")


# ── figure 1: architecture ───────────────────────────────────────────────────
def fig_architecture(ou):
    fig, ax = plt.subplots(figsize=(9.2, 3.1))
    ax.set_xlim(0, 100); ax.set_ylim(-2.4, 34); ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc="#eef3fb", ec=BLUE, fs=8, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    fc=fc, ec=ec, lw=1.1,
                                    linestyle="--" if dashed else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color=GRAY, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, color=color, lw=1.1,
                                     linestyle="--" if dashed else "-"))

    sources = ["AMS\ngrades / credits", "LMS\ninteraction traces",
               "SAS\ndemographics / attendance", "IUCP\nworkplace learning"]
    y0 = 24
    for i, s in enumerate(sources):
        yy = y0 - i * 7.4
        box(1, yy, 15, 6.2, s, fc="#f4f6f8", ec=GRAY)
        arrow(16.4, yy + 3.1, 22.5, 17, color=GRAY)
    box(23, 13, 16, 8, "per-source\nextractors\n(student, period) key")
    box(44, 13, 14, 8, "partition-parallel\nmerge", fc="#eef3fb")
    box(63, 13, 15, 8, "tensor constructor\n$X(\\tau)$ pre-cutoff\n$Y$ post-cutoff", fc="#eef3fb")
    n_par = get(ou, "oulad", "prospective", "BiLSTM-Attention")["params"]
    box(83, 20, 15, 7.2, f"BiLSTM-Attention\n({n_par/1000:.0f}K params)", fc="#f7f2fb", ec=VIOLET)
    box(83, 5.5, 15, 7.2, "ordinal metrics +\nstudent bootstrap", fc="#f7f2fb", ec=VIOLET)
    arrow(58.5, 17, 62.5, 17)
    arrow(39.5, 17, 43.5, 17)
    arrow(78.5, 17, 82.5, 21.5)
    arrow(90.5, 19.5, 90.5, 13.2, color=VIOLET)
    box(63, 1.5, 15, 6.4, "prospective cut gate\n$\\tau=0.3\\,|W|$ / $0.5\\,|W|$", fc="#fdf3f3", ec=RED, dashed=True)
    arrow(70.5, 12.5, 70.5, 8.2, color=RED, dashed=True)
    ax.text(70.5, -1.6, "label-defining signals join evaluation only, never training",
            ha="center", fontsize=7.5, color=RED)
    ax.text(50, 32.6, "multi-source fusion with an explicit prospective cut",
            ha="center", fontsize=10.5)
    fig.savefig(os.path.join(FIG, "fig1_architecture.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure 2: leakage gap ────────────────────────────────────────────────────
def fig_leakage(ou, ed):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 4.6), sharex=False)
    short = ["LR", "RF", "GBM", "XGB", "LSTM", "BiLSTM-Att"]
    for col, (res, ds, key) in enumerate(((ou, "OULAD", "oulad"), (ed, "EdNet-KT1", "ednet"))):
        dacc, dqwk = [], []
        for m in MODELS:
            pro, con = get(res, key, "prospective", m)["metrics"], \
                       get(res, key, "contemporaneous", m)["metrics"]
            dacc.append(100 * (con["accuracy"] - pro["accuracy"]))
            dqwk.append(con["qwk"] - pro["qwk"])
        for row, (vals, color, ylab) in enumerate(
                ((dacc, BLUE, r"$\Delta$ Accuracy (pp)"), (dqwk, VIOLET, r"$\Delta$ QWK"))):
            ax = axes[row, col]
            x = np.arange(len(MODELS))
            ax.bar(x, vals, width=0.62, color=color, zorder=3)
            ax.set_xticks(x, short, rotation=20)
            ax.set_ylabel(ylab)
            ax.set_title(ds if row == 0 else None)
            for xi, v in zip(x, vals):
                ax.text(xi, v + (0.15 if row == 0 else 0.004), f"{v:.1f}" if row == 0 else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, color="#333")
            ax.set_ylim(0, max(vals) * 1.22)
    fig.suptitle("Inflation from a contemporaneous feature window (contemporaneous $-$ prospective)",
                 y=1.0, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_leakage_gap.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure 3: ordinal error structure ────────────────────────────────────────
def fig_ordinal(ou, ed):
    best = {ds: max(TREE, key=lambda m: get(res, ds, "prospective", m)["metrics"]["macro_f1"])
            for ds, res in (("oulad", ou), ("ednet", ed))}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, (ds, res, title) in zip(axes[:2], (("oulad", ou, "OULAD"), ("ednet", ed, "EdNet-KT1"))):
        df = pd.read_csv(os.path.join(RES, f"preds_{ds}_prospective_{best[ds]}.csv"))
        err = (df.y_pred - df.y_true).abs().value_counts().reindex([0, 1, 2, 3], fill_value=0)
        pct = 100 * err / err.sum()
        ax.bar([str(i) for i in err.index], pct, width=0.6, color=BLUE, zorder=3)
        ax.set_title(f"{title} — {DISP.get(best[ds], best[ds])}")
        ax.set_xlabel(r"ordinal distance $|\hat{y}-y|$")
        ax.set_ylabel("% of test cases")
        for i, v in enumerate(pct):
            ax.text(i, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(100, pct.max() * 1.15))
    ax = axes[2]
    labels, vals, colors = [], [], []
    for ds, tag in (("oulad", "OULAD"), ("ednet", "EdNet")):
        res = ou if ds == "oulad" else ed
        b = best[ds]
        for regime in ("prospective", "contemporaneous"):
            labels.append(f"{tag}\n{regime[:4]}.")
            vals.append(100 * conditional_severe_under(ds, regime, b))
            colors.append(BLUE if regime == "prospective" else VIOLET)
    ax.bar(range(len(vals)), vals, width=0.62, color=colors, zorder=3)
    ax.set_xticks(range(len(vals)), labels, fontsize=7.5)
    ax.set_ylabel("severe underestimation (% of levels 2--3)")
    ax.set_title("Severe cases predicted as low risk")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_ordinal_structure.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure 4: confusion matrices ─────────────────────────────────────────────
def fig_confusion(ou, ed):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, (ds, res, title) in zip(axes, (("oulad", ou, "OULAD"), ("ednet", ed, "EdNet-KT1"))):
        best = max(TREE, key=lambda m: get(res, ds, "prospective", m)["metrics"]["macro_f1"])
        cm = np.array(get(res, ds, "prospective", best)["confusion"], dtype=float)
        row = cm / cm.sum(axis=1, keepdims=True)
        ax.grid(False)
        im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                c = "white" if row[i, j] > 0.55 else "#333"
                ax.text(j, i, f"{100*row[i,j]:.0f}%\n({int(cm[i,j]):,})".replace(",", ","),
                        ha="center", va="center", fontsize=6.8, color=c)
        ax.set_xticks(range(4), [f"L{k}" for k in range(4)])
        ax.set_yticks(range(4), [f"L{k}" for k in range(4)])
        ax.set_xlabel("predicted level"); ax.set_ylabel("true level")
        ax.set_title(f"{title} — {DISP.get(best, best)}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_confusion.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure 5: SHAP ───────────────────────────────────────────────────────────
def fig_shap():
    with open(os.path.join(RES, "shap_oulad.json")) as fh:
        sh = json.load(fh)
    feats = sh["features"][:15][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = [f["name"] for f in feats]
    vals = [f["mean_abs"] for f in feats]
    ax.barh(range(len(vals)), vals, height=0.62, color=BLUE, zorder=3)
    ax.set_yticks(range(len(vals)), names, fontsize=7.5)
    ax.set_xlabel(r"mean $|$SHAP$|$ (avg.\ over classes)")
    ax.set_title(f"Feature attribution — {sh['model']}, OULAD prospective")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, max(vals) * 1.14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_shap.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure 6: attention profile ──────────────────────────────────────────────
def fig_attention():
    alpha = np.load(os.path.join(RES, "attention_oulad_prospective.npy"))
    preds = pd.read_csv(os.path.join(RES, "preds_oulad_prospective_BiLSTM-Attention.csv"))
    if len(preds) != len(alpha):
        preds = preds.iloc[:len(alpha)]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    mat = np.zeros((4, alpha.shape[1]))
    for k in range(4):
        m = preds.y_pred.to_numpy() == k
        mat[k] = alpha[m].mean(axis=0) if m.any() else 0.0
    ax.grid(False)
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(mat.shape[1]), [f"bin {i+1}" for i in range(mat.shape[1])])
    ax.set_yticks(range(4), [f"pred. L{k}" for k in range(4)])
    ax.set_xlabel("pre-cutoff activity bins (1 = earliest, 8 = latest)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            c = "white" if mat[i, j] > 0.55 * mat.max() else "#333"
            ax.text(j, i, f"{100*mat[i,j]:.0f}", ha="center", va="center", fontsize=7, color=c)
    ax.set_title("Mean attention weights by predicted severity (BiLSTM-Attention, OULAD)")
    fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02, label="mean weight (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_attention.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    ou, ed = load("oulad"), load("ednet")
    build_numbers(ou, ed)
    build_tables(ou, ed)
    fig_architecture(ou)
    fig_leakage(ou, ed)
    fig_ordinal(ou, ed)
    fig_confusion(ou, ed)
    if os.path.exists(os.path.join(RES, "shap_oulad.json")):
        fig_shap()
    else:
        print("skip fig5 (no shap_oulad.json yet)")
    if os.path.exists(os.path.join(RES, "attention_oulad_prospective.npy")):
        fig_attention()
    else:
        print("skip fig6 (no attention array yet)")
    print("all artifacts built")


if __name__ == "__main__":
    main()
