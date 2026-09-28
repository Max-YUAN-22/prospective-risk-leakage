# ── figure: paired slopes (formulation effect magnitude) ────────────────────
def fig_leakage_slopes(ou, ed):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.2))
    metrics = [("accuracy", "Accuracy"), ("qwk", "Quadratic weighted kappa")]
    colors = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
    for col, (res, key, name) in enumerate(((ou, "oulad", "OULAD"), (ed, "ednet", "EdNet-KT1"))):
        for row, (metric, mlab) in enumerate(metrics):
            ax = axes[row, col]
            for i, mdl in enumerate(MODELS):
                pro = get(res, key, "prospective", mdl)["metrics"][metric]
                con = get(res, key, "contemporaneous", mdl)["metrics"][metric]
                a0, a1 = (pro, con) if metric == "qwk" else (100 * pro, 100 * con)
                ax.plot([0, 1], [a0, a1], "-o", color=colors[i], lw=1.4, ms=4.5,
                        label=DISP.get(mdl, mdl), zorder=3)
            ax.set_xlim(-0.25, 1.25)
            ax.set_xticks([0, 1], ["prospective", "contemporaneous"])
            ax.set_ylabel(mlab)
            if row == 0:
                ax.set_title(name)
            if col == 1 and row == 1:
                ax.legend(loc="lower right", fontsize=6.5, frameon=False)
    fig.suptitle("Every model moves up when the window covers the labelled period",
                 y=0.99, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_leakage_gap.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure: dose-response on both benchmarks ────────────────────────────────
def fig_dose_response(ou, ed):
    with open(os.path.join(RES, "ednet_dose.json")) as fh:
        edn = json.load(fh)
    with open(os.path.join(RES, "dose_response.json")) as fh:
        ould = json.load(fh)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 4.9))
    series = ["XGBoost", "BiLSTM-Attention"]
    colors = {"XGBoost": "#2a78d6", "BiLSTM-Attention": "#008300"}
    plans = ((ould, [0.30, 0.50, 0.70, 0.90, 1.00], "OULAD", "feature window (% of course)"),
             (edn, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "EdNet-KT1", "% of sequence observed"))
    for col, (data, ps, ds, xlab) in enumerate(plans):
        for row, (metric, ylab) in enumerate((("accuracy", "Accuracy"),
                                              ("qwk", "Quadratic weighted $\\kappa$"))):
            ax = axes[row, col]
            for mdl in series:
                ys = []
                for p in ps:
                    key = f"{p:.2f}" if f"{p:.2f}" in data else f"{p:.1f}"
                    ys.append(data[key][mdl][metric])
                ax.plot([100 * q for q in ps], ys, "-o", color=colors[mdl], lw=1.6,
                        ms=4.5, label=DISP.get(mdl, mdl), zorder=3)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(ds)
            ax.set_xticks([100 * q for q in ps])
            if col == 1:
                ax.axvspan(50, 100, color="#E8760E", alpha=0.06, zorder=1)
                lo, hi = ax.get_ylim()
                ax.annotate("outcome-overlapping", xy=(75, lo + 0.05 * (hi - lo)),
                            fontsize=6.5, color="#b56312", ha="center")
            if col == 0:
                ax.axvline(90, color=GRAY, lw=0.8, ls=":", zorder=1)
                lo, hi = ax.get_ylim()
                ax.annotate("outcome-determining", xy=(100, lo + 0.05 * (hi - lo)),
                            fontsize=6.5, color="#b56312", ha="right")
        axes[0, col].legend(loc="lower right", fontsize=6.5, frameon=False)
    fig.suptitle("Reported performance rises with the information window on both benchmarks",
                 y=0.99, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_dose_response.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure: decomposition waterfall + EdNet reconstruction regime ───────────
def fig_decomposition(ou, ed):
    with open(os.path.join(RES, "matched_leakage.json")) as fh:
        mk = json.load(fh)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for i, (mdl, color) in enumerate((("XGBoost", BLUE), ("BiLSTM-Attention", "#008300"))):
        b0 = mk["results"]["clean30"][mdl]["accuracy"]
        ext = mk["summary"][mdl]["window_extension_acc"]
        leak = mk["summary"][mdl]["leakage_step_acc"]
        b1, b2 = b0 + ext, b0 + ext + leak
        xs = [0.15 + i * 0.5, 0.15 + i * 0.5, 0.15 + i * 0.5]
        for x, v, al in zip(xs, (b0, b1, b2), (0.45, 0.75, 1.0)):
            ax.bar(x, v, width=0.16, color=color, alpha=al, zorder=3)
            ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.6)
        ax.plot([xs[0] + 0.08, xs[1] - 0.08], [b0, b0], color=GRAY, lw=0.8, ls=":")
        ax.plot([xs[1] + 0.08, xs[2] - 0.08], [b1, b1], color=GRAY, lw=0.8, ls=":")
        ax.annotate(f"+{100*ext:.1f}", xy=(xs[1] - 0.08, b0 + 0.02), fontsize=6.6,
                    color="#275d9e", ha="center")
        ax.annotate(f"+{100*leak:.1f}", xy=(xs[2] - 0.08, b1 + 0.02), fontsize=6.6,
                    color="#275d9e", ha="center")
    ax.set_xticks([0.15, 0.65], ["XGBoost", "BiLSTM-Attention"])
    ax.set_ylim(0.58, 0.72)
    ax.set_ylabel("OULAD accuracy")
    ax.set_title("window extension vs. label-defining injection", fontsize=8.5)
    ax = axes[1]
    pro = get(ed, "ednet", "prospective", "XGBoost")["metrics"]["accuracy"]
    con = get(ed, "ednet", "contemporaneous", "XGBoost")["metrics"]["accuracy"]
    ax.bar([0, 1], [pro, con], width=0.5, color=[BLUE, "#E8760E"], zorder=3)
    ax.set_xticks([0, 1], ["prospective\n(p=0.5)", "full sequence\n(label-determining)"])
    for x, v in zip((0, 1), (pro, con)):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
    ax.annotate("", xy=(1, con - 0.012), xytext=(1, pro + 0.06),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    ax.text(1.07, (pro + con) / 2, "label\nreconstruction", fontsize=6.8, color=RED,
            va="center")
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("XGBoost accuracy (EdNet-KT1)")
    ax.set_title("the reconstruction regime", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_decomposition.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure: future-cohort paired slopes ─────────────────────────────────────
def fig_temporal():
    with open(os.path.join(RES, "temporal_generalization.json")) as fh:
        tg = json.load(fh)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    colors = {"XGBoost": "#2a78d6", "LSTM": "#eda100", "BiLSTM-Attention": "#008300"}
    for mdl, color in colors.items():
        pro = tg["results"]["prospective"][mdl]["accuracy"]
        con = tg["results"]["contemporaneous"][mdl]["accuracy"]
        ax.plot([0, 1], [100 * pro, 100 * con], "-o", color=color, lw=1.6, ms=5,
                label=DISP.get(mdl, mdl), zorder=3)
        ax.text(-0.04, 100 * pro, f"{100*pro:.1f}", ha="right", va="center", fontsize=7)
        ax.text(1.04, 100 * con, f"{100*con:.1f}", ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.3, 1.3)
    ax.set_xticks([0, 1], ["prospective\n(p=0.3)", "contemporaneous\n(full course)"])
    ax.set_ylabel("accuracy, 2014 presentations (%)")
    ax.set_title("trained on 2013 cohorts, tested on 2014")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_temporal.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── figure: prospective model comparison, dot + CI ──────────────────────────
def fig_model_comparison():
    sup = pd.read_csv(os.path.join(RES, "supplementary_cis.csv"))
    sup = sup[(sup.regime == "prospective") & (sup.model.isin(MODELS))]
    order = {m: i for i, m in enumerate(MODELS)}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0))
    for ax, ds, title in ((axes[0], "oulad", "OULAD"), (axes[1], "ednet", "EdNet-KT1")):
        sub = sup[sup.dataset == ds].copy()
        sub["ord"] = sub.model.map(order)
        sub = sub.sort_values("ord", ascending=False)
        ax.errorbar(sub.macro_f1, range(len(sub)),
                    xerr=[sub.macro_f1 - sub.f1_lo, sub.f1_hi - sub.macro_f1],
                    fmt="o", color=BLUE, lw=1.4, ms=5, capsize=2.5, zorder=3)
        ax.set_yticks(range(len(sub)), [DISP.get(m, m) for m in sub.model], fontsize=7.5)
        ax.set_xlabel("macro-F1, prospective protocol (95% bootstrap CI)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.suptitle("No family dominates across benchmarks and metrics", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_model_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)


# ── table: EdNet dose ───────────────────────────────────────────────────────
def table_ednet_dose():
    with open(os.path.join(RES, "ednet_dose.json")) as fh:
        edn = json.load(fh)
    rows = []
    for p in sorted(edn, key=float):
        pf = float(p)
        label = {0.5: "prospective cut (tau)", 1.0: "full sequence (label-determining)"}.get(pf, "")
        cells = [label if label else f"$p = {pf:.1f}$"]
        for mdl in ("XGBoost", "BiLSTM-Attention"):
            mt = edn[p][mdl]
            cells += [fmt(mt["accuracy"]), fmt(mt["qwk"])]
        rows.append(" & ".join(cells) + r" \\")
    tbl = (
        "% AUTO-GENERATED by experiments/build_paper_artifacts.py\n"
        "\\begin{table}[ht]\n"
        "\\caption{EdNet-KT1 contamination dose--response: the prospective label is fixed "
        "(second-half correctness) while the feature window grows from the mid-sequence cut into "
        "the full sequence, whose correctness determines the label. Accuracy approaches the "
        "reconstruction ceiling as the window becomes outcome-overlapping.\\label{tab:ednet_dose}}\n"
        "\\centering\\footnotesize\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llrrrr}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{\\textbf{XGBoost}} & \\multicolumn{2}{c}{\\textbf{BiLSTM-Attention}} \\\\\n"
        "\\cmidrule(l){3-4}\\cmidrule(l){5-6}\n"
        "\\textbf{Window} & & \\textbf{Accuracy} & \\textbf{QWK} & \\textbf{Accuracy} & \\textbf{QWK} \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    open(os.path.join(MAN, "table_ednet_dose.tex"), "w").write(tbl)
