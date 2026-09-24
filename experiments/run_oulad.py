"""OULAD leakage-controlled experiment.

Two feature regimes, identical models/splits/labels:
  prospective     -- predictors restricted to the first 30% of each course
                     presentation (VLE activity + assessments due in that
                     window + enrolment demographics). The outcome
                     (final_result) is observed only at the end of the course,
                     i.e. X_t -> Y_{t+1}.
  contemporaneous -- predictors use the full course window including final-exam
                     scores, i.e. the label-defining signals themselves. This
                     mirrors regimes still common in the published OULAD
                     literature and serves as the label-reconstruction reference.

Split: stratified at the STUDENT level (70/15/15) so no student appears in
more than one split; OULAD students re-take courses (32,593 enrollments vs
28,785 students), so an observation-level split would leak.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (evaluate_run, fit_sequence_model, make_flat_models, param_count,
                    predict_sequence, save_results, seed_everything)

DATA = "/Users/Apple/Desktop/灵感投稿/教育会议/EI/03_Data/anonymisedData"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
N_BINS = 8
LABEL_MAP = {"Distinction": 0, "Pass": 1, "Fail": 2, "Withdrawn": 3}
CAT_COLS = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
NUM_COLS = ["num_of_prev_attempts", "studied_credits"]
EARLY_VLE = ["clicks_early", "days_early", "meanday_early", "stdday_early"]
FULL_VLE = ["clicks_full", "days_full", "meanday_full", "stdday_full"]
EARLY_ASX = ["n_submitted_early", "mean_score_early", "min_score_early", "any_missing_early"]
FULL_ASX = ["n_submitted_full", "mean_score_full", "min_score_full", "max_score_full", "exam_score"]


def load_enrolments():
    si = pd.read_csv(f"{DATA}/studentInfo.csv")
    courses = pd.read_csv(f"{DATA}/courses.csv")
    df = si.merge(courses, on=["code_module", "code_presentation"], how="left")
    df["label"] = df.final_result.map(LABEL_MAP)
    df["cutoff_day"] = (0.30 * df.module_presentation_length).round().astype(int)
    df["enrol_id"] = np.arange(len(df))
    return df


def student_split(df):
    """70/15/15 stratified by each student's most severe observed label."""
    from sklearn.model_selection import train_test_split
    worst = df.groupby("id_student").label.max().reset_index()
    tr_s, tmp_s = train_test_split(worst, test_size=0.30, random_state=42,
                                   stratify=worst.label)
    va_s, te_s = train_test_split(tmp_s, test_size=0.50, random_state=42,
                                  stratify=tmp_s.label)
    split_of = {}
    for name, frame in (("train", tr_s), ("val", va_s), ("test", te_s)):
        split_of.update(dict.fromkeys(frame.id_student, name))
    df["split"] = df.id_student.map(split_of)
    assert not (set(df[df.split == "train"].id_student) &
                set(df[df.split == "test"].id_student))
    return df


def vle_features(df):
    """Aggregated flat features + (N, N_BINS, 2) activity-sequence tensors over
    the EARLY window and over the FULL course window (tokens = equal-width time
    bins; token features = log1p clicks and active days per bin)."""
    sv = pd.read_csv(f"{DATA}/studentVle.csv")
    sv = sv.merge(df[["code_module", "code_presentation", "id_student", "enrol_id",
                      "cutoff_day", "module_presentation_length"]],
                  on=["code_module", "code_presentation", "id_student"], how="inner")

    def bin_tensor(g, hi):
        tok = np.zeros((N_BINS, 2))
        w = g[(g.date >= 0) & (g.date <= hi)]
        if not len(w):
            return tok
        edges = np.linspace(0, max(hi, 1), N_BINS + 1)
        b = np.clip(np.digitize(w.date.to_numpy(), edges[1:-1]), 0, N_BINS - 1)
        tb = pd.DataFrame({"bin": b, "date": w.date.to_numpy(),
                           "clicks": w.sum_click.to_numpy()})
        daily_b = tb.groupby(["bin", "date"]).clicks.sum().reset_index()
        for k in range(N_BINS):
            dk = daily_b[daily_b.bin == k]
            tok[k, 0] = np.log1p(dk.clicks.sum())
            tok[k, 1] = float(len(dk))
        return tok

    rows_flat, seq_e, seq_f, seq_ids = [], [], [], []
    for enrol, g in sv.groupby("enrol_id", sort=False):
        cutoff = int(g.cutoff_day.iloc[0])
        length = int(g.module_presentation_length.iloc[0])
        flat = {"enrol_id": enrol}
        for tag, hi in (("early", cutoff), ("full", length)):
            w = g[(g.date >= 0) & (g.date <= hi)]
            daily = w.groupby("date").sum_click.sum()
            flat[f"clicks_{tag}"] = float(w.sum_click.sum())
            flat[f"days_{tag}"] = int(daily.size)
            flat[f"meanday_{tag}"] = float(daily.mean()) if len(daily) else 0.0
            flat[f"stdday_{tag}"] = float(daily.std()) if len(daily) > 1 else 0.0
        rows_flat.append(flat)
        seq_e.append(bin_tensor(g, cutoff))
        seq_f.append(bin_tensor(g, length))
        seq_ids.append(enrol)
    flat = df[["enrol_id"]].merge(pd.DataFrame(rows_flat), on="enrol_id", how="left")
    flat[EARLY_VLE + FULL_VLE] = flat[EARLY_VLE + FULL_VLE].fillna(0.0)
    return flat, np.stack(seq_e), np.stack(seq_f), seq_ids


def assessment_features(df):
    am = pd.read_csv(f"{DATA}/assessments.csv").dropna(subset=["date"])
    sa = pd.read_csv(f"{DATA}/studentAssessment.csv")
    asx = sa.merge(am, on="id_assessment", how="inner").merge(
        df[["code_module", "code_presentation", "id_student", "enrol_id",
            "cutoff_day", "module_presentation_length"]],
        on=["code_module", "code_presentation", "id_student"], how="inner")
    out = []
    for enrol, g in asx.groupby("enrol_id", sort=False):
        cutoff = int(g.cutoff_day.iloc[0])
        length = int(g.module_presentation_length.iloc[0])
        # strict availability: a score exists at time t only if its recorded
        # submission date is no later than t.
        early = g[(g.date <= cutoff) & (g.date_submitted <= cutoff)]
        full = g[g.date_submitted <= length]
        exams = full[full.assessment_type == "Exam"]
        r = {"enrol_id": enrol,
             "n_submitted_early": len(early),
             "mean_score_early": early.score.mean() if len(early) else np.nan,
             "min_score_early": early.score.min() if len(early) else np.nan,
             "any_missing_early": float(len(early) < 1),
             "n_submitted_full": len(full),
             "mean_score_full": full.score.mean(),
             "min_score_full": full.score.min(),
             "max_score_full": full.score.max(),
             "exam_score": float(exams.score.iloc[0]) if len(exams) else np.nan}
        out.append(r)
    asx = df[["enrol_id"]].merge(pd.DataFrame(out), on="enrol_id", how="left")
    for c in ("n_submitted_early", "any_missing_early", "n_submitted_full"):
        asx[c] = asx[c].fillna(0.0)
    return asx


def preprocess(X_tr, X_va, X_te):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    cats = [c for c in X_tr.columns if c in CAT_COLS]
    nums = [c for c in X_tr.columns if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), nums),
    ])
    return pre.fit_transform(X_tr), pre.transform(X_va), pre.transform(X_te), pre


def main():
    seed_everything()
    df = load_enrolments()
    df = student_split(df)
    print(f"enrolments={len(df):,} students={df.id_student.nunique():,}")
    print("split sizes:", df.split.value_counts().to_dict())

    vle_flat, seq_early, seq_full, seq_ids = vle_features(df)
    asx = assessment_features(df)
    demo = df[["enrol_id"] + CAT_COLS + NUM_COLS]

    reg_flat = {
        "prospective": demo.merge(vle_flat[["enrol_id"] + EARLY_VLE], on="enrol_id")
                           .merge(asx[["enrol_id"] + EARLY_ASX], on="enrol_id"),
        "contemporaneous": demo.merge(vle_flat[["enrol_id"] + EARLY_VLE + FULL_VLE], on="enrol_id")
                               .merge(asx[["enrol_id"] + EARLY_ASX + FULL_ASX], on="enrol_id"),
    }
    base = df[["enrol_id", "id_student", "label", "split"]]
    summary = {"enrolments": int(len(df)), "students": int(df.id_student.nunique()),
               "splits": df.split.value_counts().to_dict(),
               "label_distribution": {str(k): int(v) for k, v in
                                      df.label.value_counts().sort_index().items()},
               "n_features": {r: int(x.shape[1] - 1) for r, x in reg_flat.items()}}
    results = []

    for regime, Xreg in reg_flat.items():
        print(f"\n=== regime: {regime} ({summary['n_features'][regime]} features) ===")
        data = base.merge(Xreg, on="enrol_id")
        fea = [c for c in data.columns if c not in ("enrol_id", "id_student", "label", "split")]
        tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
        X_tr, X_va, X_te, _pre = preprocess(tr[fea], va[fea], te[fea])
        y_tr, y_va, y_te = tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy()
        st_te = te.id_student.to_numpy()

        for name, model in make_flat_models().items():
            model.fit(X_tr, y_tr)
            yhat = model.predict(X_te)
            results.append(evaluate_run(name, y_te, yhat, st_te, OUT, "oulad", regime))
            m = results[-1]["metrics"]
            print(f"  {name:18s} acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} qwk={m['qwk']:.3f}")

        # recurrent models consume the regime-matching activity sequence, with
        # the (already scaled) flat features appended to every time step so the
        # comparison with the flat models is information-fair.
        seq_tensor = seq_early if regime == "prospective" else seq_full
        seq_lookup = dict(zip(seq_ids, seq_tensor))
        zero = np.zeros((N_BINS, 2))
        dense = lambda m: np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m)
        concat = lambda ids, flatm: np.concatenate(
            [np.stack([seq_lookup.get(i, zero) for i in ids]), np.repeat(flatm, N_BINS, 0).reshape(len(ids), N_BINS, -1)],
            axis=2)
        Xtr_s = concat(tr.enrol_id.to_numpy(), dense(X_tr))
        Xva_s = concat(va.enrol_id.to_numpy(), dense(X_va))
        Xte_s = concat(te.enrol_id.to_numpy(), dense(X_te))
        # normalise the two token dimensions using training statistics only
        tok_mean = Xtr_s[:, :, :2].reshape(-1, 2).mean(0)
        tok_std = Xtr_s[:, :, :2].reshape(-1, 2).std(0)
        tok_std[tok_std == 0] = 1.0
        for A in (Xtr_s, Xva_s, Xte_s):
            A[:, :, 0] = (A[:, :, 0] - tok_mean[0]) / tok_std[0]
            A[:, :, 1] = (A[:, :, 1] - tok_mean[1]) / tok_std[1]
        for name in ("LSTM", "BiLSTM-Attention"):
            net = fit_sequence_model(name, Xtr_s, y_tr, Xva_s, y_va)
            yhat, _, alpha = predict_sequence(net, Xte_s)
            results.append(evaluate_run(name, y_te, yhat, st_te, OUT, "oulad", regime))
            res = results[-1]
            res["params"] = int(param_count(net))
            if alpha is not None:
                np.save(os.path.join(OUT, f"attention_oulad_{regime}.npy"), alpha)
                res["attention_mean"] = alpha.mean(axis=0).tolist()
            print(f"  {name:18s} acc={res['metrics']['accuracy']:.3f} "
                  f"f1={res['metrics']['macro_f1']:.3f} qwk={res['metrics']['qwk']:.3f} "
                  f"params={res['params']:,}")

    save_results(os.path.join(OUT, "oulad_results.json"),
                 {"summary": summary, "results": results})


if __name__ == "__main__":
    main()
