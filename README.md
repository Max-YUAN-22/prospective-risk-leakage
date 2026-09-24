# prospective-risk-leakage

Reproducibility package for the manuscript

> **From Contemporaneous to Prospective: Quantifying Target Leakage in
> Student Academic Risk Prediction with Lightweight Temporal Models**

submitted to the *International Journal of Data Science and Analytics*.

The repository regenerates **every number, table, and figure** in the paper
from two public datasets. No reported value is manually transcribed.

## What the paper shows

Holding model, target, split, and protocol fixed, moving from a
*contemporaneous* feature window (predictors cover the labelled period, i.e.
the label-defining signals are among the features) to a *prospective* window
($X_t \to Y_{t+1}$: prediction strictly precedes the labelled period) costs

| | OULAD | EdNet-KT1 |
|---|---|---|
| accuracy inflation (contemporaneous − prospective) | +9.3 … +14.6 pp | +39.3 … +46.6 pp |
| quadratic-weighted κ inflation | up to +0.25 | up to +0.58 |
| severe underestimation (prospective vs contemporaneous) | 12.3% vs 1.7% | 0.7% vs 0.0% |

— inflation larger than the spread between competing model families, across
six models from logistic regression to a lightweight BiLSTM-Attention
(≈71K parameters).

## Repository layout

```
├── experiments/
│   ├── common.py                 # metrics, student-level bootstrap, models
│   ├── run_oulad.py              # OULAD leakage-controlled experiment (2 regimes × 6 models)
│   ├── run_ednet.py              # EdNet-KT1 leakage-controlled experiment
│   ├── shap_analysis.py          # TreeExplainer attribution (OULAD prospective XGBoost)
│   └── build_paper_artifacts.py  # results/*.json → numbers.tex, tables, figures
└── results/                      # committed experiment outputs (JSON, per-model
                                  # test-set predictions, SHAP ranking, attention arrays)
```

## Data (download separately)

| Dataset | Source | Licence |
|---|---|---|
| OULAD | https://analyse.kmi.open.ac.uk/open_dataset (7 CSVs) | CC BY 4.0 |
| EdNet-KT1 | https://github.com/riiid/ednet (`KT1` interactions + `contents`) | CC BY-NC 4.0 |

Set the data paths at the top of `run_oulad.py` / `run_ednet.py` to your
download locations. No restricted institutional data is used anywhere in this
package; the manuscript's institutional deployment appears only as a design
context and is not part of any experiment.

## Reproduce

```bash
pip install -r requirements.txt

python experiments/run_oulad.py              # ~5 min on a laptop
python experiments/run_ednet.py              # ~15 min (reads sampled KT1 user files)
python experiments/shap_analysis.py          # ~4 min
python experiments/build_paper_artifacts.py  # regenerates all numbers/tables/figures
python experiments/run_sweep.py              # optional: neural hyperparameter-sensitivity check
python experiments/run_dose_response.py      # optional: leakage dose-response curve (fig 3 of the paper)
python experiments/run_label_ordering.py     # optional: label-ordering sensitivity check
python experiments/export_ednet_sample.py    # optional: re-pin the 24,736-user EdNet sample
```

The committed `results/hp_sweep.json` holds the 36-configuration neural
sensitivity check reported in the manuscript's limitations (selection on
validation macro-F1 only), and `results/ednet_sample_uids.csv` pins the exact
24,736-user EdNet-KT1 sample so the cohort is reproducible byte-for-byte.
`results/dose_response.json` (leakage dose-response, fig 3) and
`results/ordering_sensitivity.json` (alternative label ordering) support the
robustness analyses in Sections 6.2 and 8.

Evaluation protocol (identical for every dataset, regime, and model):
student-level 70/15/15 splits (no student in two splits), preprocessing fitted
on the training split only, a-priori hyperparameters with validation-only
early stopping, single-shot test evaluation, ordinal metric suite (accuracy,
macro-F1, MAE, quadratic-weighted κ, one-level accuracy, severe-underestimation
rate) with 1,000-resample **student-level bootstrap** 95% CIs, random seed 42.

## Licence

Code: MIT. The datasets remain under their respective licences (see above).
