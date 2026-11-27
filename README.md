# CKD Thesis

Research prototype, not a clinical diagnostic device. Four models: Random Forest, SVM, XGBoost, and stacking with logistic regression meta-model. Raw CSV is preserved. Fixed 80/20 stratified split (seed 42); 5-fold repeated nested CV on training only; 3-fold grid search using ROC-AUC; stacking uses out-of-fold probabilities with tuning inside each internal fit. No artificial noise, label changes, test-set selection, or target accuracy manipulation. The nine screening variables are age, bp, bgr, htn, dm, cad, appet, pe, ane. This is a limited-information scenario, not proven prospective screening: timing of clinical measurements is unknown. Full 24-feature stacking comparison is performed on identical training CV folds (feature-set ablation). External validation is not available; prior analysis of the same dataset also means the holdout is not truly independent of the research history. High accuracy may reflect dataset characteristics. Results do not establish novelty or clinical utility.

## Windows PowerShell (Python 3.12 recommended)
```
py -3.12 -m venv thesis_env
.\thesis_env\Scripts\python.exe -m pip install -r requirements.txt
.\thesis_env\Scripts\python.exe -m pytest -q
.\thesis_env\Scripts\python.exe main.py --shap
```
Use `--repeats 1` for a faster exploratory run (default 2). `--shap` enables SHAP for the RF base estimator, not the full stack. Permutation importance explains the full stack. Do not use holdout explanations to revise the model and then claim the same holdout is unbiased. The McNemar tests are exploratory, Holm adjusted; exact binomial intervals assume independent observations. Repeated CV standard deviations are descriptive, not confidence intervals. No external patient IDs are available to establish patient independence. Do not load joblib files from untrusted sources. Retrain locally if library versions differ.

Outputs: models/ (joblib), results/metrics/ (test metrics, per-fold scores, summary, ablation, predictions, split, paired tests, permutation importance), results/figures/ (EDA, comparison, ROC, PR, calibration, confusion matrices, CV, ablation and explanations), results/reports/run.json (settings and best parameters). Preprocessing learns only from each training fold. Cleaned CSV is not imputed and is for inspection only. `main.py` is the entry point. No separate feature engineering or automatic feature selection is used; the predefined scenario is the feature selection. No guarantee of 88–92% accuracy.
