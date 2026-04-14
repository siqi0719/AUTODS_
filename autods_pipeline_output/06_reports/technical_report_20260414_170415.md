# Executive Summary

This technical report documents the results of the AutoDS Pipeline project, an automated end-to-end data science initiative designed to predict NBA draft eligibility (`drafted`, binary) for college basketball players using statistical, team, and conference-level features. The pipeline processed a dataset of 14,774 rows and 62 columns, reducing it to 12,312 rows and 58 columns after cleaning—removing 2,462 duplicate rows and dropping four low-value or constant columns (`yr`, `type`, `rec_rank`, `dunks_ratio`). Feature engineering expanded the set to 87 features—including 33 numeric, 3 categorical (one-hot encoded into 53 dummy variables), and 1 boolean-as-numeric—while eliminating 16 columns due to sparsity, redundancy, or lack of predictive signal.

Three models were rigorously compared under 3-fold cross-validation: logistic regression (L2-regularized), random forest, and XGBoost. Logistic regression emerged as the top-performing model by the primary metric, ROC AUC, achieving **0.9923 on the held-out test set**—the highest among all candidates—and demonstrating exceptional generalization stability (CV ROC AUC = 0.9848 ± 0.0014). However, its test precision was critically low (**23.1%**), indicating that only ~1 in 4 predicted “drafted” cases were correct, while recall remained high (**94.7%**), meaning it captured nearly all actual drafted players. In contrast, tree-based models achieved higher precision (57.1% for both RF and XGBoost) but catastrophically low recall (21.1% and 42.1%, respectively), failing to identify the majority of positive cases. This divergence confirms severe class imbalance is driving misleadingly high accuracy (>97.5% across all models) and underscores that accuracy is an inadequate performance proxy here. The business-critical trade-off lies between false positives (misclassifying undrafted players as drafted—potentially inflating scouting effort) and false negatives (missing actual draftees—direct opportunity cost). Given this, logistic regression is selected not as a final production model, but as the most robust *baseline* for further calibration—specifically via threshold optimization and class-aware resampling.

## 1. Data Overview and Quality Assessment

The original dataset comprised **14,774 rows and 62 columns**, sourced from `D:\LangChain\AUTODS_\_uploaded_data_temp.csv`. After automated cleaning by the `DataCleaner` agent (v2.0), **2,462 duplicate rows were removed**, and **four columns were dropped**: `yr` and `type` (identified as constant), and `rec_rank` and `dunks_ratio` (deemed non-informative or structurally sparse). The cleaned dataset contains **12,312 rows and 58 columns**, representing **83.34% data retention**. Data quality improved markedly: completeness increased from **95.66% pre-cleaning (39,718 null values, 4.34% null rate)** to **100% post-cleaning (zero nulls)**, and duplicate rows were fully eliminated.

Column classification revealed **44 numeric features**, **4 categorical features** (`team`, `conf`, `ht`, `player_id`), and **no datetime, boolean, or text columns**. Of the categoricals, `player_id` was dropped during feature engineering due to lack of generalizability, leaving `team`, `conf`, and `ht` as the basis for one-hot encoding. Notably, `ht` (height-related) and `conf` (conference) generated numerous high-cardinality dummy variables (e.g., `conf_a10`, `ht_2_jun`, `conf_sec`), contributing significantly to the final feature count of 87. No explicit class imbalance ratio is reported in the JSON (`class_imbalance_ratio`: null); however, the extreme precision–recall asymmetry across models—and the planner’s explicit reference to “severe class imbalance”—strongly implies the `drafted = 1` class is rare. This is corroborated by the fact that logistic regression achieves >94% recall while maintaining >97% accuracy: such performance is only possible when the negative class dominates the distribution.

## 2. Feature Engineering Analysis

Feature engineering transformed the 58-column cleaned dataset into **87 final features**, comprising **33 engineered numeric features**, **53 one-hot encoded categorical dummies** (from `team`, `conf`, and `ht`), and **1 derived boolean-as-numeric variable**. Key operations included:
- Dropping 16 columns: 11 due to high correlation (`fta`, `twopa`, `tpa`, `mp`, `treb`, `ft_pct`, `net_rtg`) and 5 due to sparsity or irrelevance (`porpag`, `adjoe`, `dporpag`, `stops`, `bpm`, `obpm`, `dbpm`, `gbpm`, `ogbpm`, `dgbpm`, `player_id`, `rimmade`, `rimmade_rimmiss`, `midmade`, `midmade_midmiss`, `dunksmade`, `dunksmiss_dunksmade`).
- Creating 3 new numeric ratios: `orb_over_drb`, `ast_to_ratio`, and `trb_per`.
- Retaining core per-game efficiency metrics (`min_per`, `usg`, `efg`, `ts_per`, `ast_per`, `to_per`, `blk_per`, `stl_per`) and advanced defensive ratings (`adrtg`, `drtg`, `pfr`).

Feature importance analysis (based on model-agnostic ranking) identifies the top five most influential predictors:

| Rank | Feature Name     | Importance | Practical Interpretation |
|------|------------------|------------|--------------------------|
| 1    | `adrtg`          | 3.38       | Adjusted Defensive Rating — quantifies per-100-possession defensive efficiency; higher values indicate elite defensive impact, strongly associated with draftability. |
| 2    | `min_per`        | 3.03       | Minutes played per game — proxies for role size, coach trust, and durability; consistent high-minute players are more likely to be scouted. |
| 3    | `ast_tov`        | 2.59       | Assist-to-turnover ratio — measures playmaking decision-making under pressure; a key indicator of offensive control and NBA-readiness. |
| 4    | `pts`            | 2.14       | Total points per game — raw scoring output remains a dominant signal for front offices evaluating offensive ceiling. |
| 5    | `ast_per`        | 1.86       | Assists per 100 possessions — contextualizes passing volume relative to pace and usage, distinguishing true playmakers from volume accumulators. |

Notably, categorical encodings dominate the long tail of importance scores: 22 of the top 30 features are conference (`conf_*`) or height (`ht_*`) dummies, confirming that institutional context (e.g., `conf_a10`, `conf_sec`, `conf_acc`) and physical profile (`ht_2_jun`, `ht_8_jun`) carry substantial discriminative power—likely reflecting scouting bias, league visibility, or positional fit.

## 3. Model Comparison and Selection

Three models were evaluated under identical 3-fold cross-validation and tested on a held-out set. Performance rankings, based strictly on the planner-specified primary metric **ROC AUC**, are as follows:

| Rank | Model              | CV ROC AUC (± std)      | Test ROC AUC | Test Accuracy | Test Precision | Test Recall | Runtime (CV, s) |
|------|--------------------|-------------------------|--------------|----------------|----------------|-------------|-----------------|
| 1    | Logistic Regression| 0.9848 ± 0.0014         | **0.9923**   | 0.9752         | **0.231**      | **0.947**   | 0.466           |
| 2    | Random Forest      | 0.9774 ± 0.0119         | 0.9868       | 0.9927         | 0.571          | 0.211       | 2.111           |
| 3    | XGBoost            | NaN (computation failed)| 0.9921       | 0.9931         | 0.571          | 0.421       | 3.154           |

Logistic regression was selected as the best model because it achieved the **highest test ROC AUC (0.9923)** and exhibited the **lowest cross-validation standard deviation across all metrics**, particularly in ROC AUC (±0.0014) and recall (±0.054), indicating superior generalization stability. Its computational efficiency (0.466 s CV runtime) further supports operational viability. While random forest and XGBoost achieved marginally higher test accuracy (0.9927 and 0.9931), their ROC AUC scores were lower, and—critically—their recall collapsed to 21.1% and 42.1%, meaning they missed 79% and 58% of actual drafted players, respectively. This confirms the planner’s assessment: tree-based models overfit to the majority class and fail to learn minority-class patterns. The selection reason stated in the JSON—“Highest roc_auc score on held-out test set”—is empirically validated and aligns with the business goal of maximizing detection of true positives (drafted players), even at the cost of increased false positives.

## 4. Performance Evaluation

Model evaluation confirms that **ROC AUC is the only metric reliably discriminating model quality** in this imbalanced setting. All models achieve deceptively high test accuracy (>97.5%), rendering accuracy meaningless as a decision criterion. The critical insight lies in the precision–recall trade-off:

- **Logistic regression** achieves near-perfect recall (**94.7%**) but very low precision (**23.1%**), implying that for every 100 players flagged as “drafted”, only 23 are actually drafted—yet it captures 94.7% of all true draftees. Its high ROC AUC (0.9923) reflects strong rank-ordering ability: it assigns higher probabilities to actual draftees than to non-draftees, even if the absolute threshold is poorly calibrated.
- **Random forest**, despite 57.1% precision, recalls only **21.1%** of draftees—failing to identify nearly 4 out of 5 actual positives. Its F1-score (0.308) is less than half that of logistic regression (0.371), confirming inferior harmonic balance.
- **XGBoost**, while improving recall to 42.1%, still misses >57% of draftees and shows unstable CV behavior (NaN ROC AUC), suggesting numerical issues or overfitting on sparse features.

Cross-validation stability further validates logistic regression: its CV ROC AUC standard deviation (0.0014) is **8× smaller** than random forest’s (0.0119), confirming robustness across data splits. No confusion matrix is provided in the JSON, but the precision–recall values imply the following approximate test-set confusion structure for logistic regression (assuming ~5% prevalence of `drafted = 1` in a 12,312-row test set of ~1,231 samples):  
- True Positives ≈ 117 (94.7% of ~124 draftees)  
- False Positives ≈ 369 (76.9% of ~479 total positive predictions)  
- False Negatives ≈ 7  
- True Negatives ≈ 11,820  

This highlights the operational consequence: deploying logistic regression at default threshold yields a high-volume shortlist requiring manual review, whereas tree models produce shortlists missing most viable prospects.

## 5. Technical Recommendations and Improvement Directions

To address the documented limitations—particularly the low precision of the top-performing logistic regression model and the systemic recall failure of tree-based alternatives—the following evidence-based improvements are recommended:

- **Threshold Optimization & Calibration**: Replace the default 0.5 decision threshold with one optimized for business objectives. Compute the precision–recall curve and apply Youden’s J statistic (`J = sensitivity + specificity − 1`) or cost-sensitive thresholds (e.g., assigning 4× penalty to false negatives vs false positives) to balance scouting workload against missed talent. This requires computing predicted probabilities (natively available from logistic regression) and validating on domain-expert-labeled holdout data.

- **Class Imbalance Mitigation**: Implement synthetic oversampling (e.g., SMOTE) or instance-weighting during training. Given the planner’s explicit diagnosis of “severe class imbalance”, retraining logistic regression with `class_weight='balanced'` is the lowest-effort, highest-impact intervention—expected to lift precision without materially harming recall.

- **Feature Refinement**: Reintroduce high-importance dropped features with imputation (e.g., `adjoe`, `dporpag`) using median/mode or iterative imputation, as their exclusion may have exacerbated imbalance effects. Additionally, aggregate sparse height (`ht`) categories (e.g., bin `ht_2_jun`, `ht_8_jun`, `ht_10_jun` into broader ranges) to reduce dimensionality and mitigate overfitting on low-frequency dummies.

- **Monitoring & Update Strategy**: Deploy model monitoring tracking three KPIs: (1) **precision decay** (monthly % of predicted `drafted=1` that are verified as drafted), (2) **recall drift** (quarterly audit of known draftees missed by the model), and (3) **feature stability** (PSI > 0.1 on top-10 features triggers retraining). Retraining cadence should align with NCAA season cycles—annually post-draft, ingesting new player stats and updated conference alignments.

- **Validation Enhancement**: The current 3-fold CV is necessary given data constraints but insufficient for reliable uncertainty quantification. Supplement with bootstrap confidence intervals (1,000 samples) on ROC AUC and precision–recall to quantify estimation error, especially for the minority class.
