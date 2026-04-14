# AutoDS Report Package

## Technical Report

# Executive Summary

This technical report was generated with the built-in template mode. The project analyses **adult.csv** for **15** using a **classification** workflow. The selected model is **lightgbm**, and the primary metric **roc_auc** reached **0.914**.

## 1. Data Overview and Quality Assessment

- Dataset size: 32,561 rows and 15 columns.
- Rows after cleaning: 22,636.
- Class imbalance ratio: N/A.
- Data quality score: N/A.
- Quality notes: {"metadata": {"agent_name": "DataCleaner", "timestamp": "2026-04-02T17:09:47.314790", "version": "2.0"}, "input_data": {"rows": 32561, "columns": 15}, "output_data": {"rows": 22636, "columns": 15}, "cleaning_summary": {"rows_removed": 9925, "columns_dropped": [], "columns_dropped_cnt": 0, "data_retention_pct": 69.52}, "column_classification": {"id_columns": [], "numeric_columns": ["1", "5", "11", "12", "13"], "categorical_columns": ["2", "4", "6", "7", "8", "9", "10", "14", "15"], "special_format_columns": ["3"], "datetime_columns": [], "boolean_columns": [], "constant_columns": [], "dropped_columns": []}, "data_quality_metrics": {"before_cleaning": {"completeness_pct": 100.0, "null_count": 0, "null_pct": 0.0, "duplicate_rows": 24}, "after_cleaning": {"completeness_pct": 100.0, "null_count": 0, "null_pct": 0.0, "duplicate_rows": 0}}, "cleaning_operations": {"anomaly_strategy": "remove", "iqr_multiplier": 1.5, "llm_enabled": false}}.

- N/A

## 2. Feature Engineering Analysis

Features created: {"numeric": ["1", "3", "5", "11", "12", "13"], "categorical": ["2", "4", "6", "7", "8", "9", "10", "14"], "datetime": [], "text": [], "bool_as_numeric": []}.
Features dropped: {"general_drop": [], "constant_columns": [], "high_correlation_drop": []}.
Encoding or transformation steps: N/A.
Top feature signals:
- Unknown feature (2,054)
- Unknown feature (1,591)
- Unknown feature (733)
- Unknown feature (671)
- Unknown feature (600)

- N/A

## 3. Model Comparison and Selection

- Selected model: **lightgbm**.
- Selection reason: Highest roc_auc score on held-out test set..
- Optimization or selection strategy: N/A.
- Training time (seconds): N/A.

## 4. Performance Evaluation

| Metric | Value |
| --- | --- |
| Rank | 1 |
| Cv Runtime Seconds | 1.895 |
| Cv Accuracy | 0.86 |
| Cv Precision | 0.772 |
| Cv Recall | 0.665 |
| Cv F1 | 0.714 |
| Cv Roc Auc | 0.919 |
| Cv Std Accuracy | 0.004 |
| Cv Std Precision | 0.008 |
| Cv Std Recall | 0.012 |
| Cv Std F1 | 0.01 |
| Cv Std Roc Auc | 0.004 |
| Test Accuracy | 0.854 |
| Test Precision | 0.755 |
| Test Recall | 0.656 |
| Test F1 | 0.702 |
| Test Roc Auc | 0.914 |
| Training Error | nan |

| Confusion Matrix Item | Value |
| --- | --- |
| N/A | N/A |

- N/A

## 5. Technical Recommendations and Improvement Directions

### Improvement Recommendations

- Establish a monitoring plan for feature drift, score drift, and periodic model refresh.
- Collect additional explanatory variables to improve model robustness and actionability.

### Recommended Visualizations

- Top-10 feature importance bar chart
- Confusion matrix heatmap
- ROC curve
- Precision-recall curve

---

## Business Report

# Executive Summary

This business report translates the **adult.csv** model output into action for **This is the UCI Adult (Census Income) dataset extracted from the 1994 US Census database. The goal is to predict whether a person's annual income exceeds $50,000 (binary classification). The target column is 'income' (values: '>50K' or '<=50K'). Input features include demographic and employment information: age, workclass, fnlwgt (census sampling weight), education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country. The dataset has approximately 48,842 instances with a mix of continuous and categorical features. Some records contain unknown values denoted by '?' — these should be treated as a separate category for categorical features rather than dropped. The class distribution is heavily imbalanced: approximately 24% earn >50K and 76% earn <=50K, so ROC-AUC and F1-score are preferred over accuracy as primary metrics. The positive class is '>50K'. Interpretability is important as this is a socioeconomic prediction task — Logistic Regression, Decision Tree, and Random Forest are preferred. Note that 'fnlwgt' is a census sampling weight, not a predictive demographic feature, and may be excluded or treated with caution.**. The primary metric **roc_auc** is **0.914**, and the recommended first-wave decisions should use threshold **N/A**.

## 1. Key Findings

- Business goal: This is the UCI Adult (Census Income) dataset extracted from the 1994 US Census database. The goal is to predict whether a person's annual income exceeds $50,000 (binary classification). The target column is 'income' (values: '>50K' or '<=50K'). Input features include demographic and employment information: age, workclass, fnlwgt (census sampling weight), education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country. The dataset has approximately 48,842 instances with a mix of continuous and categorical features. Some records contain unknown values denoted by '?' — these should be treated as a separate category for categorical features rather than dropped. The class distribution is heavily imbalanced: approximately 24% earn >50K and 76% earn <=50K, so ROC-AUC and F1-score are preferred over accuracy as primary metrics. The positive class is '>50K'. Interpretability is important as this is a socioeconomic prediction task — Logistic Regression, Decision Tree, and Random Forest are preferred. Note that 'fnlwgt' is a census sampling weight, not a predictive demographic feature, and may be excluded or treated with caution..
- Target audience: Data Science Team.
- Primary metric roc_auc: 0.914.
- Decision threshold: N/A.
- Preferred strategy: N/A.

## 2. Immediate Action Recommendations

| Priority | Action | Owner | Timeline | Expected Result | Required Resources |
| --- | --- | --- | --- | --- | --- |
| High | Launch the first intervention wave for cases above threshold N/A and prioritize the top N/A cases. | Data Science Team | 1-2 weeks | Focus limited operational capacity on the highest-value high-risk cases. | N/A |
| Medium | Create a human-review and feedback loop so intervention outcomes feed back into model monitoring. | Data Science Team | 2-4 weeks | Improve explainability, operational trust, and the evidence base for the next model refresh. | Review workflow, outcome tracking sheet, recurring governance meeting |

## 3. ROI Analysis

Exact ROI cannot be calculated because the following inputs are missing: case_count, action_cost_per_case, value_per_case. Add them to the JSON payload for a precise estimate.

## 4. Risk Notes

- Monitor data quality, model drift, and operational adoption risks during rollout.
- Confirm privacy, fairness, and governance controls before using predictions in production.

## 5. Implementation Roadmap

- Phase 1 (1-2 weeks): align on goals, thresholds, and review ownership for This is the UCI Adult (Census Income) dataset extracted from the 1994 US Census database. The goal is to predict whether a person's annual income exceeds $50,000 (binary classification). The target column is 'income' (values: '>50K' or '<=50K'). Input features include demographic and employment information: age, workclass, fnlwgt (census sampling weight), education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country. The dataset has approximately 48,842 instances with a mix of continuous and categorical features. Some records contain unknown values denoted by '?' — these should be treated as a separate category for categorical features rather than dropped. The class distribution is heavily imbalanced: approximately 24% earn >50K and 76% earn <=50K, so ROC-AUC and F1-score are preferred over accuracy as primary metrics. The positive class is '>50K'. Interpretability is important as this is a socioeconomic prediction task — Logistic Regression, Decision Tree, and Random Forest are preferred. Note that 'fnlwgt' is a census sampling weight, not a predictive demographic feature, and may be excluded or treated with caution..
- Phase 2 (2-4 weeks): run the first intervention pilot and capture outcome feedback.
- Phase 3 (ongoing): monitor performance, review ROI, and refresh the model on a regular cadence.
