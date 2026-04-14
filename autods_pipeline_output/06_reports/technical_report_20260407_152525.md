# Executive Summary

This technical report was generated with the built-in template mode. The project analyses **_test_smoke.csv** for **target** using a **classification** workflow. The selected model is **svm_rbf**, and the primary metric **roc_auc** reached **0.394**.

## 1. Data Overview and Quality Assessment

- Dataset size: 200 rows and 4 columns.
- Rows after cleaning: 200.
- Class imbalance ratio: N/A.
- Data quality score: N/A.
- Quality notes: {"metadata": {"agent_name": "DataCleaner", "timestamp": "2026-04-07T15:20:57.401571", "version": "2.0"}, "input_data": {"rows": 200, "columns": 4}, "output_data": {"rows": 200, "columns": 4}, "cleaning_summary": {"rows_removed": 0, "columns_dropped": [], "columns_dropped_cnt": 0, "data_retention_pct": 100.0}, "column_classification": {"id_columns": ["income"], "numeric_columns": ["age", "score", "target"], "categorical_columns": [], "special_format_columns": [], "datetime_columns": [], "boolean_columns": [], "constant_columns": [], "dropped_columns": []}, "data_quality_metrics": {"before_cleaning": {"completeness_pct": 100.0, "null_count": 0, "null_pct": 0.0, "duplicate_rows": 0}, "after_cleaning": {"completeness_pct": 100.0, "null_count": 0, "null_pct": 0.0, "duplicate_rows": 0}}, "cleaning_operations": {"anomaly_strategy": "remove", "iqr_multiplier": 1.5, "llm_enabled": false}}.

- N/A

## 2. Feature Engineering Analysis

Features created: {"numeric": ["age", "score"], "categorical": [], "datetime": [], "text": [], "bool_as_numeric": []}.
Features dropped: {"general_drop": ["income"], "constant_columns": [], "high_correlation_drop": []}.
Encoding or transformation steps: N/A.
Top feature signals:
- No feature-importance details were provided.

- N/A

## 3. Model Comparison and Selection

- Selected model: **svm_rbf**.
- Selection reason: Highest roc_auc score on held-out test set..
- Optimization or selection strategy: N/A.
- Training time (seconds): N/A.

- No planner-specific findings were provided.

## 4. Performance Evaluation

| Metric | Value |
| --- | --- |
| Rank | 1 |
| Cv Runtime Seconds | 0.059 |
| Cv Accuracy | 0.481 |
| Cv Precision | 0.44 |
| Cv Recall | 0.635 |
| Cv F1 | 0.518 |
| Cv Roc Auc | 0.526 |
| Cv Std Accuracy | 0.042 |
| Cv Std Precision | 0.035 |
| Cv Std Recall | 0.097 |
| Cv Std F1 | 0.054 |
| Cv Std Roc Auc | 0.098 |
| Test Accuracy | 0.575 |
| Test Precision | 0.524 |
| Test Recall | 0.611 |
| Test F1 | 0.564 |
| Test Roc Auc | 0.394 |

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
