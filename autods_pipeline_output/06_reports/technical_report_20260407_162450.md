# Executive Summary

This technical report was generated with the built-in template mode. The project analyses **rental_training.csv** for **rent** using a **regression** workflow. The selected model is **xgboost_regressor**, and the primary metric **rmse** reached **17.312**.

## 1. Data Overview and Quality Assessment

- Dataset size: 3,434 rows and 20 columns.
- Rows after cleaning: 2,813.
- Class imbalance ratio: N/A.
- Data quality score: N/A.
- Quality notes: {"metadata": {"agent_name": "DataCleaner", "timestamp": "2026-04-07T16:20:44.937504", "version": "2.0"}, "input_data": {"rows": 3434, "columns": 20}, "output_data": {"rows": 2813, "columns": 20}, "cleaning_summary": {"rows_removed": 621, "columns_dropped": [], "columns_dropped_cnt": 0, "data_retention_pct": 81.92}, "column_classification": {"id_columns": ["street_name", "phone_number", "email"], "numeric_columns": ["number_of_bedrooms", "rent", "floor_area", "number_of_bathrooms", "building_number"], "categorical_columns": ["level", "suburb", "furnished", "tenancy_preference", "point_of_contact", "secondary_address", "street_suffix", "prefix", "first_name", "last_name", "gender"], "special_format_columns": [], "datetime_columns": ["advertised_date"], "boolean_columns": [], "constant_columns": [], "dropped_columns": []}, "data_quality_metrics": {"before_cleaning": {"completeness_pct": 98.31, "null_count": 1161, "null_pct": 1.69, "duplicate_rows": 0}, "after_cleaning": {"completeness_pct": 100.0, "null_count": 0, "null_pct": 0.0, "duplicate_rows": 0}}, "cleaning_operations": {"anomaly_strategy": "remove", "iqr_multiplier": 1.5, "llm_enabled": false}}.

- N/A

## 2. Feature Engineering Analysis

Features created: {"numeric": ["number_of_bedrooms", "floor_area", "number_of_bathrooms", "building_number"], "categorical": ["level", "suburb", "furnished", "tenancy_preference", "point_of_contact", "secondary_address", "street_suffix", "prefix", "first_name", "last_name", "gender"], "datetime": ["advertised_date"], "text": [], "bool_as_numeric": []}.
Features dropped: {"general_drop": ["street_name", "phone_number", "email"], "constant_columns": [], "high_correlation_drop": []}.
Encoding or transformation steps: N/A.
Top feature signals:
- Unknown feature (0.26)
- Unknown feature (0.078)
- Unknown feature (0.057)
- Unknown feature (0.052)
- Unknown feature (0.042)

- N/A

## 3. Model Comparison and Selection

- Selected model: **xgboost_regressor**.
- Selection reason: Highest rmse score on held-out test set..
- Optimization or selection strategy: N/A.
- Training time (seconds): N/A.

- Best model: xgboost_regressor
- Evaluated N/A candidate model(s)
- Planner primary metric: rmse.
- Planner reasoning: Rule-based fallback (no LLM available).

## 4. Performance Evaluation

| Metric | Value |
| --- | --- |
| Rank | 1 |
| Cv Runtime Seconds | 1.711 |
| Cv Rmse | 19.733 |
| Cv Mae | 9.474 |
| Cv R2 | 0.671 |
| Cv Std Rmse | 2.471 |
| Cv Std Mae | 0.48 |
| Cv Std R2 | 0.078 |
| Test Rmse | 17.312 |
| Test Mae | 8.737 |
| Test R2 | 0.785 |

| Confusion Matrix Item | Value |
| --- | --- |
| N/A | N/A |

- N/A

## 5. Technical Recommendations and Improvement Directions

### Improvement Recommendations

- Establish a monitoring plan for feature drift, score drift, and periodic model refresh.
- Collect additional explanatory variables to improve model robustness and actionability.
- Monitor model performance on production data.
- Retrain periodically as new data arrives.

### Recommended Visualizations

- Top-10 feature importance bar chart
- Residual plot
- Predicted vs actual scatter plot
- Error distribution plot
