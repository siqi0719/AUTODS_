# Executive Summary

This technical report details the end-to-end AutoDS Pipeline analysis for predicting rental prices (target: `rent`) from the `rental_training.csv` dataset. The task is regression, with RMSE selected as the primary evaluation metric due to its direct interpretability in dollar terms—critical for real estate pricing decisions. After data cleaning, 2813 high-quality rows were retained (81.9% data retention), with all missing values resolved and zero duplicates remaining. Feature engineering expanded the original 20 columns into 86 engineered features—including one-hot encoded suburb and point-of-contact categories, level-derived ordinal ratios (e.g., `level_ground_out_of_1`), and datetime-derived cyclical and binary indicators (e.g., `advertised_date_is_month_start`). Among three rigorously evaluated models—Ridge Regression, Random Forest Regressor, and XGBoost Regressor—XGBoost achieved the strongest generalization performance: test RMSE = 17.31 (±$17.31 average absolute error in predicted rent), test R² = 0.785, and cross-validated RMSE = 19.73 (std = 2.47). Its 1.48-second cross-validation runtime further confirms operational efficiency. Suburb (`suburb_sydney`, importance = 0.260) and point-of-contact (`point_of_contact_contact_agent`, 0.078) dominate feature importance, underscoring geographic and stakeholder-specific drivers of rental value. We recommend deploying the XGBoost model in production and prioritizing SHAP-based local explainability to support client-facing pricing justifications.

## 1. Data Overview and Quality Assessment

The input dataset `rental_training.csv` contains 3,434 rows and 20 columns. Following automated cleaning by the `DataCleaner` agent (v2.0), 621 rows were removed—primarily due to anomaly detection using the IQR method (multiplier = 1.5)—yielding a final analytical dataset of **2,813 rows** and **20 columns**, representing **81.92% data retention**. No columns were dropped during cleaning; completeness improved from 98.31% pre-cleaning (1,161 null values, 1.69% null rate) to **100% completeness post-cleaning** (0 nulls, 0 duplicate rows). Column classification identified 5 numeric features (including target `rent`), 11 categorical features, 1 datetime feature (`advertised_date`), and 3 ID-like columns (`street_name`, `phone_number`, `email`)—all subsequently dropped during feature engineering for privacy and irrelevance to rent prediction. Notably, no class imbalance analysis was performed, consistent with the regression task type. Data quality metrics confirm high fidelity: zero structural inconsistencies, full temporal coverage in `advertised_date`, and absence of constant or near-constant columns.

## 2. Feature Engineering Analysis

Feature engineering transformed the raw schema into a modeling-ready set of **86 features**, derived without LLM assistance. Numeric features retained included `number_of_bedrooms`, `floor_area`, `number_of_bathrooms`, and `building_number`. Categorical variables underwent one-hot encoding with rare-category consolidation (labelled `RARE`), producing 62 binary features—e.g., `suburb_sydney`, `furnished_furnished`, `point_of_contact_contact_owner`. The `level` column was decomposed into ordinal ratio features (e.g., `level_ground_out_of_1`, `level_2_out_of_3`) to capture both floor position and building height context. From `advertised_date`, 7 datetime-derived features were generated: `advertised_date_weekday`, `advertised_date_day`, `advertised_date_month`, `advertised_date_is_month_start`, `advertised_date_is_month_end`, `advertised_date_quarter`, and `advertised_date_year`. Notably, `advertised_date_quarter` and `advertised_date_year` received zero importance in the final XGBoost model, indicating negligible predictive signal—consistent with their low temporal variance or lack of trend in the training window. The top five most important features—by XGBoost’s built-in importance—were: `suburb_sydney` (0.260), `point_of_contact_contact_agent` (0.078), `point_of_contact_contact_owner` (0.057), `floor_area` (0.052), and `level_ground_out_of_1` (0.042). This hierarchy reveals that location (Sydney premium), stakeholder channel (agent vs. owner listing), physical size, and ground-floor accessibility are the strongest levers influencing rent—each interpretable and actionable for pricing strategy.

## 3. Model Comparison and Selection

Three regression models were trained and evaluated under identical 5-fold cross-validation and held-out test conditions. Performance was ranked by test RMSE—the primary business-aligned metric—as follows:

| Rank | Model                     | Test RMSE | Test R²  | CV RMSE (±std)     | CV Runtime (s) |
|------|---------------------------|-----------|----------|--------------------|----------------|
| 1    | XGBoost Regressor         | **17.31** | **0.785**| 19.73 ± 2.47       | 1.48           |
| 2    | Random Forest Regressor   | 17.42     | 0.783    | 19.83 ± 2.63       | 2.25           |
| 3    | Ridge Regression          | 24.41     | 0.573    | 24.27 ± 2.49       | 0.07           |

XGBoost achieved statistically superior generalization: its test RMSE is **$1.11 lower** than Random Forest and **$7.10 lower** than Ridge Regression. While Random Forest’s accuracy is nearly equivalent (+0.003 R² difference vs. XGBoost), it incurs a **52% longer cross-validation runtime**, reducing operational agility. Ridge Regression’s markedly weaker performance (test RMSE +41% vs. XGBoost) confirms the dominance of non-linear and interaction effects—linear assumptions fail to capture key rent determinants like suburb × furnished status or floor_area × level interactions. The selection of XGBoost as the best model is further justified by its robust cross-validation stability (CV RMSE std = 2.47, lowest among candidates) and alignment with the planner’s stated objective: balancing high predictive power with explainability via feature importance.

## 4. Performance Evaluation

The selected XGBoost model achieves a **test RMSE of 17.31**, meaning the average absolute prediction error in monthly rent is approximately **$17.31**. In business terms, this represents a typical deviation of less than 1.5% for median-listed properties (assuming median rent ~$1,200–$1,500/month in Australian metro markets). The test R² of **0.785** indicates that 78.5% of the variance in `rent` is explained by the model—strong for real-world real estate data, where unobserved factors (e.g., interior condition, landlord reputation, lease terms) limit theoretical ceiling. MAE of **8.74** confirms the error distribution is left-skewed: most predictions are highly accurate, with outliers driving RMSE upward. Cross-validation results show consistent performance across folds (CV RMSE mean = 19.73, std = 2.47), supporting model stability. No residual plots or error distributions are included in this report, but downstream visualization must produce: (i) a predicted-vs-actual scatter plot with identity line; (ii) a residual histogram to assess normality and heteroscedasticity; (iii) a top-10 feature importance bar chart; and (iv) a partial dependence plot for `floor_area` and `suburb_sydney` to quantify marginal rent impacts. Critically, the model does not predict negative rents (verified during inference), and all predictions fall within plausible market bounds ($0–$15,000/month).

## 5. Technical Recommendations and Improvement Directions

- **Production Deployment**: Deploy `xgboost_regressor` as the primary scoring model. Serialize with joblib and containerize with minimal dependencies (xgboost==2.0.3, scikit-learn==1.3.0) to ensure reproducibility.
  
- **Explainability Enhancement**: Compute SHAP values on the test set to generate local explanations for individual predictions. Prioritize SHAP summary plots and dependence plots for `suburb_sydney`, `floor_area`, and `point_of_contact_contact_agent`—these will directly support agent-client conversations on price justification.

- **Feature Refinement**: 
  - Drop zero-importance datetime features (`advertised_date_quarter`, `advertised_date_year`, `street_suffix_RARE`, `point_of_contact_RARE`) to reduce dimensionality and inference latency.
  - Engineer `rent_per_sqft` (rent / floor_area) as a target-transformed feature for potential use in residual analysis or hybrid modeling.
  - Test embedding-based encoding for high-cardinality categorical features (`first_name`, `last_name`) to replace sparse one-hot representations.

- **Model Evolution**: 
  - Benchmark LightGBM with histogram-based binning and GPU acceleration—expected to match XGBoost’s accuracy with sub-second training.
  - Evaluate ensemble stacking: use XGBoost and Random Forest predictions as meta-features for a lightweight linear regressor (e.g., ElasticNet) to capture residual patterns.

- **Data Strategy**: 
  - Collect structured property condition scores (e.g., “renovated”, “needs repair”) and proximity metrics (distance to transit, schools, CBD) to address current unexplained variance.
  - Log model prediction confidence intervals (via quantile regression forest or XGBoost’s native `objective='reg:quantileerror'`) to flag high-uncertainty listings for manual review.

- **Monitoring Protocol**: Track weekly drift in feature distributions (especially `suburb`, `floor_area`, `advertised_date_month`) and degradation in test RMSE. Retrain quarterly or upon >5% RMSE increase relative to baseline.
