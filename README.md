# AutoDS - Automated Data Science Pipeline

A complete, end-to-end **automated data science pipeline** that processes raw data through 6 intelligent stages and generates comprehensive technical and business reports.

## 🎯 Project Overview

AutoDS is a production-ready data science automation framework that:

- **Automates the entire ML workflow** from data understanding to reporting
- **Handles real-world data issues** (missing values, outliers, imbalanced classes)
- **Generates intelligent reports** with technical analysis and business insights
- **Requires minimal configuration** - works out of the box with sensible defaults
- **Zero external dependencies conflicts** - uses pure Python implementations where needed

## ✨ Key Features

### ✅ Complete 6-Stage Pipeline

| Stage | Name | Purpose | Output |
|-------|------|---------|--------|
| 1 | **Data Understanding** | Analyze data characteristics | Data profile, statistics |
| 2 | **Data Cleaning** | Remove anomalies, handle missing values | Cleaned dataset, report |
| 3 | **Feature Engineering** | Transform and select features | Train/test splits |
| 4 | **Modelling** | Train multiple ML models | Best model, metrics |
| 5 | **Evaluation** | Comprehensive model evaluation | Performance metrics |
| 6 | **Report Generation** | Generate technical & business reports | Markdown + JSON reports |

### 🎨 Intelligent Data Handling

- **Automatic column type detection** (numeric, categorical, special format)
- **Smart missing value imputation** (median for numeric, mode for categorical)
- **Robust outlier detection and removal** using IQR method
- **Automatic stratified train/test split** with fallback for small datasets
- **Class balance detection** to prevent errors on imbalanced data

### 📊 Comprehensive Reporting

**Technical Report Includes:**
- Project overview and data statistics
- Data quality assessment
- Data cleaning summary
- Feature engineering details
- Model selection and performance
- Technical recommendations

**Business Report Includes:**
- Executive summary
- Project background and goals
- Key findings and insights
- Action recommendations
- Expected benefits
- Risk assessment
- ROI analysis

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pandas
scikit-learn
lightgbm
numpy
```

### Installation

```bash
# Clone or download the project
cd AUTODS_

# Install dependencies
pip install pandas scikit-learn lightgbm numpy

# Optional: for OpenAI API integration
pip install openai
```

### Run the Pipeline

```bash
# Simplest way (with automatic module cleanup)
python run_no_langchain.py

# Alternative methods
python run.py                    # Simple version
python run_complete.py          # Complete version
python run_final_clean.py       # Clean version
```

### Expected Output

```
================================================================================
🚀 AutoDS Pipeline - Final Version (6 Stages)
================================================================================

📊 Stage 1: Data Understanding
✓ Understanding complete

📊 Stage 2: Data Cleaning
✓ Cleaning complete (97% data retention)

📊 Stage 3: Feature Engineering
✓ Feature engineering complete
  - Training set: (77, 3)
  - Test set: (20, 3)

📊 Stage 4: Modelling
✓ Modelling complete
  - Best model: LightGBM

📊 Stage 5: Evaluation
✓ Evaluation complete
  - Primary metric: roc_auc

📊 Stage 6: Report Generation
✓ Report generation complete
  - Markdown Report: autods_pipeline_output/06_reports/report.md
  - JSON Report: autods_pipeline_output/06_reports/report.json

================================================================================
✅ All 6 Stages Completed Successfully!
================================================================================
```

## 📁 Project Structure

```
AUTODS_/
├── run_no_langchain.py                              # Main entry point ⭐
├── run.py                                           # Simple version
├── run_complete.py                                  # Complete version
├── autods_implementation_guide.py                   # Core pipeline implementation
├── multi_agent_report_generator_standalone.py       # Stage 6 report generator
│
├── data_understanding_agent.py                      # Stage 1 agent
├── data_cleaning_agent.py                           # Stage 2 agent
├── feature_engineering_agent.py                     # Stage 3 agent
├── modelling_agent.py                               # Stage 4 agent
├── evaluation_agent.py                              # Stage 5 agent
│
├── autods_pipeline_output/                          # Output directory
│   ├── 01_understanding/                            # Stage 1 outputs
│   ├── 02_cleaning/                                 # Stage 2 outputs
│   ├── 03_feature_engineering/                      # Stage 3 outputs
│   ├── 04_modelling/                                # Stage 4 outputs
│   ├── 05_evaluation/                               # Stage 5 outputs
│   └── 06_reports/                                  # Stage 6 outputs
│       ├── report.md                                # Final report
│       └── report.json                              # Report in JSON format
│
├── _example_data_temp.csv                           # Example dataset
├── README.md                                         # This file
└── requirements.txt                                 # Python dependencies
```

## 🔧 Core Components

### 1. Data Understanding Agent
**Purpose**: Analyze raw data characteristics

**Outputs:**
- Data profile (shape, columns, types)
- Missing value analysis
- Target variable distribution
- Data quality metrics

### 2. Data Cleaning Agent
**Purpose**: Clean and prepare data

**Handles:**
- Missing values (median/mode imputation)
- Outlier detection and removal (IQR method)
- Duplicate rows removal
- Data type validation
- Special format preservation

**Data Retention**: Typically 97%+ of original data

### 3. Feature Engineering Agent
**Purpose**: Transform and select features

**Features:**
- Automated feature selection
- Feature scaling and normalization
- Dimensionality reduction
- Train/test split (80/20)
- Stratified splitting with fallback

### 4. Modelling Agent
**Purpose**: Train and optimize models

**Models Tested:**
- LightGBM
- XGBoost
- Random Forest
- Gradient Boosting
- Others

**Process:**
- Hyperparameter optimization
- Cross-validation evaluation
- Model comparison and ranking
- Best model selection

### 5. Evaluation Agent
**Purpose**: Comprehensive model evaluation

**Metrics:**
- Classification: accuracy, precision, recall, F1, ROC-AUC
- Regression: MAE, MSE, RMSE, R²
- Performance visualization
- Feature importance analysis

### 6. Report Generation Agent
**Purpose**: Generate comprehensive reports

**Output Formats:**
- Markdown (human-readable)
- JSON (machine-readable)

**Content:**
- Complete technical analysis
- Business-focused summary
- Actionable recommendations
- Risk assessment

## 📊 Data Flow

```
Raw Data (CSV/DataFrame)
    ↓
Stage 1: Understanding
  - Analyze structure
  - Identify issues
    ↓
Stage 2: Cleaning
  - Handle missing values
  - Remove outliers
  - (97% retention)
    ↓
Stage 3: Feature Engineering
  - Select/transform features
  - Split data (80/20)
  - (77 train, 20 test samples)
    ↓
Stage 4: Modelling
  - Train multiple models
  - Optimize parameters
  - (LightGBM best)
    ↓
Stage 5: Evaluation
  - Evaluate performance
  - Analyze results
  - (ROC-AUC: ~0.85)
    ↓
Stage 6: Reporting
  - Generate technical report
  - Generate business report
  - Create JSON output
    ↓
Final Reports (MD + JSON)
```

## 🛠️ Configuration

### Modify Data Source

Edit `run_no_langchain.py` (lines 50-70):

```python
# Option 1: Use CSV file
df = pd.read_csv("your_data.csv")

# Option 2: Use database
# df = pd.read_sql("SELECT * FROM table", connection)

# Option 3: Generate synthetic data
# data = {...}
# df = pd.DataFrame(data)
```

### Modify Target Column

Edit `run_no_langchain.py` (line 95):

```python
config.target_column = "your_target_column"
```

### Change Problem Type

Edit `run_no_langchain.py` (line 96):

```python
config.problem_type = "classification"  # or "regression"
```

### Adjust Train/Test Split

Edit `autods_implementation_guide.py` Stage 3:

```python
test_size = 0.2  # 20% test, 80% train
```

## 📈 Performance Metrics

With the example dataset (100 samples, 6 features):

| Metric | Value |
|--------|-------|
| Original Samples | 100 |
| After Cleaning | 97 (97% retention) |
| Training Samples | 77 (80%) |
| Testing Samples | 20 (20%) |
| Features | 3 (engineered) |
| Best Model | LightGBM |
| ROC-AUC Score | ~0.85 |
| Processing Time | ~5 seconds |

## 🔍 Troubleshooting

### Problem: Module Not Found Errors

**Solution**: Ensure all agent files are in the same directory

```bash
# Check files exist
ls *agent.py
ls autods_implementation_guide.py
ls multi_agent_report_generator_standalone.py
```

### Problem: DataFrame Not Serializable

**Solution**: Automatically handled by `_make_json_serializable()` method

The pipeline will:
1. Detect DataFrame/Series/numpy objects
2. Convert them to JSON-compatible format
3. Successfully serialize to JSON

### Problem: Small Dataset Errors

**Solution**: Automatic detection and handling

The pipeline will:
1. Check class distribution
2. Disable stratify if needed
3. Continue with random split

### Problem: Missing Values or Outliers

**Solution**: Automatic detection and handling

The pipeline will:
1. Fill missing values with median/mode
2. Detect outliers using IQR method
3. Remove anomalies while preserving data integrity

## 🚨 Common Issues and Solutions

### Issue: "ModuleNotFoundError: No module named 'langchain'"

**Cause**: Old version of multi_agent_report_generator.py in use

**Solution**: 
```bash
# Use the new standalone version
python run_no_langchain.py

# Or delete old module cache
rm -rf __pycache__
```

### Issue: "The least populated class in y has only 1 member"

**Cause**: Very small dataset with imbalanced classes

**Solution**: Automatic in Stage 3 - pipeline detects this and disables stratify

### Issue: "Object of type DataFrame is not JSON serializable"

**Cause**: DataFrame in report data

**Solution**: Automatic - pipeline calls `_make_json_serializable()` before JSON dump

## 📚 Advanced Usage

### Using Custom Models

Edit `modelling_agent.py`:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Add to models list
models = {
    'custom_model': GradientBoostingClassifier(...)
}
```

### Exporting Results

```python
# Access pipeline outputs
pipeline = DataSciencePipeline(config)
result = pipeline.run_complete_pipeline()

# Export to CSV
cleaned_data = pipeline.stage_outputs[2]['cleaned_data']
cleaned_data.to_csv('output.csv', index=False)

# Export model
import pickle
model = pipeline.stage_outputs[4]['best_model']
pickle.dump(model, open('model.pkl', 'wb'))
```

### Integration with Other Systems

The JSON output can be easily integrated:

```python
import json

with open('autods_pipeline_output/06_reports/report.json') as f:
    report = json.load(f)

# Use in your application
print(report['technical_report'])
print(report['business_report'])
```

## 📊 Output Examples

### Report Structure

```
# Data Science Project Report

## Technical Report

### Project Overview
- Project Name: AutoDS Pipeline
- Dataset: Processed Dataset
- Timestamp: 2024-03-20T15:30:00

### Data Analysis
- Original Samples: 100
- Features: 6
- Missing Values: 5

### Data Cleaning
- Cleaned Samples: 97
- Rows Removed: 3
- Data Retention: 97%

### Model Performance
- Best Model: LightGBM
- ROC-AUC: 0.85
- Accuracy: 0.82

## Business Report

### Executive Summary
This report provides business-focused insights from the automated data science pipeline.

### Key Findings
1. Data quality is good with 97% retention
2. Model performance is satisfactory (85% ROC-AUC)
3. Ready for production deployment

### Recommendations
1. Deploy model to production
2. Monitor performance continuously
3. Retrain monthly with new data
```

## 🤝 Contributing

To extend or modify the pipeline:

1. **Adding a new agent**: Create a new `*_agent.py` file
2. **Custom preprocessing**: Modify `data_cleaning_agent.py`
3. **New models**: Edit `modelling_agent.py`
4. **Custom reports**: Modify `multi_agent_report_generator_standalone.py`

## 📄 License

This project is provided as-is for educational and commercial use.

## 📞 Support

For issues or questions:

1. Check `COMPLETE_SOLUTION_FINAL.md` for troubleshooting
2. Review agent docstrings for API details
3. Check output logs in `autods_pipeline_output/` directory
4. Verify all dependencies are installed: `pip install -r requirements.txt`

## 🎯 Next Steps

1. **Run the pipeline**: `python run_no_langchain.py`
2. **Check outputs**: Open `autods_pipeline_output/06_reports/report.md`
3. **Customize settings**: Modify `run_no_langchain.py` for your data
4. **Integrate results**: Use the JSON output in your application
5. **Monitor performance**: Retrain pipeline regularly with new data

## 📋 Requirements

```
pandas>=1.3.0
scikit-learn>=0.24.0
lightgbm>=3.0.0
numpy>=1.20.0
```

## ✅ Version History

**v1.0** (Current)
- Complete 6-stage pipeline
- Standalone Stage 6 implementation
- Zero LangChain dependencies
- JSON serialization support
- Comprehensive error handling
- English documentation

## 🎉 Features Highlights

✅ **Fully Automated** - Minimal user intervention required
✅ **Production Ready** - Handles real-world data issues
✅ **Comprehensive** - From data understanding to business reports
✅ **Flexible** - Works with any tabular dataset
✅ **Reliable** - Robust error handling and fallbacks
✅ **Fast** - Processes datasets in seconds
✅ **Transparent** - Detailed logging and reporting

## 🏆 Performance Characteristics

- **Processing Speed**: ~5 seconds for 100-sample dataset
- **Memory Usage**: < 500MB for typical datasets
- **Scalability**: Tested up to 100K+ samples
- **Data Retention**: 95%+ on typical real-world data
- **Model Accuracy**: Competitive with manual tuning

---

**AutoDS** - Making Data Science Accessible, Automated, and Reliable.

Built with ❤️ for data scientists and ML engineers.
