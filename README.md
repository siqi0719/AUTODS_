# AutoDS - Automated Data Science Pipeline

A complete, end-to-end **automated data science pipeline** that processes raw data through 7 intelligent stages and generates comprehensive technical and business reports.

---

## 🎯 Project Overview

AutoDS is a production-ready data science automation framework that:

- **Understands your business goal** from a plain-English description and configures the entire pipeline automatically
- **Automates the entire ML workflow** from data understanding to reporting
- **Handles real-world data issues** (missing values, outliers, imbalanced classes, inconsistent formats, mixed separators)
- **Generates intelligent reports** with technical analysis and business insights
- **Requires minimal configuration** — works out of the box with sensible defaults
- **Supports LLM integration** at multiple stages (Anthropic Claude and OpenAI) with rule-based fallback

---

## ✨ Key Features

### ✅ Complete 7-Stage Pipeline

| Stage | Name | Purpose | Output |
|-------|------|---------|--------|
| 0 | **Planner** | Parse business description, generate all agent configs | Per-agent config plan, adaptive adjustments |
| 1 | **Data Understanding** | Analyse data characteristics | Data profile, statistics |
| 2 | **Data Cleaning** | Remove anomalies, handle missing values | Cleaned dataset, quality report |
| 3 | **Feature Engineering** | Transform and select features | Train / test splits |
| 4 | **Modelling** | Train multiple ML models | Best model, leaderboard |
| 5 | **Evaluation** | Comprehensive model evaluation | Performance metrics |
| 6 | **Report Generation** | Generate technical & business reports | Markdown + JSON reports |

### 🧠 LLM-Powered Planning (Stage 0)

- Accepts a **natural-language business description in any language** and automatically infers `target_column`, `problem_type`, `primary_metric`, and candidate models
- **Input normalisation**: informal, non-technical, or non-English descriptions are automatically interpreted and converted into a structured task specification before planning — no data science jargon required
- **Adaptive replanning** after Stage 1: detects class imbalance, high missing rates, and small datasets, and adjusts downstream configs accordingly
- **Post-modelling review** generates a professional narrative summary for the final report
- Works with **Anthropic Claude** (primary) or **OpenAI** (fallback); silently falls back to rule-based defaults when no API key is present

### 🎨 Intelligent Data Cleaning (Stage 2)

- **Column-name normalisation** — strip, lowercase, snake_case
- **Pseudo-null unification** — `"None"`, `"nan"`, `"N/A"`, `""`, `"-"`, `"unknown"`, … → `NaN`
- **ID / constant / duplicate column detection and removal**
- **High-missing-rate column removal** (configurable threshold, default 60 %)
- **Smart missing-value imputation** — median / mean / zero / drop-row for numeric; mode or `"MISSING"` for categorical; LLM-generated strategy when enabled
- **Anomaly handling** — IQR-based remove *or* clip (configurable); target column always protected
- **Datetime column detection and ISO-8601 normalisation**
- **Boolean column unification** → `"0"` / `"1"`
- **Numeric range constraints** (per-column min / max)
- **Special-format preservation** — phone numbers, postal codes, ID cards, order numbers, bank cards
- Optional **LLM column-type advisor** for ambiguous columns

### 📊 Comprehensive Reporting

**Technical Report includes:**
- Project overview and data statistics
- Data quality assessment (before / after cleaning)
- Data cleaning summary
- Feature engineering details
- Model selection and performance
- Planner reasoning and adaptive adjustments
- Technical recommendations

**Business Report includes:**
- Executive summary
- Project background and goals
- Key findings and insights
- Action recommendations
- Expected benefits
- Risk assessment
- ROI analysis

---

## 🚀 Quick Start

### Prerequisites

```
Python 3.9+
pandas · scikit-learn · lightgbm · numpy
```

### Installation

```bash
cd AUTODS_
pip install pandas scikit-learn lightgbm numpy python-dotenv

# For LLM features (optional — pipeline runs without these)
pip install anthropic                  # Claude (recommended)
pip install langchain-openai langchain-core   # OpenAI alternative
```

### API Key Setup (optional)

Create a `.env` file in the project root:

```bash
# Anthropic Claude (preferred by PlannerAgent and DataCleaningAgent)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxx

# OpenAI (used as fallback, and by FeatureEngineeringAgent / ModellingAgent)
OPENAI_API_KEY=sk-xxxxxxxxxx
```

If neither key is present, all LLM features fall back to rule-based defaults and the pipeline runs normally.

### Configure and Run

Edit `run.py`:

```python
# Describe your task in plain English (any language, any style)
config.business_description = (
    "We have a customer dataset and want to predict whether a customer "
    "will churn. The target column is 'churn'. The model should be interpretable."
)
config.use_planner   = True      # set False to skip Stage 0
config.target_column = "churn"   # optional — Planner can infer this
config.problem_type  = "classification"

# Optional: supply a metadata file to help the Planner label columns
# and convert non-CSV formats automatically
config.metadata_file = "data_dictionary.json"   # or .csv / .txt
```

Then run:

```bash
python run.py
```

### Expected Output

```
================================================================================
STAGE 0: PLANNER — Pre-run Planning
============================================================
[PlannerAgent] Using Anthropic Claude (claude-sonnet-4-6)
  target_column  : churn
  problem_type   : classification
  primary_metric : roc_auc
  candidate_models: ['LogisticRegression', 'RandomForest', 'GradientBoosting']

STAGE 1: DATA UNDERSTANDING
================================================================================
✓ Understanding complete

STAGE 2: DATA CLEANING
================================================================================
[DataCleaner] Input  : 100 rows × 6 cols
[DataCleaner] Output : 94 rows × 5 cols

STAGE 3: FEATURE ENGINEERING
================================================================================
✓ Feature engineering complete — Training set: (75, 4)  Test set: (19, 4)

STAGE 4: MODELLING
================================================================================
✓ Modelling complete — Best model: RandomForest

STAGE 5: EVALUATION
================================================================================
✓ Evaluation complete — Primary metric: roc_auc

STAGE 6: REPORT GENERATION
================================================================================
✓ Report saved: autods_pipeline_output/06_reports/report.md

================================================================================
✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!
================================================================================
```

---

## 📁 Project Structure

```
AUTODS_/
├── run.py                                      # Entry point ⭐
├── autods_implementation_guide.py              # Core pipeline orchestration
│
├── planner_agent.py                            # Stage 0 — Planner
├── data_understanding_agent.py                 # Stage 1 — Understanding
├── data_cleaning_agent.py                      # Stage 2 — Cleaning (v2.0)
├── feature_engineering_agent.py                # Stage 3 — Feature Engineering
├── modelling_agent.py                          # Stage 4 — Modelling
├── evaluation.py                               # Stage 5 — Evaluation
├── multi_agent_report_generator.py             # Stage 6 — Report Generation
│
├── autods_pipeline_output/                     # Runtime outputs
│   ├── 00_planning/                            # Stage 0: plan JSON files
│   ├── 01_understanding/                       # Stage 1: data profile
│   ├── 02_cleaning/                            # Stage 2: cleaned data + report
│   ├── 03_feature_engineering/                 # Stage 3: train/test splits
│   ├── 04_modelling/                           # Stage 4: models + leaderboard
│   ├── 05_evaluation/                          # Stage 5: evaluation metrics
│   └── 06_reports/                             # Stage 6: report.md + report.json
│
├── README.md                                   # This file
├── README_DataCleaningAgent.md                 # DataCleaningAgent v2.0 API reference
├── requirements.txt                            # Python dependencies
└── pyproject.toml                              # Build configuration
```

---

## 🔧 Core Components

### Stage 0 — Planner Agent (`planner_agent.py`)

**Purpose**: Translate a business description into machine-readable configs for all downstream agents.

**Three functions:**

| Function | Trigger point | What it does |
|----------|--------------|-------------|
| `plan()` | Pipeline start | Reads business description + data schema (+ optional extra files) → outputs target column, problem type, metric, candidate models, feature task description |
| `replan_after_understanding()` | After Stage 1 | Inspects DataUnderstanding output; adjusts metric (e.g., imbalance → F1), disables LLM planner on high-missing data, reduces CV folds on small datasets |
| `review_modelling()` | After Stage 5 | Reads leaderboard; writes key findings and a 2–4 sentence narrative for the business report |

**Config** (`PlannerConfig`):

```python
PlannerConfig(
    llm_model_anthropic = "claude-sonnet-4-6",
    llm_model_openai    = "gpt-4o-mini",
    temperature         = 0.0,
    use_adaptive_replanning = True,
    data_sample_rows    = 5,
)
```

**Input normalisation** (automatic, no configuration required):

Before generating the pipeline configuration, `plan()` runs a dedicated LLM pass to interpret the `business_description`.  This means you can write in any language or style:

| Input style | Example |
|-------------|---------|
| Plain Chinese | `"我想预测客户会不会流失，越准越好"` |
| Colloquial English | `"figure out which loans are gonna go bad"` |
| Domain jargon | `"identify churners using RFM features"` |
| Incomplete | `"predict the label column"` |

The normalised description and the original raw text are both saved to `initial_plan.json` under the keys `normalised_description` and `raw_business_description` for full traceability.  When no LLM is available the raw text is used as-is.

**Extra file input** (`extra_files` parameter of `plan()`):

`plan()` accepts an optional `extra_files` list of file paths that provide supplementary domain context — for example, a data dictionary, a business requirements document, or a metadata JSON exported from another tool. The Planner reads each file, extracts a text summary, and appends it to the LLM prompt so that the generated plan can reflect domain-specific knowledge that is not visible in the data schema alone.

Supported formats:

| Format | How it is read |
|--------|---------------|
| `.json` | Pretty-printed content (truncated to 3 000 chars) |
| `.xlsx` / `.xls` | Shape + column names + first 3 rows |
| `.csv` | Shape + column names + first 3 rows (separator auto-detected) |
| `.txt` / `.md` | Raw text (truncated to 3 000 chars) |

Example usage in `run.py`:

```python
plan = planner.plan(
    business_description = config.business_description,
    data_sample          = raw_data.head(5),
    extra_files          = [
        "data_dictionary.json",   # column descriptions
        "business_requirements.md",
    ],
)
```

**Data preparation** (`prepare_data()` method):

Before the main planning step, `prepare_data()` converts a raw data file into a pipeline-ready DataFrame (and optionally a CSV).  It is called automatically when `config.metadata_file` is set.

Supported input formats:

| Format | How it is loaded |
|--------|-----------------|
| `.csv` / `.tsv` / `.txt` | Delimited text — separator auto-detected |
| `.xlsx` / `.xls` | Excel — first sheet used |
| `.json` | Array of records or dict-of-lists |

Supported metadata formats:

| Format | How it is parsed |
|--------|-----------------|
| `.json` | `{"columns": [...], "value_mappings": {...}}` |
| `.csv` | First column treated as ordered column names |
| `.txt` / `.names` | One column name per line (UCI style) |

Example usage in `run.py`:

```python
config.data_path     = "raw_data.tsv"        # could be TSV, Excel, JSON, …
config.metadata_file = "data_dictionary.json" # optional column descriptions
```

When `metadata_file` is set the prepared CSV is saved to
`autods_pipeline_output/00_planning/prepared_data.csv` and `config.data_path`
is updated automatically so all downstream stages use it.

**Plan output saved to** `autods_pipeline_output/00_planning/`:
- `initial_plan.json`
- `prepared_data.csv` *(only when `metadata_file` is set)*
- `replan_after_understanding.json`
- `modelling_review.json`

---

### Stage 1 — Data Understanding (`data_understanding_agent.py`)

**Purpose**: Analyse raw data characteristics and produce a structured profile.

**Outputs:**
- Data shape, column types, missing-value analysis
- Target variable distribution and class imbalance detection
- Data quality metrics
- Inferred problem type (used when Planner is disabled)

---

### Stage 2 — Data Cleaning (`data_cleaning_agent.py`)

**Purpose**: Produce a clean, analysis-ready DataFrame.

**Cleaning pipeline (in execution order):**

```
Column-name normalisation     → strip · lowercase · snake_case
Pseudo-null unification       → "None"/"nan"/"N/A"/""/"–"/… → NaN
Column-type identification    → ID · special · numeric · datetime · bool · categorical
Constant-column removal       → unique non-null values == 1
Duplicate-column removal      → identical content columns
High-missing-rate removal     → missing rate > threshold (default 60 %)
Duplicate-row removal
Missing-value imputation      → numeric: median/mean/zero/drop_row (LLM plan or default)
                                categorical: mode or "MISSING"
Anomaly handling (IQR)        → remove rows or clip to bounds; target column protected
Datetime normalisation        → YYYY-MM-DD
Boolean unification           → "0" / "1"
Special-format preservation   → trim only
Categorical consistency       → strip + lowercase
Numeric range constraints     → per-column min / max
```

**Config** (`DataCleaningConfig`):

```python
DataCleaningConfig(
    target_column            = "churn",     # protected from anomaly removal
    anomaly_strategy         = "clip",      # "remove" (default) or "clip"
    iqr_multiplier           = 1.5,
    missing_drop_threshold   = 0.6,
    use_llm_column_advisor   = True,        # LLM for ambiguous columns
    column_constraints       = {
        "age":   {"min": 0, "max": 120},
        "score": {"min": 0, "max": 100},
    },
)
```

For full API documentation see [`README_DataCleaningAgent.md`](README_DataCleaningAgent.md).

---

### Stage 3 — Feature Engineering (`feature_engineering_agent.py`)

**Purpose**: Transform features and prepare train / test splits.

- Automated feature selection and scaling
- Rare-category grouping
- High-correlation column removal
- Optional LLM-generated feature actions
- Task description and `use_llm_planner` flag set automatically by Planner (Stage 0)

---

### Stage 4 — Modelling (`modelling_agent.py`)

**Purpose**: Train and compare multiple ML models.

- Candidate model list set automatically by Planner (Stage 0)
- Primary metric and CV folds also propagated from Planner
- Cross-validation evaluation and leaderboard ranking
- Best model selection

---

### Stage 5 — Evaluation (`evaluation.py`)

**Purpose**: Comprehensive model evaluation and selection.

**Metrics:**
- Classification: accuracy, precision, recall, F1, ROC-AUC
- Regression: MAE, MSE, RMSE, R²

---

### Stage 6 — Report Generation (`multi_agent_report_generator.py`)

**Purpose**: Generate final technical and business reports.

**Output formats:** Markdown + JSON

**Content includes:**
- Full technical analysis across all stages
- Business-focused summary
- Planner reasoning, adaptive adjustments, and modelling review
- Actionable recommendations and risk assessment

---

## 📊 Data Flow

```
Business Description + CSV
        │
        ▼
┌─────────────────────┐
│  Stage 0: Planner   │  → initial_plan.json
│  (LLM or rules)     │
└────────┬────────────┘
         │  configs for Stages 1–4
         ▼
┌─────────────────────┐
│  Stage 1:           │
│  Understanding      │  → data profile, quality metrics
└────────┬────────────┘
         │  adaptive replanning
         ▼
┌─────────────────────┐
│  Stage 2: Cleaning  │  → cleaned DataFrame, cleaning_report.json
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Stage 3: Feature   │
│  Engineering        │  → X_train, X_test, y_train, y_test
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Stage 4: Modelling │  → leaderboard, best model
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Stage 5: Evaluation│  → evaluation metrics
└────────┬────────────┘
         │  post-modelling review
         ▼
┌─────────────────────┐
│  Stage 6: Report    │  → report.md, report.json
└─────────────────────┘
```

---

## 🛠️ Configuration

### Pipeline-Level Settings (`PipelineConfig`)

```python
config = PipelineConfig()
config.data_path            = "your_data.csv"
config.target_column        = "target"        # optional if use_planner=True
config.problem_type         = "classification" # optional if use_planner=True
config.random_state         = 42

# Separator — leave unset (None) for auto-detection, or override explicitly:
# config.csv_sep = ";"   # e.g. for Bank Marketing and other semicolon files
# config.csv_sep = "\t"  # for TSV files

# Planner settings
config.business_description = "Predict customer churn from usage data."
config.use_planner          = True    # False to skip Stage 0 entirely
config.metadata_file        = None    # optional: path to data dictionary file
```

### Skipping the Planner

```python
config.use_planner   = False
config.target_column = "target"
config.problem_type  = "classification"
# All other configs use their own defaults
```

### Custom Cleaning Config

Pass a `DataCleaningConfig` directly inside `run_stage_2_cleaning()` in `autods_implementation_guide.py` if you need full control over cleaning behaviour. See [`README_DataCleaningAgent.md`](README_DataCleaningAgent.md) for all options.

---

## 🔍 Troubleshooting

### No LLM API key

All LLM features (Planner, DataCleaning column advisor, FeatureEngineering, Modelling) degrade gracefully to rule-based defaults. The pipeline always runs end-to-end without any API key.

### Wrong number of columns / single-column DataFrame

The pipeline auto-detects the CSV separator (comma, semicolon, tab, pipe) before reading any data.  If detection produces the wrong result, override it explicitly:

```python
config.csv_sep = ";"   # Bank Marketing and other semicolon-delimited files
config.csv_sep = "\t"  # TSV files
```

### Small dataset / class imbalance

Stage 0 (Planner) detects these automatically and adjusts CV folds and the primary metric. If the Planner is disabled, Stage 3 already handles stratified-split fallback automatically.

### DataFrame not JSON-serialisable

Handled automatically by `_make_json_serializable()` in the pipeline orchestrator.

### Module not found

Ensure all `*_agent.py` files, `evaluation.py`, `multi_agent_report_generator.py`, and `autods_implementation_guide.py` are in the same directory as `run.py`.

---

## 📈 Performance Characteristics

- **Processing speed**: ~5–10 seconds for a 100-row dataset
- **Memory usage**: < 500 MB for typical datasets
- **Scalability**: tested up to 100 K+ rows
- **Data retention**: typically 90 %+ after cleaning on real-world data

---

## 📚 Advanced Usage

### Accessing Intermediate Results

```python
from autods_implementation_guide import PipelineConfig, DataSciencePipeline

config = PipelineConfig()
config.data_path            = "data.csv"
config.target_column        = "target"
config.business_description = "Predict customer churn."

pipeline = DataSciencePipeline(config)
result   = pipeline.run_complete_pipeline()

# Access any stage output
planner_plan  = pipeline.stage_outputs[0]["plan"]
cleaned_df    = pipeline.stage_outputs[2]["cleaned_data"]
best_model    = pipeline.stage_outputs[4]["modelling_result"]["best_model_name"]
review        = pipeline.stage_outputs[5]["planner_review"]["review_text"]
```

### Using DataCleaningAgent Standalone

```python
from data_cleaning_agent import DataCleaningAgent, DataCleaningConfig

config = DataCleaningConfig(
    target_column   = "churn",
    anomaly_strategy= "clip",
)
agent  = DataCleaningAgent(config=config)
result = agent.run("data.csv")

cleaned_df = result["data"]
report     = result["report"]
```

### Integrating the Final Report

```python
import json

with open("autods_pipeline_output/06_reports/report.json") as f:
    report = json.load(f)

print(report["technical_report"])
print(report["business_report"])
print(report["planner_review"])   # Planner's modelling narrative
```

---

## 📋 Requirements

```
pandas>=1.3.0
scikit-learn>=0.24.0
lightgbm>=3.0.0
numpy>=1.20.0
python-dotenv>=0.21.0

# Optional — LLM features
anthropic>=0.20.0
langchain-openai>=0.1.0
langchain-core>=0.1.0
```

---

## ✅ Version History

**v0.2** (current)
- PlannerAgent `plan()` now accepts `extra_files` — pass JSON metadata, Excel data dictionaries, Markdown specs, or CSV reference tables as supplementary planning context
- PlannerAgent input normalisation: `business_description` is interpreted by a dedicated LLM pass before planning, supporting any language, informal phrasing, and non-technical terminology
- PlannerAgent `prepare_data()`: converts raw data files (Excel, JSON, TSV, CSV) into a pipeline-ready DataFrame, guided by an optional metadata file for column naming and value mapping; set `config.metadata_file` to enable
- CSV separator auto-detection: `config.csv_sep` defaults to `None`; the pipeline sniffs comma / semicolon / tab / pipe from the file and sets the separator automatically before any stage reads data; explicit override still supported
- Stage 6 report agent upgraded to `MultiAgentReportGenerator` (LLM-powered technical + business reports)
- `pos_label` auto-detection in ModellingAgent for non-numeric binary targets (`'e'`/`'p'`, `'0'`/`'1'`, etc.)
- Unified `utils.json_default` for numpy-type JSON serialisation across all agents

**v0.1**
- 7-stage pipeline with Planner Agent (Stage 0)
- LLM-driven config generation, adaptive replanning, post-modelling review
- DataCleaningAgent v2.0 with 13-step cleaning pipeline and `DataCleaningConfig`
- Anthropic Claude and OpenAI support; rule-based fallback throughout
- Lean project structure — superseded and one-off files removed

**v0.0**
- 6-stage pipeline (Understanding → Cleaning → Features → Modelling → Evaluation → Report)
- Standalone Stage 6 implementation with no LangChain dependency
- JSON serialisation support and comprehensive error handling

---

## 🎉 Feature Highlights

✅ **Plain-English Configuration** — describe your task in one sentence
✅ **Fully Automated** — minimal user intervention required
✅ **LLM-Enhanced** — smarter decisions at every stage with graceful fallback
✅ **Production Ready** — robust error handling, target-column protection, constraint validation
✅ **Comprehensive** — from raw data to business-ready report in one command
✅ **Flexible** — works with any tabular dataset; every threshold is configurable
✅ **Transparent** — detailed logs, per-stage JSON artifacts, and before/after quality metrics

---

*AutoDS — Making Data Science Accessible, Automated, and Reliable.*
