# DataCleaningAgent — v2.0

> Stage 2 of the AutoDS Pipeline
> File: `data_cleaning_agent.py`

---

## Overview

`DataCleaningAgent` is the data-cleaning stage of the AutoDS pipeline.
It accepts a raw `DataFrame` (or a CSV path) and returns a cleaned `DataFrame` together with a structured report.

**v2.0 changes at a glance**

| Category | What changed |
|----------|-------------|
| Interface | `run()` is now the main entry point; `execute()` kept as backward-compatible alias |
| Return type | `run()` returns a `Dict` instead of a bare `DataFrame` |
| Configuration | New `DataCleaningConfig` dataclass; no more hardcoded thresholds |
| Quality report | `before_cleaning` and `after_cleaning` metrics now use genuinely different DataFrames |
| Safety | Column-type lists are reset on every `run()` call |
| LLM | Optional OpenAI integration for column-type advice and imputation strategy |
| New features | 12 additional cleaning capabilities (see feature list below) |

---

## Quick Start

```python
from data_cleaning_agent import DataCleaningAgent, DataCleaningConfig

# Default config — no LLM, sensible thresholds
agent = DataCleaningAgent(name="Cleaner")
result = agent.run("data.csv")          # or pass a DataFrame directly

cleaned_df = result["data"]
print(result["summary"])
```

### With custom config

```python
config = DataCleaningConfig(
    target_column   = "churn",          # protect label column from anomaly removal
    anomaly_strategy= "clip",           # clip instead of deleting rows
    iqr_multiplier  = 2.0,              # wider IQR window
    missing_drop_threshold = 0.5,       # drop columns missing > 50 %
    column_constraints = {
        "age":   {"min": 0, "max": 120},
        "score": {"min": 0, "max": 100},
    },
)
agent  = DataCleaningAgent(name="Cleaner", config=config)
result = agent.run(df)
```

### With LLM assistance

```python
# Requires OPENAI_API_KEY in environment / .env file
config = DataCleaningConfig(
    use_llm_column_advisor = True,
    llm_model              = "gpt-4o-mini",
    llm_sample_size        = 5,
)
agent  = DataCleaningAgent(config=config)
result = agent.run(df)
```

---

## Configuration Reference — `DataCleaningConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | `str` | `"./outputs"` | Directory for JSON report |
| `special_format_match_threshold` | `float` | `0.80` | Fraction of values that must match a special-format pattern for a column to be classified as such |
| `numeric_conversion_threshold` | `float` | `0.80` | Fraction of non-null values that must convert successfully to numeric |
| `iqr_multiplier` | `float` | `1.50` | Multiplier applied to IQR for anomaly bounds (Q1 − k·IQR, Q3 + k·IQR) |
| `id_unique_ratio_threshold` | `float` | `0.95` | Columns whose non-null unique ratio exceeds this are treated as ID columns and skipped |
| `missing_drop_threshold` | `float` | `0.60` | Columns with a missing rate above this are dropped entirely |
| `anomaly_strategy` | `str` | `"remove"` | `"remove"` deletes rows outside bounds; `"clip"` replaces them with the bound value |
| `target_column` | `str \| None` | `None` | Label column; excluded from anomaly removal |
| `column_constraints` | `dict` | `{}` | Per-column numeric bounds: `{"col": {"min": 0, "max": 100}}` |
| `use_llm_column_advisor` | `bool` | `False` | Enable LLM for ambiguous column classification and imputation planning |
| `llm_model` | `str` | `"gpt-4o-mini"` | OpenAI model used when LLM is enabled |
| `llm_temperature` | `float` | `0.0` | LLM sampling temperature (0 = deterministic) |
| `llm_sample_size` | `int` | `5` | Number of sample values shown to the LLM per column |

---

## Cleaning Pipeline — Step by Step

The pipeline executes the following steps in order:

```
Input (CSV path or DataFrame)
│
├─ [F11] Step 0  Column-name normalisation
│          strip · lowercase · snake_case
│
├─ [F4]  Step 1  Pseudo-null unification
│          "None" / "nan" / "N/A" / "" / "-" / … → NaN
│
├─       Step 2  Column-type identification
│          ID · special-format · numeric · datetime · boolean · categorical
│          (LLM used for ambiguous columns when enabled)
│
├─ [F1]  Step 3  Constant-column removal
│          unique non-null values == 1 → dropped
│
├─ [F3]  Step 4  Duplicate-column removal
│          identical content → second occurrence dropped
│
├─ [F5]  Step 5  High-missing-rate column removal
│          missing rate > missing_drop_threshold → dropped
│          (target_column and ID columns are protected)
│
├─       Step 6  Duplicate-row removal
│
├─ [F6]  Step 7  Missing-value imputation
│          numeric  → median / mean / zero / drop_row  (LLM plan or default median)
│          categorical → mode fill, or "MISSING" if no mode exists
│
├─ [F7]  Step 8  Anomaly handling  (IQR-based)
│  [F8]          target_column is always skipped
│                anomaly_strategy = "remove" → drop rows
│                anomaly_strategy = "clip"   → clip to bounds
│
├─ [F9]  Step 9  Datetime normalisation
│          parsed strings → YYYY-MM-DD
│
├─ [F10] Step 10 Boolean unification
│          yes/no/true/false/y/n/t/f → "0" / "1"
│
├─       Step 11 Special-format preservation
│          trim only; no lowercase, no transformation
│
├─       Step 12 Categorical consistency
│          strip + lowercase
│
└─ [F12] Step 13 Numeric range constraints
           rows outside {min, max} per column → dropped
```

---

## Column Types Recognised

| Type | Detection logic |
|------|----------------|
| **ID** | `nunique / n_non_null ≥ id_unique_ratio_threshold` |
| **Special format** | Regex patterns: CN/US phone, CN postal code, CN ID card, bank card, order numbers |
| **Numeric** | `dtype` already numeric, or converts successfully at rate ≥ threshold |
| **Datetime** | `pd.to_datetime` succeeds on ≥ 80 % of a 20-row sample |
| **Boolean** | All non-null values are a subset of `{yes, no, true, false, y, n, t, f, 0, 1}` |
| **Categorical** | Fallback; or LLM classification if `use_llm_column_advisor=True` |

Pseudo-null patterns unified to `NaN` before type detection:

```
"None"  "none"  "nan"  "NaN"  "N/A"  "n/a"  "NA"  "na"
"null"  "NULL"  "-"    ""     "unknown"  "missing"  "undefined"
"nd"    "not available"  "not applicable"
```

---

## API Reference

### `DataCleaningAgent(name, config, output_dir)`

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `name` | `str` | `"DataCleaner"` | Used in log messages |
| `config` | `DataCleaningConfig \| None` | `None` → defaults | Full configuration object |
| `output_dir` | `str \| None` | `None` | Legacy shortcut; overrides `config.output_dir` |

---

### `run(input_data) → Dict`

Main entry point.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `input_data` | `str \| pd.DataFrame` | CSV file path or in-memory DataFrame |

**Returns**

```python
{
    "status":     "success" | "failure",
    "data":       pd.DataFrame,          # cleaned DataFrame
    "summary":    dict,                  # shape, column lists, retention %
    "report":     dict,                  # full JSON-serialisable report
    "agent_name": str,
    # on failure only:
    "error_message": str,
}
```

---

### `execute(input_data) → pd.DataFrame`

Backward-compatible alias for `run()`.
Returns only the cleaned `DataFrame`; raises `RuntimeError` on failure.

```python
# Still works — no changes needed in existing code
cleaned_df = agent.execute(raw_df)
```

---

### `get_summary() → dict`

Returns a summary of the last `run()` call.

```python
{
    "original_shape":         (100, 12),
    "final_shape":            (94, 10),
    "rows_removed":           6,
    "dropped_columns":        ["id", "constant_col"],
    "numeric_columns":        ["age", "income"],
    "categorical_columns":    ["city", "gender"],
    "special_format_columns": ["phone"],
    "datetime_columns":       ["signup_date"],
    "boolean_columns":        ["is_active"],
    "data_quality": {
        "rows_retained_pct": 94.0,
        "rows_removed_pct":  6.0,
    },
}
```

---

### `get_cleaning_report() → dict`

Returns the full cleaning report (also saved to `cleaning_report.json`).

```python
{
    "metadata": { "agent_name": ..., "timestamp": ..., "version": "2.0" },
    "input_data":  { "rows": 100, "columns": 12 },
    "output_data": { "rows": 94,  "columns": 10 },
    "cleaning_summary": {
        "rows_removed": 6,
        "columns_dropped": ["id", "constant_col"],
        "columns_dropped_cnt": 2,
        "data_retention_pct": 94.0,
    },
    "column_classification": { ... },
    "data_quality_metrics": {
        "before_cleaning": {
            "completeness_pct": 91.2,
            "null_count": 88,
            "null_pct": 8.8,
            "duplicate_rows": 2,
        },
        "after_cleaning": {
            "completeness_pct": 100.0,
            "null_count": 0,
            "null_pct": 0.0,
            "duplicate_rows": 0,
        },
    },
    "cleaning_operations": {
        "anomaly_strategy": "remove",
        "iqr_multiplier": 1.5,
        "llm_enabled": false,
    },
}
```

---

## LLM Integration

When `use_llm_column_advisor = True`, the agent makes at most **two** LLM calls per `run()`:

| Call | When | What is sent | What is returned |
|------|------|-------------|-----------------|
| Column-type advice | Per ambiguous column | Column name, sample values, dtype, unique count | `"special_format"` / `"numeric"` / `"categorical"` |
| Imputation plan | Once before Step 7 | Column name, missing %, sample values | JSON map `{"col": "median"\|"mean"\|"zero"\|"drop_row"}` |

Both calls have full fallback: if the LLM is unavailable or returns invalid output, the agent silently falls back to rule-based defaults.

**Setup**

```bash
# Install dependency
pip install langchain-openai python-dotenv

# Create .env file in project root
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## Pipeline Integration

`DataCleaningAgent` is called in Stage 2 of `autods_implementation_guide.py`.
Because `execute()` still returns a bare `DataFrame`, no changes to the Pipeline code are needed when upgrading from v1.

```python
# autods_implementation_guide.py — Stage 2 (unchanged)
agent = DataCleaningAgent(
    name="DataCleaner",
    output_dir=str(self.config.stage_dirs[2])
)
cleaned_data = agent.execute(raw_data)       # backward-compatible ✓
report       = agent.get_cleaning_report()   # new in v2.0
```

To pass a custom config (e.g., from the Planner Agent):

```python
from data_cleaning_agent import DataCleaningAgent, DataCleaningConfig

config = DataCleaningConfig(
    target_column    = self.config.target_column,
    anomaly_strategy = "clip",
    output_dir       = str(self.config.stage_dirs[2]),
)
agent = DataCleaningAgent(name="DataCleaner", config=config)
result = agent.run(raw_data)
cleaned_data = result["data"]
```

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `cleaning_report.json` | `config.output_dir` | Full structured cleaning report |

---

## Version History

| Version | Changes |
|---------|---------|
| **v2.0** | `DataCleaningConfig`; `run()` returns Dict; 12 new cleaning features; LLM integration; `before_cleaning` / `after_cleaning` quality metrics fixed |
| **v1.0** | Initial release — `execute()` returning bare DataFrame, median-only imputation, phone-only special-format detection |
