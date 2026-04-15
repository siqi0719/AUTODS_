"""
DataCleaningAgent  v2.0
=======================
PDF Issues Fixed:
  [P1] execute() renamed to run(); execute() kept as backward-compatible alias
  [P2] run() returns Dict instead of bare DataFrame
  [P3] DataCleaningConfig dataclass; hardcoded thresholds removed
  [P4] original_df saved at entry; before/after quality metrics are now different
  [P5] column-type lists reset at the top of every run() call
  [P6] Optional LLM for column-type identification and imputation strategy

New Features Added:
  [F1]  Constant-column detection and removal
  [F2]  ID-column detection and exclusion from all cleaning steps
  [F3]  Duplicate-column detection and removal
  [F4]  Pseudo-null unification ("None", "nan", "N/A", "", "-", …)
  [F5]  High-missing-rate column removal (configurable threshold)
  [F6]  Categorical missing-value imputation (mode or "MISSING")
  [F7]  Anomaly handling strategy: "remove" (default) or "clip"
  [F8]  Target-column protection (skip anomaly removal on label column)
  [F9]  Datetime column detection and ISO-8601 normalisation
  [F10] Boolean column unification → "0"/"1"
  [F11] Column-name normalisation (lowercase, strip, snake_case)
  [F12] Numeric range constraint validation (min / max per column)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ============================================================================
# Config
# ============================================================================

@dataclass
class DataCleaningConfig:
    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "./outputs"

    # ── Thresholds (P3) ───────────────────────────────────────────────────────
    special_format_match_threshold: float = 0.80   # line 131 original
    numeric_conversion_threshold:   float = 0.80   # line 152 original
    iqr_multiplier:                 float = 3.00   # wider bounds — 1.5 removes too many rows across many columns

    # ── F2: ID-column detection ───────────────────────────────────────────────
    id_unique_ratio_threshold: float = 0.95

    # ── F5: High-missing-rate column removal ──────────────────────────────────
    missing_drop_threshold: float = 0.60

    # ── F7: Anomaly handling strategy ─────────────────────────────────────────
    anomaly_strategy: str = "clip"     # "remove" | "clip" — clip is safer for wide datasets
    # Safety: if "remove" would delete more than this fraction, fall back to clip
    max_anomaly_remove_ratio: float = 0.30

    # ── F8: Target-column protection ──────────────────────────────────────────
    target_column: Optional[str] = None

    # ── F12: Per-column numeric constraints {"col": {"min": 0, "max": 150}} ───
    column_constraints: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── P6 / LLM settings ────────────────────────────────────────────────────
    use_llm_column_advisor: bool = False
    llm_model:              str  = "gpt-4o-mini"
    llm_temperature:        float = 0.0
    llm_sample_size:        int   = 5


# ============================================================================
# Agent
# ============================================================================

class DataCleaningAgent:
    """
    Automated Data Cleaning Agent — v2.0

    Cleaning pipeline (in execution order):
      0.  Column-name normalisation          [F11]
      1.  Pseudo-null unification            [F4]
      2.  Column-type identification         [P3, P6, F2, F9, F10]
      3.  Constant-column removal            [F1]
      4.  Duplicate-column removal           [F3]
      5.  High-missing-rate column removal   [F5]
      6.  Duplicate-row removal
      7.  Missing-value imputation           [P4, P6, F6]
      8.  Anomaly handling (IQR)             [P3, F7, F8]
      9.  Datetime normalisation             [F9]
      10. Boolean unification                [F10]
      11. Special-format preservation        [P3, P6]
      12. Categorical consistency
      13. Numeric range constraints          [F12]
    """

    # Strings treated as missing
    _PSEUDO_NULLS: frozenset = frozenset({
        "none", "nan", "n/a", "na", "null", "-", "", "unknown",
        "missing", "undefined", "nd", "not available", "not applicable",
    })

    # Boolean value mapping → "0" / "1"
    _BOOL_MAP: Dict[str, str] = {
        "yes": "1", "no": "0",
        "true": "1", "false": "0",
        "y": "1", "n": "0",
        "t": "1", "f": "0",
        "1": "1", "0": "0",
    }

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str = "DataCleaner",
        config: Optional[DataCleaningConfig] = None,
        # backward-compatible positional kwarg
        output_dir: Optional[str] = None,
    ):
        self.name   = name
        self.config = config or DataCleaningConfig()

        # backward-compat: if output_dir passed directly, honour it
        if output_dir is not None:
            self.config.output_dir = output_dir

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Column-classification lists (reset in run())  [P5]
        self.id_cols:         List[str] = []
        self.constant_cols:   List[str] = []
        self.numeric_cols:    List[str] = []
        self.categorical_cols:List[str] = []
        self.special_cols:    List[str] = []
        self.datetime_cols:   List[str] = []
        self.bool_cols:       List[str] = []
        self.dropped_cols:    List[str] = []

        self.execution_log:  List[Dict[str, Any]] = []
        self.cleaning_report: Dict[str, Any]      = {}

        # LLM (P6)
        self.llm = self._init_llm()

    # ================================================================== #
    # LLM initialisation (P6)
    # ================================================================== #

    def _init_llm(self):
        if not self.config.use_llm_column_advisor:
            return None
        from utils import build_chat_llm
        llm = build_chat_llm(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
        )
        if llm is None:
            print(f"[{self.name}] No LLM available — column advisor disabled.")
        return llm

    # ================================================================== #
    # Main entry point  (P1, P2)
    # ================================================================== #

    def run(self, input_data: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Execute the full cleaning pipeline.

        Returns
        -------
        dict with keys:
            status      : "success" | "failure"
            data        : cleaned pd.DataFrame
            summary     : dict (shape, counts, …)
            report      : detailed JSON-serialisable report
            agent_name  : str
        """
        # P5 — reset column-type lists at the top of every call
        self.id_cols          = []
        self.constant_cols    = []
        self.numeric_cols     = []
        self.categorical_cols = []
        self.special_cols     = []
        self.datetime_cols    = []
        self.bool_cols        = []
        self.dropped_cols     = []

        try:
            # ── Load ──────────────────────────────────────────────────
            if isinstance(input_data, str):
                df = pd.read_csv(input_data)
            else:
                df = input_data.copy()

            print(f"\n[{self.name}] ══ Data Cleaning Start ══")
            print(f"[{self.name}] Input  : {df.shape[0]} rows × {df.shape[1]} cols")

            original_shape = df.shape
            original_df    = df.copy()   # P4 — preserve for before/after comparison

            # ── Step 0  Column-name normalisation  [F11] ──────────────
            df = self._normalize_column_names(df)

            # ── Step 1  Pseudo-null unification  [F4] ─────────────────
            df = self._replace_pseudo_nulls(df)

            # ── Step 2  Identify column types ─────────────────────────
            self._identify_column_types(df)

            # ── Step 3  Drop constant columns  [F1] ───────────────────
            df = self._drop_constant_columns(df)

            # ── Step 4  Drop duplicate columns  [F3] ──────────────────
            df = self._drop_duplicate_columns(df)

            # ── Step 5  Drop high-missing-rate columns  [F5] ──────────
            df = self._drop_high_missing_columns(df)

            # ── Step 6  Remove duplicate rows ─────────────────────────
            df = self._remove_duplicates(df)

            # ── Step 7  Missing-value imputation  [P6, F6] ────────────
            df = self._handle_missing_values(df)

            # ── Step 8  Anomaly handling  [P3, F7, F8] ────────────────
            df = self._handle_anomalies(df)

            # ── Step 9  Datetime normalisation  [F9] ──────────────────
            df = self._normalize_datetime_columns(df)

            # ── Step 10 Boolean unification  [F10] ────────────────────
            df = self._unify_boolean_columns(df)

            # ── Step 11 Preserve special formats ──────────────────────
            df = self._preserve_special_formats(df)

            # ── Step 12 Categorical consistency ───────────────────────
            df = self._ensure_consistency(df)

            # ── Step 13 Numeric range constraints  [F12] ──────────────
            df = self._apply_column_constraints(df)

            print(f"\n[{self.name}] Output : {df.shape[0]} rows × {df.shape[1]} cols")
            print(f"[{self.name}] Rows removed   : {original_shape[0] - df.shape[0]}")
            print(f"[{self.name}] Cols removed   : {len(self.dropped_cols)}")
            print(f"[{self.name}] ══ Cleaning Complete ══")

            # ── Log & report ──────────────────────────────────────────
            self.execution_log.append({
                "original_shape":    original_shape,
                "final_shape":       df.shape,
                "rows_removed":      original_shape[0] - df.shape[0],
                "dropped_columns":   list(self.dropped_cols),
                "id_cols":           list(self.id_cols),
                "numeric_cols":      list(self.numeric_cols),
                "categorical_cols":  list(self.categorical_cols),
                "special_cols":      list(self.special_cols),
                "datetime_cols":     list(self.datetime_cols),
                "bool_cols":         list(self.bool_cols),
            })

            # P4 — pass original_df so metrics differ
            self._generate_json_report(original_shape, df, original_df)

            return {
                "status":     "success",
                "data":       df,
                "summary":    self.get_summary(),
                "report":     self.cleaning_report,
                "agent_name": self.name,
            }

        except Exception as exc:
            import traceback
            print(f"[{self.name}] ✗ Cleaning failed: {exc}")
            traceback.print_exc()
            return {
                "status":       "failure",
                "error_message": str(exc),
                "agent_name":   self.name,
            }

    def execute(self, input_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Backward-compatible alias — returns only the cleaned DataFrame."""
        result = self.run(input_data)
        if result["status"] == "failure":
            raise RuntimeError(result.get("error_message", "Cleaning failed"))
        return result["data"]

    # ================================================================== #
    # F11  Column-name normalisation
    # ================================================================== #

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n[{self.name}] [F11] Normalising column names…")
        old = df.columns.tolist()
        new = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"[^\w]+", "_", regex=True)
            .str.replace(r"_+",     "_", regex=True)
            .str.strip("_")
        )
        df.columns = new
        changed = [(o, n) for o, n in zip(old, new) if o != n]
        for o, n in changed:
            print(f"  rename : '{o}' → '{n}'")
        if not changed:
            print("  ✓ All column names already normalised")
        # Keep target_column in sync if it was renamed
        if self.config.target_column:
            rename_map = {o: n for o, n in changed}
            if self.config.target_column in rename_map:
                self.config.target_column = rename_map[self.config.target_column]
        return df

    # ================================================================== #
    # F4  Pseudo-null unification
    # ================================================================== #

    def _replace_pseudo_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n[{self.name}] [F4] Replacing pseudo-null values…")
        before = int(df.isna().sum().sum())

        def _replace_cell(val):
            if isinstance(val, str) and val.strip().lower() in self._PSEUDO_NULLS:
                return np.nan
            return val

        df = df.map(_replace_cell) if hasattr(df, "map") else df.applymap(_replace_cell)
        after = int(df.isna().sum().sum())
        converted = after - before
        print(f"  ✓ {converted} pseudo-null cell(s) converted to NaN"
              if converted else "  ✓ No pseudo-null values found")
        return df

    # ================================================================== #
    # Step 2  Column-type identification  (P3, P6, F2, F9, F10)
    # ================================================================== #

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        print(f"\n[{self.name}] Identifying column types…")
        for col in df.columns:
            series   = df[col]
            non_null = series.dropna()
            n_non    = len(non_null)

            # ── F2  ID column ──────────────────────────────────────────
            if n_non > 0 and non_null.nunique() / n_non >= self.config.id_unique_ratio_threshold:
                self.id_cols.append(col)
                print(f"  {col:30s} → ID           (skipped in all cleaning steps)")
                continue

            # ── Special format ─────────────────────────────────────────
            if self._is_special_format(series):
                self.special_cols.append(col)
                print(f"  {col:30s} → SPECIAL FORMAT")
                continue

            # ── Already numeric dtype ──────────────────────────────────
            if series.dtype in ("float64", "int64", "float32", "int32"):
                self.numeric_cols.append(col)
                print(f"  {col:30s} → NUMERIC")
                continue

            # ── F9  Datetime ───────────────────────────────────────────
            if self._can_parse_as_datetime(series):
                self.datetime_cols.append(col)
                print(f"  {col:30s} → DATETIME")
                continue

            # ── F10 Boolean ────────────────────────────────────────────
            if self._is_boolean_col(series):
                self.bool_cols.append(col)
                print(f"  {col:30s} → BOOLEAN")
                continue

            # ── Convertible to numeric ─────────────────────────────────
            if self._can_convert_to_numeric(series):
                self.numeric_cols.append(col)
                print(f"  {col:30s} → NUMERIC (convertible)")
                continue

            # ── Ambiguous: ask LLM or default to categorical ───────────
            if self.config.use_llm_column_advisor and self.llm is not None:
                col_type = self._ask_llm_column_type(col, series)
                tag = col_type.upper() + " (LLM)"
            else:
                col_type = "categorical"
                tag = "CATEGORICAL"

            {"special_format": self.special_cols,
             "numeric":        self.numeric_cols,
             "categorical":    self.categorical_cols}.get(
                col_type, self.categorical_cols).append(col)
            print(f"  {col:30s} → {tag}")

    # -- helpers -----------------------------------------------------------

    def _is_special_format(self, series: pd.Series) -> bool:
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        patterns = (
            r"^1[3-9]\d{9}$",             # CN mobile
            r"^\+?1?\d{10,11}$",           # US phone
            r"^\d{3}-\d{3}-\d{4}$",        # US formatted phone
            r"^\d{3} \d{3} \d{4}$",        # US spaced phone
            r"^\d{6}$",                    # CN postal
            r"^\d{15}(\d{3}[Xx]?)?$",      # CN ID card
            r"^\d{16,19}$",                # bank card
            r"^[A-Z0-9]{6,20}$",           # order / serial
        )
        combined = "|".join(f"(?:{p})" for p in patterns)
        matches = sum(
            1 for v in non_null.astype(str)
            if re.match(combined, str(v).strip())
        )
        return matches / len(non_null) > self.config.special_format_match_threshold

    def _can_parse_as_datetime(self, series: pd.Series) -> bool:
        if series.dtype in ("float64", "int64", "float32", "int32"):
            return False
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        sample = non_null.head(20).astype(str)
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            return parsed.notna().mean() >= 0.80
        except Exception:
            return False

    def _is_boolean_col(self, series: pd.Series) -> bool:
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        unique_lower = set(non_null.astype(str).str.strip().str.lower().unique())
        return unique_lower <= set(self._BOOL_MAP.keys()) and len(unique_lower) <= 2

    def _can_convert_to_numeric(self, series: pd.Series) -> bool:
        if series.dtype in ("float64", "int64", "float32", "int32"):
            return True
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        try:
            converted = pd.to_numeric(non_null, errors="coerce")
            return converted.notna().sum() / len(non_null) > self.config.numeric_conversion_threshold
        except Exception:
            return False

    def _ask_llm_column_type(self, col_name: str, series: pd.Series) -> str:
        """P6 — ask LLM to classify an ambiguous column."""
        try:
            from langchain_core.messages import HumanMessage
            samples = series.dropna().head(self.config.llm_sample_size).tolist()
            prompt = (
                f"Column name: '{col_name}'\n"
                f"Sample values: {samples}\n"
                f"dtype: {series.dtype}, unique count: {series.nunique()}\n\n"
                "Classify this column as exactly one of:\n"
                "  special_format  — IDs, codes, phone/postal numbers, serial numbers\n"
                "  numeric         — continuous or discrete numbers\n"
                "  categorical     — text labels, names, categories\n"
                "Reply with ONLY one word."
            )
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            result = resp.content.strip().lower()
            return result if result in ("special_format", "numeric", "categorical") else "categorical"
        except Exception:
            return "categorical"

    # ================================================================== #
    # F1  Drop constant columns
    # ================================================================== #

    def _drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n[{self.name}] [F1] Detecting constant columns…")
        to_drop = [
            col for col in df.columns
            if col not in self.id_cols
            and col != self.config.target_column
            and df[col].dropna().nunique() <= 1
        ]
        if to_drop:
            self.constant_cols.extend(to_drop)
            self.dropped_cols.extend(to_drop)
            df = df.drop(columns=to_drop)
            self._remove_from_type_lists(to_drop)
            print(f"  ✓ Dropped {len(to_drop)} constant column(s): {to_drop}")
        else:
            print("  ✓ No constant columns")
        return df

    # ================================================================== #
    # F3  Drop duplicate columns
    # ================================================================== #

    def _drop_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n[{self.name}] [F3] Detecting duplicate columns…")
        seen: Dict[int, str] = {}
        to_drop: List[str]   = []
        for col in df.columns:
            h = hash(tuple(pd.util.hash_array(df[col].values)))
            if h in seen:
                to_drop.append(col)
                print(f"  duplicate: '{col}' == '{seen[h]}' — dropping")
            else:
                seen[h] = col
        if to_drop:
            self.dropped_cols.extend(to_drop)
            df = df.drop(columns=to_drop)
            self._remove_from_type_lists(to_drop)
        else:
            print("  ✓ No duplicate columns")
        return df

    # ================================================================== #
    # F5  Drop high-missing-rate columns
    # ================================================================== #

    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        thr = self.config.missing_drop_threshold
        print(f"\n[{self.name}] [F5] Dropping columns with missing rate > {thr:.0%}…")
        missing_rates = {
            col: df[col].isna().mean()
            for col in df.columns
            if col not in self.id_cols and col != self.config.target_column
        }
        to_drop = [col for col, rate in missing_rates.items() if rate > thr]
        if to_drop:
            for col in to_drop:
                print(f"  drop: '{col}'  missing={missing_rates[col]:.1%}")
            self.dropped_cols.extend(to_drop)
            df = df.drop(columns=to_drop)
            self._remove_from_type_lists(to_drop)
        else:
            print("  ✓ All columns within threshold")
        return df

    # ================================================================== #
    # Step 6  Remove duplicate rows
    # ================================================================== #

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n[{self.name}] Removing duplicate rows…")
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        print(f"  ✓ Removed {removed} duplicate row(s)" if removed else "  ✓ No duplicate rows")
        return df

    # ================================================================== #
    # Step 7  Missing-value imputation  (P4, P6, F6)
    # ================================================================== #

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n[{self.name}] Handling missing values…")

        # LLM imputation plan for numeric columns (P6)
        imputation_plan = self._get_llm_imputation_plan(df) if self.llm is not None else {}

        # ── Numeric columns ───────────────────────────────────────────
        for col in self.numeric_cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
            # Convert if needed
            if df[col].dtype not in ("float64", "int64", "float32", "int32"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].dtype not in ("float64", "int64", "float32", "int32"):
                continue

            n_miss   = int(df[col].isna().sum())
            strategy = imputation_plan.get(col, "median")

            if strategy == "mean":
                v = df[col].mean();  df[col] = df[col].fillna(v)
                print(f"  ✓ {col}: filled {n_miss} NaN → mean ({v:.4g})")
            elif strategy == "zero":
                df[col] = df[col].fillna(0)
                print(f"  ✓ {col}: filled {n_miss} NaN → 0")
            elif strategy == "drop_row":
                df = df.dropna(subset=[col])
                print(f"  ✓ {col}: dropped {n_miss} rows with NaN")
            else:                                   # "median" (default)
                v = df[col].median(); df[col] = df[col].fillna(v)
                print(f"  ✓ {col}: filled {n_miss} NaN → median ({v:.4g})")

        # ── F6  Categorical columns ───────────────────────────────────
        for col in self.categorical_cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
            n_miss = int(df[col].isna().sum())
            modes  = df[col].mode()
            if len(modes) > 0:
                df[col] = df[col].fillna(modes[0])
                print(f"  ✓ {col}: filled {n_miss} NaN → mode ('{modes[0]}')")
            else:
                df[col] = df[col].fillna("MISSING")
                print(f"  ✓ {col}: filled {n_miss} NaN → 'MISSING'")

        return df

    def _get_llm_imputation_plan(self, df: pd.DataFrame) -> Dict[str, str]:
        """P6 — ask LLM which imputation strategy to use per numeric column."""
        try:
            from langchain_core.messages import HumanMessage
            cols_info: List[str] = []
            for col in self.numeric_cols:
                if col not in df.columns:
                    continue
                rate = df[col].isna().mean()
                if rate == 0:
                    continue
                samples = df[col].dropna().head(self.config.llm_sample_size).tolist()
                cols_info.append(f"- {col}: missing={rate:.1%}, samples={samples}")
            if not cols_info:
                return {}
            prompt = (
                "For each column below choose ONE imputation strategy.\n"
                "Options: median  mean  zero  drop_row\n"
                "Return ONLY a JSON object like {\"col\": \"strategy\"}.\n\n"
                + "\n".join(cols_info)
            )
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            raw  = resp.content.strip()
            # Extract JSON from potential markdown fences
            match = re.search(r"\{[\s\S]*\}", raw)
            plan  = json.loads(match.group() if match else raw)
            valid = {"median", "mean", "zero", "drop_row"}
            return {k: v for k, v in plan.items() if v in valid}
        except Exception:
            return {}

    # ================================================================== #
    # Step 8  Anomaly handling  (P3, F7, F8)
    # ================================================================== #

    def _handle_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.anomaly_strategy
        print(f"\n[{self.name}] Handling anomalies (strategy='{strategy}')…")
        rows_before = len(df)

        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            # F8 — skip target column
            if col == self.config.target_column:
                print(f"  skip : {col} (target column protected)")
                continue
            if df[col].dtype not in ("float64", "int64", "float32", "int32"):
                df[col] = pd.to_numeric(df[col], errors="coerce")

            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            if IQR == 0:
                continue

            lo = Q1 - self.config.iqr_multiplier * IQR
            hi = Q3 + self.config.iqr_multiplier * IQR
            mask = (df[col] < lo) | (df[col] > hi)
            n    = int(mask.sum())
            if n == 0:
                continue

            if strategy == "clip":                          # F7
                df[col] = df[col].clip(lower=lo, upper=hi)
                print(f"  clip : {col}  {n} value(s) → [{lo:.4g}, {hi:.4g}]")
            else:
                # Safety check: if removing would exceed max_anomaly_remove_ratio,
                # fall back to clipping to avoid catastrophic data loss.
                remove_ratio = n / len(df)
                if remove_ratio > self.config.max_anomaly_remove_ratio:
                    df[col] = df[col].clip(lower=lo, upper=hi)
                    print(f"  clip : {col}  {n} row(s) would exceed {self.config.max_anomaly_remove_ratio:.0%} limit → clipped instead")
                else:
                    df = df[~mask]
                    print(f"  drop : {col}  {n} row(s)  bounds=[{lo:.4g}, {hi:.4g}]")

        removed = rows_before - len(df)
        if removed:
            print(f"  Total rows removed by anomaly handling: {removed}")
        return df

    # ================================================================== #
    # F9  Datetime normalisation
    # ================================================================== #

    def _normalize_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.datetime_cols:
            return df
        print(f"\n[{self.name}] [F9] Normalising datetime columns…")
        for col in self.datetime_cols:
            if col not in df.columns:
                continue
            try:
                df[col] = pd.to_datetime(
                    df[col], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                print(f"  ✓ {col} → YYYY-MM-DD")
            except Exception as exc:
                print(f"  ⚠ {col}: normalisation failed ({exc})")
        return df

    # ================================================================== #
    # F10  Boolean unification
    # ================================================================== #

    def _unify_boolean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.bool_cols:
            return df
        print(f"\n[{self.name}] [F10] Unifying boolean columns…")
        for col in self.bool_cols:
            if col not in df.columns:
                continue
            df[col] = (
                df[col].astype(str).str.strip().str.lower()
                .map(self._BOOL_MAP)
            )
            print(f"  ✓ {col} → 0/1")
        return df

    # ================================================================== #
    # Step 11  Preserve special formats
    # ================================================================== #

    def _preserve_special_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.special_cols:
            return df
        print(f"\n[{self.name}] Preserving special-format columns…")
        for col in self.special_cols:
            if col not in df.columns:
                continue
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
                print(f"  ✓ {col}: trimmed only")
        return df

    # ================================================================== #
    # Step 12  Categorical consistency
    # ================================================================== #

    def _ensure_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.categorical_cols:
            return df
        print(f"\n[{self.name}] Ensuring categorical consistency…")
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.lower()
                print(f"  ✓ {col}: strip + lowercase")
        return df

    # ================================================================== #
    # F12  Numeric range constraints
    # ================================================================== #

    def _apply_column_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.column_constraints:
            return df
        print(f"\n[{self.name}] [F12] Applying numeric range constraints…")
        for col, bounds in self.config.column_constraints.items():
            if col not in df.columns:
                continue
            before  = len(df)
            lo, hi  = bounds.get("min"), bounds.get("max")
            if lo is not None:
                df = df[df[col].isna() | (df[col] >= lo)]
            if hi is not None:
                df = df[df[col].isna() | (df[col] <= hi)]
            removed = before - len(df)
            if removed:
                print(f"  ✓ {col}: removed {removed} out-of-range rows (min={lo}, max={hi})")
        return df

    # ================================================================== #
    # Reporting  (P4 fixed)
    # ================================================================== #

    def _generate_json_report(
        self,
        original_shape: tuple,
        cleaned_df:     pd.DataFrame,
        original_df:    pd.DataFrame,   # P4 — now a real original
    ) -> None:
        self.cleaning_report = {
            "metadata": {
                "agent_name": self.name,
                "timestamp":  datetime.now().isoformat(),
                "version":    "2.0",
            },
            "input_data": {
                "rows":    int(original_shape[0]),
                "columns": int(original_shape[1]),
            },
            "output_data": {
                "rows":    int(cleaned_df.shape[0]),
                "columns": int(cleaned_df.shape[1]),
            },
            "cleaning_summary": {
                "rows_removed":        int(original_shape[0] - cleaned_df.shape[0]),
                "columns_dropped":     list(self.dropped_cols),
                "columns_dropped_cnt": len(self.dropped_cols),
                "data_retention_pct":  round(
                    cleaned_df.shape[0] / original_shape[0] * 100, 2
                ) if original_shape[0] else 0,
            },
            "column_classification": {
                "id_columns":            self.id_cols,
                "numeric_columns":       self.numeric_cols,
                "categorical_columns":   self.categorical_cols,
                "special_format_columns":self.special_cols,
                "datetime_columns":      self.datetime_cols,
                "boolean_columns":       self.bool_cols,
                "constant_columns":      self.constant_cols,
                "dropped_columns":       self.dropped_cols,
            },
            # P4 fixed — original_df vs cleaned_df (used to be both cleaned_df)
            "data_quality_metrics": {
                "before_cleaning": self._calculate_quality_metrics(original_df),
                "after_cleaning":  self._calculate_quality_metrics(cleaned_df),
            },
            "cleaning_operations": {
                "anomaly_strategy": self.config.anomaly_strategy,
                "iqr_multiplier":   self.config.iqr_multiplier,
                "llm_enabled":      self.llm is not None,
            },
        }
        self._save_json_report()

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """P4 — now accepts any DataFrame, no unused data_type arg."""
        total = df.shape[0] * df.shape[1]
        return {
            "completeness_pct": round(df.notna().sum().sum() / total * 100, 2) if total else 0,
            "null_count":       int(df.isna().sum().sum()),
            "null_pct":         round(df.isna().sum().sum() / total * 100, 2) if total else 0,
            "duplicate_rows":   int(df.duplicated().sum()),
        }

    def _save_json_report(self) -> None:
        path = self.output_dir / "cleaning_report.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.cleaning_report, fh, indent=2, ensure_ascii=False, default=str)
        print(f"\n[{self.name}] ✓ Report → {path}")

    # ================================================================== #
    # Public accessors
    # ================================================================== #

    def get_cleaning_report(self) -> Dict[str, Any]:
        return self.cleaning_report

    def get_log(self) -> List[Dict[str, Any]]:
        return self.execution_log

    def get_summary(self) -> Dict[str, Any]:
        if not self.execution_log:
            return {}
        log = self.execution_log[-1]
        n_orig = log["original_shape"][0]
        return {
            "original_shape":        log["original_shape"],
            "final_shape":           log["final_shape"],
            "rows_removed":          log["rows_removed"],
            "dropped_columns":       log["dropped_columns"],
            "numeric_columns":       log["numeric_cols"],
            "categorical_columns":   log["categorical_cols"],
            "special_format_columns":log["special_cols"],
            "datetime_columns":      log["datetime_cols"],
            "boolean_columns":       log["bool_cols"],
            "data_quality": {
                "rows_retained_pct": round(log["final_shape"][0] / n_orig * 100, 2) if n_orig else 0,
                "rows_removed_pct":  round(log["rows_removed"]   / n_orig * 100, 2) if n_orig else 0,
            },
        }

    # ================================================================== #
    # Helpers
    # ================================================================== #

    def _remove_from_type_lists(self, cols: List[str]) -> None:
        """Remove dropped columns from every type list."""
        for lst in (
            self.numeric_cols, self.categorical_cols, self.special_cols,
            self.datetime_cols, self.bool_cols,
        ):
            for col in cols:
                if col in lst:
                    lst.remove(col)

    def _print_column_info(self, df: pd.DataFrame) -> None:
        """Utility: print per-column dtype / null counts."""
        for col in df.columns:
            nn = df[col].notna().sum()
            nv = df[col].isna().sum()
            print(f"  {col}: dtype={df[col].dtype}  non_null={nn}  null={nv}")
