from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str
    missing_ratio: float
    nunique: int
    sample_values: List[str]
    dropped_reason: Optional[str] = None


@dataclass
class FeatureEngineeringConfig:
    target_column: str
    problem_type: str = "classification"
    task_description: str = "General predictive modelling"

    missing_drop_threshold: float = 0.60
    id_unique_ratio_threshold: float = 0.95
    text_unique_ratio_threshold: float = 0.85

    rare_category_threshold: float = 0.01
    rare_category_label: str = "__RARE__"

    parse_datetime_strings: bool = True
    scale_numeric: bool = True
    drop_high_correlation: bool = True
    correlation_threshold: float = 0.95

    save_artifacts: bool = False
    output_dir: str = "feature_outputs"

    random_state: int = 42

    use_llm_planner: bool = True
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.0


class RareCategoryGrouper:
    def __init__(self, threshold: float = 0.01, rare_label: str = "__RARE__"):
        self.threshold = threshold
        self.rare_label = rare_label
        self.frequent_categories_: Dict[str, set] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RareCategoryGrouper":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.frequent_categories_ = {}
        for col in X.columns:
            series = X[col].astype("string").fillna("__MISSING__")
            freq = series.value_counts(normalize=True, dropna=False)
            frequent = set(freq[freq >= self.threshold].index.tolist())
            self.frequent_categories_[col] = frequent
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()
        for col in X_out.columns:
            series = X_out[col].astype("string").fillna("__MISSING__")
            frequent = self.frequent_categories_.get(col, set())
            X_out[col] = series.apply(lambda v: v if v in frequent else self.rare_label)
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class DateTimeFeatureExtractor:
    def __init__(self):
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DateTimeFeatureExtractor":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_ if self.columns_ else None)

        parts: List[pd.DataFrame] = []
        for col in X.columns:
            dt = pd.to_datetime(X[col], errors="coerce")

            df_part = pd.DataFrame(
                {
                    f"{col}__year": dt.dt.year,
                    f"{col}__month": dt.dt.month,
                    f"{col}__day": dt.dt.day,
                    f"{col}__weekday": dt.dt.weekday,
                    f"{col}__quarter": dt.dt.quarter,
                    f"{col}__is_month_start": dt.dt.is_month_start.astype("float"),
                    f"{col}__is_month_end": dt.dt.is_month_end.astype("float"),
                }
            )
            parts.append(df_part)

        if not parts:
            return pd.DataFrame(index=X.index)

        result = pd.concat(parts, axis=1)
        result.index = X.index
        return result

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class TextStatsExtractor:
    def __init__(self):
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TextStatsExtractor":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_ if self.columns_ else None)

        parts: List[pd.DataFrame] = []
        for col in X.columns:
            s = X[col].astype("string").fillna("")
            df_part = pd.DataFrame(
                {
                    f"{col}__char_len": s.str.len(),
                    f"{col}__word_count": s.str.split().str.len(),
                    f"{col}__digit_count": s.str.count(r"\d"),
                    f"{col}__uppercase_count": s.str.count(r"[A-Z]"),
                }
            )
            parts.append(df_part)

        if not parts:
            return pd.DataFrame(index=X.index)

        result = pd.concat(parts, axis=1)
        result.index = X.index
        return result

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class FeatureEngineeringAgent:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.column_profiles_: List[ColumnProfile] = []
        self.drop_columns_: List[str] = []
        self.numeric_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.datetime_columns_: List[str] = []
        self.text_columns_: List[str] = []
        self.bool_columns_: List[str] = []
        self.constant_columns_: List[str] = []
        self.high_corr_drop_columns_: List[str] = []
        self.generated_feature_log_: List[Dict[str, Any]] = []
        self.llm_plan_: Dict[str, Any] = {}

        self.preprocessor_: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []
        self.summary_: Dict[str, Any] = {}

        try:
            from utils import load_project_env
            load_project_env(__file__)
        except Exception:
            load_dotenv()
        self.llm = self._init_llm()

    def _init_llm(self):
        if not self.config.use_llm_planner:
            return None
        from utils import build_chat_llm
        return build_chat_llm(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
        )

    def run(self, df: pd.DataFrame, fit: bool = True) -> Dict[str, Any]:
        self._validate_input(df)

        work_df = df.copy()
        y = work_df[self.config.target_column].copy()
        X = work_df.drop(columns=[self.config.target_column])

        if fit:
            self._profile_columns(X)
            X = self._drop_useless_columns(X)
            X = self._coerce_datetime_like_columns(X)
            self._assign_column_roles(X)

            if self.config.use_llm_planner:
                self.llm_plan_ = self._generate_llm_plan(X, y)

            X = self._apply_llm_plan(X, fit=True)
            X = self._drop_high_correlation_columns(X)

            self.preprocessor_ = self._build_preprocessor()
            X_transformed = self.preprocessor_.fit_transform(X)
            self.feature_names_ = self._get_feature_names()
            if not self.feature_names_:
               self.feature_names_ = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        else:
            if self.preprocessor_ is None:
                raise ValueError("Preprocessor has not been fitted. Run with fit=True first.")

            X = X.drop(columns=[c for c in self.drop_columns_ if c in X.columns], errors="ignore")
            X = X.drop(columns=[c for c in self.high_corr_drop_columns_ if c in X.columns], errors="ignore")
            X = self._coerce_datetime_like_columns(X, fit_mode=False)
            X = self._apply_saved_generated_features(X)
            X_transformed = self.preprocessor_.transform(X)

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=self.feature_names_,
            index=X.index,
        )

        result = {
            "status": "success",
            "X": X_transformed_df,
            "y": y,
            "feature_names": self.feature_names_,
            "summary": self._build_summary(),
            "metadata": self._build_metadata(),
            "preprocessor": self.preprocessor_,
            "llm_plan": self.llm_plan_,
        }

        if self.config.save_artifacts:
            self._save_artifacts(result)

        return result

    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.run(df, fit=False)

    def get_planner_payload(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a planner-friendly payload describing the feature engineering
        configuration, dataset profile, and optional LLM-driven feature plan.

        This method is intended for use by a PlannerAgent or pipeline orchestrator.
        """
        self._validate_input(df)

        work_df = df.copy()
        y = work_df[self.config.target_column].copy()
        X = work_df.drop(columns=[self.config.target_column])

        self._profile_columns(X)
        X = self._drop_useless_columns(X)
        X = self._coerce_datetime_like_columns(X)
        self._assign_column_roles(X)

        if self.config.use_llm_planner:
            llm_plan = self._generate_llm_plan(X, y)
        else:
            llm_plan = {
                "planner_status": "disabled_by_config",
                "actions": [],
                "reason": "LLM planner disabled in FeatureEngineeringConfig.",
            }

        self.llm_plan_ = llm_plan

        return {
            "feature_config": asdict(self.config),
            "dataset_profile": {
                "row_count": int(len(X)),
                "column_profiles": [asdict(p) for p in self.column_profiles_],
                "current_roles": {
                    "numeric": self.numeric_columns_,
                    "categorical": self.categorical_columns_,
                    "datetime": self.datetime_columns_,
                    "text": self.text_columns_,
                    "bool_as_numeric": self.bool_columns_,
                },
            },
            "llm_plan": llm_plan,
        }

    def _validate_input(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in DataFrame.")
        if len(df) == 0:
            raise ValueError("Input DataFrame is empty.")

    def _profile_columns(self, X: pd.DataFrame) -> None:
        profiles: List[ColumnProfile] = []
        n_rows = len(X)

        for col in X.columns:
            series = X[col]
            missing_ratio = float(series.isna().mean())
            nunique = int(series.nunique(dropna=True))
            sample_values = series.dropna().astype(str).head(5).tolist()

            inferred_type = self._infer_column_type(series, n_rows)
            profiles.append(
                ColumnProfile(
                    name=col,
                    inferred_type=inferred_type,
                    missing_ratio=missing_ratio,
                    nunique=nunique,
                    sample_values=sample_values,
                )
            )

        self.column_profiles_ = profiles

    def _infer_column_type(self, series: pd.Series, n_rows: int) -> str:
        non_null = series.dropna()

        if is_bool_dtype(series):
            return "bool"

        if is_datetime64_any_dtype(series):
            return "datetime"

        if is_numeric_dtype(series):
            unique_ratio = non_null.nunique() / max(len(non_null), 1)
            if unique_ratio > self.config.id_unique_ratio_threshold and non_null.nunique() > 20:
                return "id_like"
            return "numeric"

        if is_object_dtype(series) or is_string_dtype(series):
            s = non_null.astype(str)

            if self.config.parse_datetime_strings and len(s) > 0:
                parsed = pd.to_datetime(s, errors="coerce")
                parse_ratio = parsed.notna().mean()
                if parse_ratio >= 0.80:
                    return "datetime"

            unique_ratio = s.nunique() / max(len(s), 1)
            avg_len = s.str.len().mean() if len(s) else 0.0

            if unique_ratio > self.config.id_unique_ratio_threshold and s.nunique() > 20 and avg_len < 40:
                return "id_like"

            if avg_len > 30 and unique_ratio > self.config.text_unique_ratio_threshold:
                return "text"

            return "categorical"

        return "categorical"

    def _drop_useless_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        drop_cols: List[str] = []
        constant_cols: List[str] = []
        profile_map = {p.name: p for p in self.column_profiles_}

        for col in X.columns:
            series = X[col]
            nunique = series.nunique(dropna=True)

            if nunique <= 1:
                constant_cols.append(col)
                drop_cols.append(col)
                profile_map[col].dropped_reason = "constant_or_single_value"
                continue

            missing_ratio = float(series.isna().mean())
            if missing_ratio >= self.config.missing_drop_threshold:
                drop_cols.append(col)
                profile_map[col].dropped_reason = "high_missing_ratio"
                continue

            inferred_type = profile_map[col].inferred_type
            if inferred_type == "id_like":
                drop_cols.append(col)
                profile_map[col].dropped_reason = "id_like_column"
                continue

        self.constant_columns_ = constant_cols
        self.drop_columns_ = drop_cols
        return X.drop(columns=drop_cols, errors="ignore")

    def _coerce_datetime_like_columns(self, X: pd.DataFrame, fit_mode: bool = True) -> pd.DataFrame:
        X = X.copy()

        if fit_mode:
            datetime_candidates = []
            for p in self.column_profiles_:
                if p.name in X.columns and p.inferred_type == "datetime":
                    datetime_candidates.append(p.name)
            self.datetime_columns_ = datetime_candidates

        for col in self.datetime_columns_:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors="coerce")

        return X

    def _assign_column_roles(self, X: pd.DataFrame) -> None:
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        self.text_columns_ = []
        self.bool_columns_ = []

        profile_map = {p.name: p for p in self.column_profiles_}

        for col in X.columns:
            inferred_type = profile_map[col].inferred_type

            if inferred_type == "numeric":
                self.numeric_columns_.append(col)
            elif inferred_type == "categorical":
                self.categorical_columns_.append(col)
            elif inferred_type == "text":
                self.text_columns_.append(col)
            elif inferred_type == "bool":
                self.bool_columns_.append(col)

        for col in self.bool_columns_:
            if col in X.columns:
                X[col] = X[col].astype("float")

        self.numeric_columns_.extend(self.bool_columns_)

    def _generate_llm_plan(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        if self.llm is None or HumanMessage is None or SystemMessage is None:
            return {
                "planner_status": "disabled_or_unavailable",
                "actions": [],
                "reason": "LLM is not configured. Falling back to rule-based feature engineering."
            }

        dataset_profile = {
            "target_column": self.config.target_column,
            "problem_type": self.config.problem_type,
            "task_description": self.config.task_description,
            "row_count": int(len(X)),
            "column_profiles": [asdict(p) for p in self.column_profiles_ if p.name in X.columns],
            "current_roles": {
                "numeric": self.numeric_columns_,
                "categorical": self.categorical_columns_,
                "datetime": self.datetime_columns_,
                "text": self.text_columns_,
                "bool_as_numeric": self.bool_columns_,
            },
        }

        system_prompt = """
You are a feature engineering planner for machine learning pipelines.
Your job is to propose SAFE, structured feature engineering actions.

Rules:
1. Return valid JSON only.
2. Do not suggest arbitrary code.
3. Only use the following action types:
   - add_ratio_feature
   - add_sum_feature
   - add_difference_feature
   - add_product_feature
   - add_log_feature
   - add_binned_feature
   - drop_column
4. Only suggest actions when they are reasonable from the provided schema.
5. Never touch the target column.
6. Avoid leakage-prone suggestions.
7. Prefer simple, interpretable features.

JSON format:
{
  "planner_status": "ok",
  "actions": [
    {
      "action": "add_ratio_feature",
      "source_columns": ["income", "age"],
      "new_column": "income_per_age",
      "reason": "..."
    }
  ]
}
"""

        human_prompt = f"""
Dataset profile:
{json.dumps(dataset_profile, ensure_ascii=False, indent=2)}

Please generate a safe feature engineering plan.
"""

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            )
            content = response.content.strip()

            plan = self._extract_json_from_text(content)
            if not isinstance(plan, dict):
                raise ValueError("LLM response is not a valid JSON object.")

            if "actions" not in plan or not isinstance(plan["actions"], list):
                plan["actions"] = []

            if "planner_status" not in plan:
                plan["planner_status"] = "ok"

            return plan

        except Exception as e:
            return {
                "planner_status": "failed",
                "actions": [],
                "reason": f"LLM planning failed: {e}"
            }

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        text = text.strip()

        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError("No JSON object found in LLM response.")

    def _apply_llm_plan(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        X = X.copy()
        actions = self.llm_plan_.get("actions", []) if isinstance(self.llm_plan_, dict) else []

        if not fit or not actions:
            return X

        valid_actions = []
        for action in actions:
            applied = self._apply_single_llm_action(X, action)
            if applied:
                valid_actions.append(action)

        self.generated_feature_log_ = valid_actions
        return X

    def _apply_saved_generated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for action in self.generated_feature_log_:
            self._apply_single_llm_action(X, action, store_log=False)
        return X

    def _apply_single_llm_action(
        self,
        X: pd.DataFrame,
        action: Dict[str, Any],
        store_log: bool = True
    ) -> bool:
        action_type = action.get("action")
        source_columns = action.get("source_columns", [])
        new_column = action.get("new_column")

        if not action_type or not isinstance(source_columns, list):
            return False

        if any(col == self.config.target_column for col in source_columns):
            return False

        if action_type == "drop_column":
            col = action.get("column") or (source_columns[0] if source_columns else None)
            if col in X.columns:
                X.drop(columns=[col], inplace=True)
                if col in self.numeric_columns_:
                    self.numeric_columns_.remove(col)
                if col in self.categorical_columns_:
                    self.categorical_columns_.remove(col)
                if col in self.datetime_columns_:
                    self.datetime_columns_.remove(col)
                if col in self.text_columns_:
                    self.text_columns_.remove(col)
                self.drop_columns_.append(col)
                return True
            return False

        if not new_column or new_column in X.columns:
            return False

        try:
            if action_type == "add_ratio_feature" and len(source_columns) == 2:
                a, b = source_columns
                if a in X.columns and b in X.columns:
                    X[new_column] = pd.to_numeric(X[a], errors="coerce") / (
                        pd.to_numeric(X[b], errors="coerce").replace(0, np.nan)
                    )
                    self.numeric_columns_.append(new_column)
                    return True

            elif action_type == "add_sum_feature" and len(source_columns) >= 2:
                cols = [c for c in source_columns if c in X.columns]
                if len(cols) >= 2:
                    X[new_column] = X[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
                    self.numeric_columns_.append(new_column)
                    return True

            elif action_type == "add_difference_feature" and len(source_columns) == 2:
                a, b = source_columns
                if a in X.columns and b in X.columns:
                    X[new_column] = pd.to_numeric(X[a], errors="coerce") - pd.to_numeric(X[b], errors="coerce")
                    self.numeric_columns_.append(new_column)
                    return True

            elif action_type == "add_product_feature" and len(source_columns) == 2:
                a, b = source_columns
                if a in X.columns and b in X.columns:
                    X[new_column] = pd.to_numeric(X[a], errors="coerce") * pd.to_numeric(X[b], errors="coerce")
                    self.numeric_columns_.append(new_column)
                    return True

            elif action_type == "add_log_feature" and len(source_columns) == 1:
                a = source_columns[0]
                if a in X.columns:
                    s = pd.to_numeric(X[a], errors="coerce")
                    X[new_column] = np.where(s > 0, np.log1p(s), np.nan)
                    self.numeric_columns_.append(new_column)
                    return True

            elif action_type == "add_binned_feature" and len(source_columns) == 1:
                a = source_columns[0]
                bins = int(action.get("bins", 4))
                if a in X.columns:
                    s = pd.to_numeric(X[a], errors="coerce")
                    X[new_column] = pd.qcut(s, q=min(bins, s.nunique(dropna=True)), duplicates="drop").astype("string")
                    self.categorical_columns_.append(new_column)
                    return True

        except Exception:
            return False

        return False

    def _drop_high_correlation_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.drop_high_correlation:
            self.high_corr_drop_columns_ = []
            return X

        numeric_cols = [c for c in X.columns if c in self.numeric_columns_]
        if len(numeric_cols) < 2:
            self.high_corr_drop_columns_ = []
            return X

        num_df = X[numeric_cols].copy()
        num_df = num_df.apply(pd.to_numeric, errors="coerce")

        if num_df.shape[1] < 2:
            self.high_corr_drop_columns_ = []
            return X

        corr = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.config.correlation_threshold)]

        self.high_corr_drop_columns_ = to_drop
        self.numeric_columns_ = [c for c in self.numeric_columns_ if c not in to_drop]
        return X.drop(columns=to_drop, errors="ignore")

    def _build_preprocessor(self) -> ColumnTransformer:
        transformers = []

        if self.numeric_columns_:
            numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
            if self.config.scale_numeric:
                numeric_steps.append(("scaler", StandardScaler()))
            numeric_pipeline = Pipeline(steps=numeric_steps)
            transformers.append(("num", numeric_pipeline, self.numeric_columns_))

        if self.categorical_columns_:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("rare_grouper", RareCategoryGrouper(
                        threshold=self.config.rare_category_threshold,
                        rare_label=self.config.rare_category_label
                    )),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat", categorical_pipeline, self.categorical_columns_))

        if self.datetime_columns_:
            datetime_pipeline = Pipeline(
                steps=[
                    ("datetime_features", DateTimeFeatureExtractor()),
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]
            )
            transformers.append(("dt", datetime_pipeline, self.datetime_columns_))

        if self.text_columns_:
            text_pipeline = Pipeline(
                steps=[
                    ("text_stats", TextStatsExtractor()),
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ]
            )
            transformers.append(("txt", text_pipeline, self.text_columns_))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _get_feature_names(self) -> List[str]:
        if self.preprocessor_ is None:
            return []

        names: List[str] = []

        for group_name, transformer, cols in self.preprocessor_.transformers_:
            if group_name == "remainder":
                continue

            if group_name == "num":
                names.extend(list(cols))

            elif group_name == "cat":
                try:
                    pipe = self.preprocessor_.named_transformers_["cat"]
                    onehot = pipe.named_steps["onehot"]
                    cat_names = list(onehot.get_feature_names_out(cols))
                    names.extend(cat_names)
                except Exception:
                    names.extend([f"{c}__encoded" for c in cols])

            elif group_name == "dt":
                for c in cols:
                    names.extend([
                        f"{c}__year",
                        f"{c}__month",
                        f"{c}__day",
                        f"{c}__weekday",
                        f"{c}__quarter",
                        f"{c}__is_month_start",
                        f"{c}__is_month_end",
                    ])

            elif group_name == "txt":
                for c in cols:
                    names.extend([
                        f"{c}__char_len",
                        f"{c}__word_count",
                        f"{c}__digit_count",
                        f"{c}__uppercase_count",
                    ])

        return [self._sanitize_feature_name(n) for n in names]

    def _sanitize_feature_name(self, name: str) -> str:
        name = str(name)
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name

    def _build_summary(self) -> Dict[str, Any]:
        if self.summary_:
            return self.summary_

        summary = {
            "target_column": self.config.target_column,
            "problem_type": self.config.problem_type,
            "task_description": self.config.task_description,
            "llm_plan_status": self.llm_plan_.get("planner_status", "not_used") if isinstance(self.llm_plan_, dict) else "not_used",
            "llm_actions_count": len(self.generated_feature_log_),
            "llm_actions_applied": self.generated_feature_log_,
            "dropped_columns": {
                "general_drop": self.drop_columns_,
                "constant_columns": self.constant_columns_,
                "high_correlation_drop": self.high_corr_drop_columns_,
            },
            "used_columns": {
                "numeric": self.numeric_columns_,
                "categorical": self.categorical_columns_,
                "datetime": self.datetime_columns_,
                "text": self.text_columns_,
                "bool_as_numeric": self.bool_columns_,
            },
            "final_feature_count": len(self.feature_names_),
        }
        self.summary_ = summary
        return summary

    def _build_metadata(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "column_profiles": [asdict(p) for p in self.column_profiles_],
            "feature_names": self.feature_names_,
            "llm_plan": self.llm_plan_,
        }

    def _save_artifacts(self, result: Dict[str, Any]) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result["X"].to_csv(out_dir / "X_engineered.csv", index=False)
        pd.DataFrame({self.config.target_column: result["y"]}).to_csv(out_dir / "y.csv", index=False)

        with open(out_dir / "feature_summary.json", "w", encoding="utf-8") as f:
            json.dump(result["summary"], f, ensure_ascii=False, indent=2)

        with open(out_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
            json.dump(result["metadata"], f, ensure_ascii=False, indent=2)

        with open(out_dir / "llm_plan.json", "w", encoding="utf-8") as f:
            json.dump(result["llm_plan"], f, ensure_ascii=False, indent=2)

        joblib.dump(result["preprocessor"], out_dir / "preprocessor.joblib")

    def explain(self) -> str:
        summary = self._build_summary()
        lines = [
            "Feature Engineering Summary",
            f"- Target column: {summary['target_column']}",
            f"- Problem type: {summary['problem_type']}",
            f"- Task description: {summary['task_description']}",
            f"- LLM plan status: {summary['llm_plan_status']}",
            f"- LLM actions applied: {summary['llm_actions_count']}",
            f"- Numeric columns used: {len(summary['used_columns']['numeric'])}",
            f"- Categorical columns used: {len(summary['used_columns']['categorical'])}",
            f"- Datetime columns used: {len(summary['used_columns']['datetime'])}",
            f"- Text columns used: {len(summary['used_columns']['text'])}",
            f"- Dropped columns: {len(summary['dropped_columns']['general_drop']) + len(summary['dropped_columns']['high_correlation_drop'])}",
            f"- Final engineered feature count: {summary['final_feature_count']}",
        ]
        return "\n".join(lines)


def demo() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["A001", "A002", "A003", "A004", "A005", "A006"],
            "age": [23, 35, np.nan, 29, 41, 37],
            "income": [50000, 72000, 68000, 59000, 81000, 79000],
            "spend": [1200, 2200, 1800, 1400, 2600, 2400],
            "city": ["Sydney", "Melbourne", "Sydney", "Brisbane", "Sydney", "OtherTown"],
            "signup_date": [
                "2024-01-01",
                "2024-02-15",
                "2024-03-20",
                "2024-03-25",
                None,
                "2024-04-10",
            ],
            "is_premium": [True, False, True, False, True, False],
            "review_text": [
                "Very good product and fast shipping",
                "Average",
                "Loved it, will buy again",
                None,
                "Bad packaging but decent quality",
                "Excellent service",
            ],
            "target": [1, 0, 1, 0, 1, 1],
        }
    )

    config = FeatureEngineeringConfig(
        target_column="target",
        problem_type="classification",
        task_description="Predict customer conversion",
        use_llm_planner=True,
        save_artifacts=False,
    )

    agent = FeatureEngineeringAgent(config)
    result = agent.run(df)

    print(agent.explain())
    print("\nLLM plan:")
    print(json.dumps(result["llm_plan"], ensure_ascii=False, indent=2))
    print("\nEngineered X head:")
    print(result["X"].head())


if __name__ == "__main__":
    demo()