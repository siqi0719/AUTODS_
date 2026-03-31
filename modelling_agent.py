"""Hyperparameter-tuning modelling agent (V1) for tabular learning experiments.

V1 extends the V0 baseline with a two-stage pipeline:
  Stage 1 — quick baseline comparison across all candidates (identical to V0).
  Stage 2 — Optuna TPE hyperparameter search on the top-N Stage 1 models,
             with adaptive trial budget, early stopping, and optional LLM
             candidate selection.

V0 artifact format is preserved; V1 adds tuning_summary.json and
tuning_history.json to the output directory when tuning is enabled.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from utils import json_default

try:
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None  # type: ignore[assignment]
    TPESampler = None  # type: ignore[assignment]

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None


@dataclass
class CandidateModel:
    name: str
    estimator: Any


@dataclass
class ModellingConfig:
    target_column: str
    problem_type: str = "classification"
    task_description: str = "General predictive modelling"
    primary_metric: Optional[str] = None
    cv_folds: int = 5
    random_state: int = 42
    save_artifacts: bool = True
    output_dir: str = "model_outputs"
    use_llm_planner: bool = True
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    candidate_model_names: Optional[List[str]] = None
    supported_future_problem_families: List[str] = field(
        default_factory=lambda: ["classification", "regression", "unsupervised", "nlp"]
    )

    def resolved_primary_metric(self) -> str:
        if self.primary_metric:
            return self.primary_metric
        return "roc_auc" if self.problem_type == "classification" else "rmse"


@dataclass
class TuningConfig:
    """Configuration for Stage 2 Optuna hyperparameter search."""

    enable_tuning: bool = True
    n_top_models_to_tune: int = 3
    early_stopping_rounds: int = 10  # consecutive non-improving trials before study stops
    tuning_intensity: str = "auto"  # "auto" | "light" | "full"
    use_llm_stage2_selection: bool = True  # falls back to top-N Python when LLM unavailable


# ---------------------------------------------------------------------------
# Search-space definitions for Optuna TPE.
# Format per param: (type, low, high)  where type ∈ {"int","float","float_log"}.
# Models that wrap a sklearn Pipeline use the "model__" prefix.
# XGBoost classification uses _XGBAutoObjectivePipeline (no prefix; built inline).
# ---------------------------------------------------------------------------
SEARCH_SPACES: Dict[str, Dict[str, tuple]] = {
    # --- classification ---
    "logistic_regression": {
        "model__C": ("float_log", 1e-3, 10.0),
    },
    "random_forest": {
        "model__n_estimators": ("int", 100, 500),
        "model__max_depth": ("int", 3, 15),
        "model__min_samples_split": ("int", 2, 10),
    },
    "svm_rbf": {
        "model__C": ("float_log", 1e-2, 100.0),
        "model__gamma": ("float_log", 1e-4, 1.0),
    },
    "xgboost": {  # handled by _build_xgb_tuning_estimator — no model__ prefix
        "n_estimators": ("int", 100, 500),
        "max_depth": ("int", 3, 10),
        "learning_rate": ("float_log", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    },
    "lightgbm": {
        "model__n_estimators": ("int", 100, 500),
        "model__learning_rate": ("float_log", 0.01, 0.3),
        "model__num_leaves": ("int", 20, 150),
    },
    # --- regression ---
    "ridge_regression": {
        "model__alpha": ("float_log", 1e-3, 100.0),
    },
    "random_forest_regressor": {
        "model__n_estimators": ("int", 100, 500),
        "model__max_depth": ("int", 3, 15),
        "model__min_samples_split": ("int", 2, 10),
    },
    "svr_rbf": {
        "model__C": ("float_log", 1e-2, 100.0),
        "model__gamma": ("float_log", 1e-4, 1.0),
    },
    "xgboost_regressor": {  # standard Pipeline, uses model__ prefix
        "model__n_estimators": ("int", 100, 500),
        "model__max_depth": ("int", 3, 10),
        "model__learning_rate": ("float_log", 0.01, 0.3),
        "model__subsample": ("float", 0.6, 1.0),
        "model__colsample_bytree": ("float", 0.6, 1.0),
    },
    "lightgbm_regressor": {
        "model__n_estimators": ("int", 100, 500),
        "model__learning_rate": ("float_log", 0.01, 0.3),
        "model__num_leaves": ("int", 20, 150),
    },
}


class _EarlyStoppingCallback:
    """Optuna callback that halts a study when improvement stalls.

    Stops the study if the per-trial best value has not improved by at least
    ``min_improvement`` (relative) for ``rounds`` consecutive trials.
    """

    def __init__(self, rounds: int = 10, min_improvement: float = 0.01) -> None:
        self._rounds = rounds
        self._min_improvement = min_improvement
        self._reference: Optional[float] = None
        self._consecutive = 0

    def __call__(self, study: Any, trial: Any) -> None:  # type: ignore[override]
        current = study.best_value
        if self._reference is None:
            self._reference = current
            return
        denominator = abs(self._reference)
        if denominator < 1e-10:
            improved = abs(current - self._reference) > self._min_improvement
        else:
            improved = (current - self._reference) / denominator > self._min_improvement
        if improved:
            self._reference = current
            self._consecutive = 0
        else:
            self._consecutive += 1
        if self._consecutive >= self._rounds:
            study.stop()


class _XGBAutoObjectivePipeline:
    """Thin sklearn-compatible wrapper around XGBClassifier that picks the
    correct objective at fit time based on the number of unique classes.

    P0 Fix: XGBClassifier with objective='binary:logistic' crashes or silently
    produces wrong predictions when n_classes > 2.  This wrapper defers the
    objective choice until fit() is called, so the same registry entry works
    for both binary and multi-class problems.
    """

    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state
        self._pipeline: Optional[Any] = None

    def _build(self, y: Any) -> Any:
        n_classes = len(set(y))
        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective=objective,
            eval_metric=eval_metric,
            random_state=self._random_state,
            use_label_encoder=False,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    def fit(self, X: Any, y: Any) -> "_XGBAutoObjectivePipeline":
        self._pipeline = self._build(y)
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        return self._pipeline.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._pipeline.predict_proba(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"random_state": self._random_state}

    def set_params(self, **params: Any) -> "_XGBAutoObjectivePipeline":
        if "random_state" in params:
            self._random_state = params["random_state"]
        return self

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Expose named_steps so _extract_core_estimator works correctly."""
        if self._pipeline is not None:
            return self._pipeline.named_steps
        return {}


class ModellingAgent:
    def __init__(
        self,
        config: ModellingConfig,
        tuning_config: Optional[TuningConfig] = None,
    ):
        self.config = config
        self.tuning_config: TuningConfig = tuning_config or TuningConfig()
        self.summary_: Dict[str, Any] = {}
        self.metadata_: Dict[str, Any] = {}
        self.best_model_name_: Optional[str] = None
        self.best_model_: Optional[Any] = None
        self.best_model_metrics_: Dict[str, Any] = {}
        self.leaderboard_: pd.DataFrame = pd.DataFrame()
        self.llm_plan_: Dict[str, Any] = {}
        self.diagnostics_: Dict[str, Any] = {}
        self.best_model_feature_importance_: pd.DataFrame = pd.DataFrame()
        self.resolved_cv_folds_: int = self.config.cv_folds
        self.tuning_summary_: Dict[str, Any] = {}
        self.tuning_history_: Dict[str, List[Dict[str, Any]]] = {}

        load_dotenv()
        self.llm = self._init_llm()

    def _init_llm(self):
        if not self.config.use_llm_planner:
            return None
        if ChatOpenAI is None or HumanMessage is None or SystemMessage is None:
            return None
        if not os.getenv("OPENAI_API_KEY"):
            return None

        return ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
        )

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        feature_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Issue #1: accept full data; split internally with guaranteed stratification
        self._expected_columns_: Optional[List[str]] = None  # reset before validation
        self._validate_inputs(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=y if self.config.problem_type == "classification" else None,
        )
        # P2: record train columns so predict-time column drift can be caught
        self._expected_columns_ = list(X_train.columns)

        # Detect the positive label for binary classification.
        # sklearn string scorers default to pos_label=1 (int), which fails when
        # labels are strings like 'e'/'p' or 'yes'/'no'.
        unique_labels = sorted(y_train.dropna().unique(), key=str)
        if self.config.problem_type == "classification" and len(unique_labels) == 2:
            _pos_candidates = {1, "1", "yes", "true", "y", "t", "p", "positive"}
            self.pos_label_ = next(
                (lb for lb in unique_labels if lb in _pos_candidates),
                unique_labels[-1],
            )
        else:
            self.pos_label_ = 1

        self.llm_plan_ = self._generate_llm_plan(X_train, y_train, feature_metadata)
        candidates = self._get_candidate_models()
        leaderboard_rows: List[Dict[str, Any]] = []
        fitted_models: Dict[str, Any] = {}
        prediction_outputs: Dict[str, Dict[str, List[Any]]] = {}

        for candidate in candidates:
            # P1: isolate per-model failures so one bad model doesn't abort the run
            try:
                row, fitted_model, predictions = self._evaluate_candidate(
                    candidate=candidate,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                )
            except Exception as exc:  # noqa: BLE001
                row = {
                    "model_name": candidate.name,
                    "cv_runtime_seconds": np.nan,
                    "training_error": str(exc),
                    **{k: np.nan for k in self._empty_test_metric_dict()},
                }
                fitted_model = None
                predictions = {}
            leaderboard_rows.append(row)
            if fitted_model is not None:
                fitted_models[candidate.name] = fitted_model
            prediction_outputs[candidate.name] = predictions

        leaderboard = pd.DataFrame(leaderboard_rows)
        leaderboard = self._rank_leaderboard(leaderboard)
        self.leaderboard_ = leaderboard

        # ── STAGE 2: Optuna hyperparameter tuning on top-N Stage 1 models ──
        tuning_summary: Dict[str, Any] = {}
        tuning_history: Dict[str, List[Dict[str, Any]]] = {}

        if self.tuning_config.enable_tuning and optuna is not None:
            stage2_names = self._select_stage2_candidates(leaderboard, candidates)
            n_trials, cv_folds_tuning = self._resolve_trial_budget(
                len(X_train), self.tuning_config.tuning_intensity
            )
            candidate_map = {c.name: c for c in candidates}

            for model_name in stage2_names:
                candidate = candidate_map.get(model_name)
                if candidate is None or model_name not in SEARCH_SPACES:
                    continue
                # Skip models that failed Stage 1
                stage1_data = next(
                    (r for r in leaderboard_rows if r.get("model_name") == model_name), None
                )
                if stage1_data is None or "training_error" in stage1_data:
                    continue

                try:
                    best_params, best_score, history = self._tune_model(
                        model_name=model_name,
                        base_pipeline=candidate.estimator,
                        X_train=X_train,
                        y_train=y_train,
                        n_trials=n_trials,
                        cv_folds_tuning=cv_folds_tuning,
                    )
                except Exception as exc:  # noqa: BLE001
                    tuning_summary[model_name] = {"status": "failed", "error": str(exc)}
                    continue

                primary_cv_col = self._primary_cv_col()
                stage1_cv_score = float(stage1_data.get(primary_cv_col, np.nan))
                improvement = (
                    round(best_score - stage1_cv_score, 6)
                    if not (np.isnan(best_score) or np.isnan(stage1_cv_score))
                    else None
                )
                tuning_summary[model_name] = {
                    "status": "ok",
                    "best_tuned_cv_score": best_score,
                    "baseline_cv_score": stage1_cv_score,
                    "improvement": improvement,
                    "n_trials_actual": len(history),
                    "best_params": best_params,
                }
                tuning_history[model_name] = history

                # Re-evaluate the tuned model on the full train/test split
                try:
                    tuned_est = self._build_tuned_estimator(
                        model_name, candidate.estimator, best_params, y_train
                    )
                    tuned_candidate = CandidateModel(name=model_name, estimator=tuned_est)
                    tuned_row, tuned_fitted, tuned_preds = self._evaluate_candidate(
                        candidate=tuned_candidate,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                    )
                    tuned_row["tuned"] = True
                    tuned_row["stage1_cv_score"] = stage1_cv_score
                    # Replace Stage 1 row in-place
                    for i, r in enumerate(leaderboard_rows):
                        if r.get("model_name") == model_name:
                            leaderboard_rows[i] = tuned_row
                            break
                    fitted_models[model_name] = tuned_fitted
                    prediction_outputs[model_name] = tuned_preds
                except Exception as exc:  # noqa: BLE001
                    tuning_summary[model_name]["re_evaluate_error"] = str(exc)

            # Re-rank if any models were successfully tuned
            if any(s.get("status") == "ok" for s in tuning_summary.values()):
                leaderboard = pd.DataFrame(leaderboard_rows)
                leaderboard = self._rank_leaderboard(leaderboard)
                self.leaderboard_ = leaderboard

            # Optional LLM text interpretation of tuning results
            if tuning_summary:
                llm_interp = self._generate_llm_tuning_interpretation(tuning_summary)
                if llm_interp:
                    tuning_summary["_llm_interpretation"] = llm_interp

        self.tuning_summary_ = tuning_summary
        self.tuning_history_ = tuning_history

        # P1: skip models that failed training when picking best
        successful_models = [r["model_name"] for r in leaderboard_rows if "training_error" not in r]
        if not successful_models:
            raise RuntimeError("All candidate models failed during training. Check data and model configs.")
        best_row = leaderboard[leaderboard["model_name"].isin(successful_models)].iloc[0].to_dict()
        self.best_model_name_ = str(best_row["model_name"])
        self.best_model_ = fitted_models[self.best_model_name_]
        self.best_model_metrics_ = best_row

        self.summary_ = self._build_summary(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_metadata=feature_metadata,
        )
        self.best_model_feature_importance_ = self._extract_feature_importance(X_train.columns)
        self.diagnostics_ = self._build_diagnostics(
            X_test=X_test,
            y_test=y_test,
            prediction_outputs=prediction_outputs,
            feature_names=list(X_train.columns),
        )
        self.metadata_ = self._build_metadata(feature_metadata=feature_metadata)

        result = {
            "status": "success",
            "problem_type": self.config.problem_type,
            "primary_metric": self.config.resolved_primary_metric(),
            "leaderboard": leaderboard,
            "best_model_name": self.best_model_name_,
            "best_model": self.best_model_,
            "best_model_metrics": self.best_model_metrics_,
            "predictions": prediction_outputs,
            "diagnostics": self.diagnostics_,
            "best_model_feature_importance": self.best_model_feature_importance_,
            "summary": self.summary_,
            "metadata": self.metadata_,
            "llm_plan": self.llm_plan_,
            "tuning_summary": tuning_summary,
            "tuning_history": tuning_history,
        }

        if self.config.save_artifacts:
            self._save_artifacts(result)

        return result

    def _validate_inputs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        # Issue #1: validate full dataset (before internal split)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Data must not be empty.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        if self.config.problem_type not in self.config.supported_future_problem_families:
            raise ValueError(f"Unsupported problem type: {self.config.problem_type}")

        # P2: train/test column consistency
        if hasattr(self, "_expected_columns_") and self._expected_columns_ is not None:
            missing = set(self._expected_columns_) - set(X.columns)
            extra = set(X.columns) - set(self._expected_columns_)
            if missing or extra:
                raise ValueError(
                    f"Column mismatch — missing: {sorted(missing)}, unexpected: {sorted(extra)}"
                )

        if self.config.problem_type in {"unsupervised", "nlp"}:
            raise NotImplementedError(
                f"Problem type '{self.config.problem_type}' is reserved for a later project phase."
            )
        if self.config.problem_type == "classification" and y.nunique(dropna=True) < 2:
            raise ValueError("Classification requires at least two target classes in y.")

    def _generate_llm_plan(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.llm is None:
            # Issue #8: unified LLM status — "not_configured" when planner is off or no API key
            return {
                "planner_status": "not_configured",
                "recommended_models": [],
                "reason": "LLM is not configured. Falling back to deterministic candidate models.",
            }

        profile = {
            "problem_type": self.config.problem_type,
            "task_description": self.config.task_description,
            "row_count": int(len(X_train)),
            "feature_count": int(X_train.shape[1]),
            "target_distribution": y_train.value_counts(dropna=False).to_dict()
            if self.config.problem_type == "classification"
            else {
                "mean": float(y_train.mean()),
                "std": float(y_train.std(ddof=0)),
            },
            "feature_metadata_available": feature_metadata is not None,
            # Issue #7: pass actual feature type counts so LLM can give targeted recommendations
            "feature_summary": {
                "numeric_count": len(feature_metadata.get("used_columns", {}).get("numeric", [])),
                "categorical_count": len(feature_metadata.get("used_columns", {}).get("categorical", [])),
                "datetime_count": len(feature_metadata.get("used_columns", {}).get("datetime", [])),
                "text_count": len(feature_metadata.get("used_columns", {}).get("text", [])),
            } if feature_metadata else None,
            "primary_metric": self.config.resolved_primary_metric(),
        }

        system_prompt = """
You are a modelling planner for tabular machine learning workflows.
Return valid JSON only.
Recommend a small, academically defensible shortlist of model families.
Do not generate code.

JSON format:
{
  "planner_status": "ok",
  "recommended_models": ["logistic_regression", "random_forest"],
  "reason": "..."
}
"""

        human_prompt = f"""
Modelling context:
{json.dumps(profile, ensure_ascii=False, indent=2)}

Recommend a concise shortlist for the current phase.
"""

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            )
            content = str(response.content).strip()
            plan = self._extract_json(content)
            if "recommended_models" not in plan or not isinstance(plan["recommended_models"], list):
                plan["recommended_models"] = []
            if "planner_status" not in plan:
                plan["planner_status"] = "ok"
            return plan
        except Exception as exc:
            return {
                "planner_status": "failed",
                "recommended_models": [],
                "reason": f"LLM modelling planning failed: {exc}",
            }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("No JSON object found in LLM response.")

    def _get_candidate_models(self) -> List[CandidateModel]:
        if self.config.problem_type == "classification":
            models = self._classification_models()
        elif self.config.problem_type == "regression":
            models = self._regression_models()
        else:
            raise NotImplementedError(
                f"Problem type '{self.config.problem_type}' is not implemented in the current phase."
            )

        requested_names = self.config.candidate_model_names
        if requested_names:
            requested_set = set(requested_names)
            models = [model for model in models if model.name in requested_set]

        llm_recommendations = self.llm_plan_.get("recommended_models", [])
        if llm_recommendations:
            # Issue #6: validate LLM names against available registry; record unrecognized names
            available_names = {model.name for model in models}
            unrecognized = [name for name in llm_recommendations if name not in available_names]
            if unrecognized:
                self.llm_plan_["unrecognized_model_names"] = unrecognized
            recommended_set = set(llm_recommendations)
            recommended = [model for model in models if model.name in recommended_set]
            remaining = [model for model in models if model.name not in recommended_set]
            models = recommended + remaining

        if not models:
            raise ValueError("No candidate models are available after filtering.")
        return models

    def available_model_names(self) -> List[str]:
        if self.config.problem_type == "classification":
            return [model.name for model in self._classification_models()]
        if self.config.problem_type == "regression":
            return [model.name for model in self._regression_models()]
        return []

    def _make_numeric_pipeline(self, model: Any) -> Pipeline:
        """Standard pipeline for scale-invariant models (trees, etc.)."""
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])

    def _make_scaled_pipeline(self, model: Any) -> Pipeline:
        """Standard pipeline for scale-sensitive models (SVM, LR). P1 + P2."""
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    def _classification_models(self) -> List[CandidateModel]:
        models: List[CandidateModel] = [
            CandidateModel(
                name="logistic_regression",
                # P1: LR needs StandardScaler for correct L2 regularisation
                estimator=self._make_scaled_pipeline(
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=self.config.random_state,
                    )
                ),
            ),
            CandidateModel(
                name="random_forest",
                estimator=self._make_numeric_pipeline(
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=self.config.random_state,
                        n_jobs=-1,
                    )
                ),
            ),
            CandidateModel(
                name="svm_rbf",
                # P1: SVM RBF kernel is distance-based; StandardScaler is required
                estimator=self._make_scaled_pipeline(
                    SVC(probability=True, class_weight="balanced", random_state=self.config.random_state)
                ),
            ),
        ]

        if XGBClassifier is not None:
            # P0: select XGBoost objective based on n_classes; cannot know at registry-build time
            # so we defer to a factory that reads from data at fit time via a wrapper shim.
            # Simplest robust approach: let XGBoost auto-detect via num_class when n_classes > 2.
            models.append(
                CandidateModel(
                    name="xgboost",
                    estimator=_XGBAutoObjectivePipeline(random_state=self.config.random_state),
                )
            )

        if LGBMClassifier is not None:
            models.append(
                CandidateModel(
                    name="lightgbm",
                    estimator=self._make_numeric_pipeline(
                        LGBMClassifier(
                            n_estimators=300,
                            learning_rate=0.05,
                            random_state=self.config.random_state,
                            verbose=-1,
                        )
                    ),
                )
            )

        return models

    def _regression_models(self) -> List[CandidateModel]:
        models: List[CandidateModel] = [
            CandidateModel(
                name="ridge_regression",
                # P1: Ridge uses L2 regularisation; scaling is required
                estimator=self._make_scaled_pipeline(
                    Ridge(random_state=self.config.random_state)
                ),
            ),
            CandidateModel(
                name="random_forest_regressor",
                estimator=self._make_numeric_pipeline(
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                    )
                ),
            ),
            CandidateModel(
                name="svr_rbf",
                # P1: SVR RBF kernel is distance-based; StandardScaler is required
                estimator=self._make_scaled_pipeline(SVR()),
            ),
        ]

        if XGBRegressor is not None:
            models.append(
                CandidateModel(
                    name="xgboost_regressor",
                    estimator=self._make_numeric_pipeline(
                        XGBRegressor(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="reg:squarederror",
                            random_state=self.config.random_state,
                        )
                    ),
                )
            )

        if LGBMRegressor is not None:
            models.append(
                CandidateModel(
                    name="lightgbm_regressor",
                    estimator=self._make_numeric_pipeline(
                        LGBMRegressor(
                            n_estimators=300,
                            learning_rate=0.05,
                            random_state=self.config.random_state,
                            verbose=-1,
                        )
                    ),
                )
            )

        return models

    def _evaluate_candidate(
        self,
        candidate: CandidateModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
    ) -> tuple[Dict[str, Any], Any, Dict[str, List[Any]]]:
        estimator = clone(candidate.estimator)
        scoring = self._scoring_metrics(y_train)
        cv = self._cross_validator(y_train)

        start_time = time.perf_counter()
        cv_result = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        cv_runtime = time.perf_counter() - start_time

        estimator.fit(X_train, y_train)

        predictions: Dict[str, List[Any]] = {}
        test_metrics = self._empty_test_metric_dict()

        if X_test is not None:
            y_pred = estimator.predict(X_test)
            predictions["test_predictions"] = np.asarray(y_pred).tolist()
            if y_test is not None:
                predictions["test_truth"] = np.asarray(y_test).tolist()

            if self.config.problem_type == "classification":
                y_score = self._predict_scores(estimator, X_test)
                if y_score is not None:
                    arr = np.asarray(y_score)
                    if arr.ndim == 2:
                        # Multi-class: expand probability matrix into per-class columns so
                        # pd.DataFrame produces numeric columns, not stringified lists.
                        for i in range(arr.shape[1]):
                            predictions[f"test_score_class_{i}"] = arr[:, i].tolist()
                    else:
                        # Binary: single positive-class probability column.
                        predictions["test_scores"] = arr.tolist()
                else:
                    predictions["test_scores"] = []
                if y_test is not None:
                    test_metrics = self._classification_test_metrics(y_test, y_pred, y_score)
            else:
                if y_test is not None:
                    test_metrics = self._regression_test_metrics(y_test, y_pred)

        row = {
            "model_name": candidate.name,
            "cv_runtime_seconds": round(cv_runtime, 4),
            **self._mean_cv_metrics(cv_result),
            **self._std_cv_metrics(cv_result),
            **test_metrics,
        }
        return row, estimator, predictions

    def _scoring_metrics(self, y_train: pd.Series) -> Dict[str, Any]:
        if self.config.problem_type == "classification":
            n_classes = y_train.nunique()
            binary = n_classes == 2
            if binary:
                pl = self.pos_label_
                return {
                    "accuracy": "accuracy",
                    "precision": make_scorer(precision_score, pos_label=pl, zero_division=0),
                    "recall": make_scorer(recall_score, pos_label=pl, zero_division=0),
                    "f1": make_scorer(f1_score, pos_label=pl, zero_division=0),
                    "roc_auc": "roc_auc",
                }
            return {
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted",
                "roc_auc": "roc_auc_ovr_weighted",
            }
        return {
            "neg_rmse": "neg_root_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

    def _cross_validator(self, y_train: pd.Series):
        if self.config.problem_type == "classification":
            class_counts = y_train.value_counts(dropna=False)
            min_class_count = int(class_counts.min()) if not class_counts.empty else 2
            # Issue #2: raise a clear error before StratifiedKFold chokes on it
            if min_class_count < 2:
                raise ValueError(
                    f"Class '{class_counts.idxmin()}' has only {min_class_count} sample(s). "
                    f"StratifiedKFold requires at least 2 samples per class."
                )
            n_splits = max(2, min(self.config.cv_folds, min_class_count))
            self.resolved_cv_folds_ = n_splits
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.config.random_state,
            )
        n_splits = max(2, min(self.config.cv_folds, len(y_train)))
        self.resolved_cv_folds_ = n_splits
        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

    def _mean_cv_metrics(self, cv_result: Dict[str, np.ndarray]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for key, values in cv_result.items():
            if not key.startswith("test_"):
                continue
            metric_name = key.replace("test_", "cv_")
            mean_value = float(np.mean(values))
            if metric_name == "cv_neg_rmse":
                metrics["cv_rmse"] = abs(mean_value)
            elif metric_name == "cv_neg_mae":
                metrics["cv_mae"] = abs(mean_value)
            else:
                metrics[metric_name] = mean_value
        return metrics

    def _std_cv_metrics(self, cv_result: Dict[str, np.ndarray]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for key, values in cv_result.items():
            if not key.startswith("test_"):
                continue
            metric_name = key.replace("test_", "cv_std_")
            std_value = float(np.std(values, ddof=0))
            if metric_name == "cv_std_neg_rmse":
                metrics["cv_std_rmse"] = std_value
            elif metric_name == "cv_std_neg_mae":
                metrics["cv_std_mae"] = std_value
            else:
                metrics[metric_name] = std_value
        return metrics

    def _empty_test_metric_dict(self) -> Dict[str, Any]:
        if self.config.problem_type == "classification":
            return {
                "test_accuracy": np.nan,
                "test_precision": np.nan,
                "test_recall": np.nan,
                "test_f1": np.nan,
                "test_roc_auc": np.nan,
            }
        return {
            "test_rmse": np.nan,
            "test_mae": np.nan,
            "test_r2": np.nan,
        }

    def _predict_scores(self, estimator: Any, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        """Return probability scores suitable for roc_auc.

        Binary: probability of positive class (column 1). Shape (n,).
        Multi-class: full probability matrix. Shape (n, n_classes).
        P3: previously always returned probs[:, 1], silently wrong for multi-class.
        """
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(X_test)
            if probs.ndim == 2:
                if probs.shape[1] == 2:
                    return probs[:, 1]  # binary: positive-class probability
                return probs  # multi-class: full matrix
            return probs.ravel()
        if hasattr(estimator, "decision_function"):
            return estimator.decision_function(X_test)
        return None

    def _classification_test_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        n_classes = len(np.unique(y_true))
        binary = n_classes == 2
        # P0 + P3: use correct averaging for multi-class; roc_auc uses full prob matrix
        if binary:
            avg_kwargs = {"average": "binary", "pos_label": self.pos_label_}
        else:
            avg_kwargs = {"average": "weighted"}
        metrics = {
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_precision": float(precision_score(y_true, y_pred, zero_division=0, **avg_kwargs)),
            "test_recall": float(recall_score(y_true, y_pred, zero_division=0, **avg_kwargs)),
            "test_f1": float(f1_score(y_true, y_pred, zero_division=0, **avg_kwargs)),
            "test_roc_auc": np.nan,
        }
        if y_score is not None:
            try:
                if binary:
                    metrics["test_roc_auc"] = float(roc_auc_score(y_true, y_score))
                else:
                    metrics["test_roc_auc"] = float(
                        roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
                    )
            except Exception:
                pass
        return metrics

    def _regression_test_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
        }

    def _rank_leaderboard(self, leaderboard: pd.DataFrame) -> pd.DataFrame:
        primary_metric = self.config.resolved_primary_metric()

        # Issue #4: always rank by CV metric; test columns exist but are reference only
        sort_column_map = {
            "roc_auc": "cv_roc_auc",
            "accuracy": "cv_accuracy",
            "f1": "cv_f1",
            "rmse": "cv_rmse",
            "mae": "cv_mae",
            "r2": "cv_r2",
        }
        sort_column = sort_column_map[primary_metric]
        ascending = primary_metric in {"rmse", "mae"}

        leaderboard = leaderboard.sort_values(by=sort_column, ascending=ascending, na_position="last")
        leaderboard = leaderboard.reset_index(drop=True)
        leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
        return leaderboard

    def _build_diagnostics(
        self,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        prediction_outputs: Dict[str, Dict[str, List[Any]]],
        feature_names: List[str],
    ) -> Dict[str, Any]:
        if X_test is None or y_test is None or not self.best_model_name_:
            return {}

        best_predictions = prediction_outputs.get(self.best_model_name_, {})
        if "test_predictions" not in best_predictions:
            return {}

        y_pred = np.asarray(best_predictions["test_predictions"])
        diagnostics: Dict[str, Any] = {
            "best_model_name": self.best_model_name_,
            "train_rows": self.summary_.get("train_rows"),
            "test_rows": self.summary_.get("test_rows"),
            "data_quality_warnings": self._data_quality_warnings(
                train_rows=self.summary_.get("train_rows", 0),
                test_rows=self.summary_.get("test_rows", 0),
                feature_names=feature_names,
            ),
        }

        if self.config.problem_type == "classification":
            labels = sorted(pd.Series(y_test).dropna().unique().tolist())
            diagnostics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=labels).tolist()
            diagnostics["labels"] = labels
            diagnostics["class_distribution_test"] = pd.Series(y_test).value_counts(dropna=False).to_dict()
        else:
            residuals = pd.Series(y_test).reset_index(drop=True) - pd.Series(y_pred)
            diagnostics["residual_summary"] = {
                "mean": float(residuals.mean()),
                "std": float(residuals.std(ddof=0)),
                "min": float(residuals.min()),
                "max": float(residuals.max()),
            }

        return diagnostics

    def _extract_feature_importance(self, feature_names: pd.Index) -> pd.DataFrame:
        if self.best_model_ is None:
            return pd.DataFrame(columns=["feature_name", "importance"])

        core_model = self._extract_core_estimator(self.best_model_)
        feature_list = list(feature_names)

        if hasattr(core_model, "feature_importances_"):
            importance = np.asarray(core_model.feature_importances_, dtype=float)
        elif hasattr(core_model, "coef_"):
            coef = np.asarray(core_model.coef_, dtype=float)
            if coef.ndim == 1:
                importance = np.abs(coef)
            else:
                importance = np.mean(np.abs(coef), axis=0)
        else:
            return pd.DataFrame(columns=["feature_name", "importance"])

        if len(importance) != len(feature_list):
            return pd.DataFrame(columns=["feature_name", "importance"])

        importance_frame = pd.DataFrame(
            {
                "feature_name": feature_list,
                "importance": importance,
            }
        )
        importance_frame = importance_frame.sort_values(by="importance", ascending=False).reset_index(drop=True)
        return importance_frame

    def _extract_core_estimator(self, estimator: Any) -> Any:
        if isinstance(estimator, Pipeline):
            return estimator.named_steps.get("model", estimator)
        return estimator

    def _build_summary(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
        feature_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "problem_type": self.config.problem_type,
            "task_description": self.config.task_description,
            "primary_metric": self.config.resolved_primary_metric(),
            "resolved_cv_folds": self.resolved_cv_folds_,
            "available_models": self.available_model_names(),
            "candidate_models": self.leaderboard_["model_name"].tolist() if not self.leaderboard_.empty else [],
            "best_model_name": self.best_model_name_,
            "train_rows": int(len(X_train)),
            "train_feature_count": int(X_train.shape[1]),
            "test_rows": int(len(X_test)) if X_test is not None else 0,
            "llm_plan_status": self.llm_plan_.get("planner_status", "not_used"),
            "feature_metadata_available": feature_metadata is not None,
            "target_distribution": y_train.value_counts(dropna=False).to_dict()
            if self.config.problem_type == "classification"
            else None,
            "test_target_available": y_test is not None,
            "limitations": self._data_quality_warnings(
                train_rows=int(len(X_train)),
                test_rows=int(len(X_test)) if X_test is not None else 0,
                feature_names=list(X_train.columns),
            ),
            "tuning_enabled": self.tuning_config.enable_tuning,
            "tuning_models_tuned": len(
                [s for s in self.tuning_summary_.values() if isinstance(s, dict) and s.get("status") == "ok"]
            ),
        }

    def _build_metadata(self, feature_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "llm_plan": self.llm_plan_,
            "available_models": self.available_model_names(),
            "best_model_name": self.best_model_name_,
            "resolved_cv_folds": self.resolved_cv_folds_,
            "leaderboard_columns": self.leaderboard_.columns.tolist() if not self.leaderboard_.empty else [],
            "diagnostics_available": bool(self.diagnostics_),
            "feature_importance_available": not self.best_model_feature_importance_.empty,
            "upstream_feature_metadata": feature_metadata,
            "tuning_config": asdict(self.tuning_config),
            "tuning_enabled": self.tuning_config.enable_tuning,
        }

    def _data_quality_warnings(
        self,
        train_rows: int,
        test_rows: int,
        feature_names: List[str],
    ) -> List[str]:
        warnings: List[str] = []

        if train_rows < 100:
            warnings.append(
                "Training sample size is very small for academic performance claims; treat current metrics as smoke-test evidence only."
            )
        if test_rows < 30:
            warnings.append(
                "Test sample size is too small for stable generalisation claims; prefer a larger held-out test set before reporting benchmark conclusions."
            )

        id_like_features = [
            feature_name
            for feature_name in feature_names
            if re.search(r"(^id$|_id$|^id_|customer_id|user_id|account_id)", feature_name)
        ]
        if id_like_features:
            warnings.append(
                "Potential identifier-derived features detected: "
                + ", ".join(sorted(id_like_features)[:5])
                + ". Review for leakage before drawing substantive conclusions."
            )

        return warnings

    def _save_artifacts(self, result: Dict[str, Any]) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result["leaderboard"].to_csv(output_dir / "leaderboard.csv", index=False)

        # Issue #5: use json_default to handle numpy scalar types (np.int64, np.float64, etc.)
        with open(output_dir / "modelling_summary.json", "w", encoding="utf-8") as file:
            json.dump(result["summary"], file, ensure_ascii=False, indent=2, default=json_default)

        with open(output_dir / "modelling_metadata.json", "w", encoding="utf-8") as file:
            json.dump(result["metadata"], file, ensure_ascii=False, indent=2, default=json_default)

        with open(output_dir / "llm_plan.json", "w", encoding="utf-8") as file:
            json.dump(result["llm_plan"], file, ensure_ascii=False, indent=2, default=json_default)

        with open(output_dir / "best_model_metrics.json", "w", encoding="utf-8") as file:
            json.dump(result["best_model_metrics"], file, ensure_ascii=False, indent=2, default=json_default)

        with open(output_dir / "diagnostics.json", "w", encoding="utf-8") as file:
            json.dump(result["diagnostics"], file, ensure_ascii=False, indent=2, default=json_default)

        best_predictions = result["predictions"].get(self.best_model_name_ or "", {})
        if best_predictions:
            prediction_frame = pd.DataFrame(best_predictions)
            prediction_frame.to_csv(output_dir / "best_model_predictions.csv", index=False)

        if not result["best_model_feature_importance"].empty:
            result["best_model_feature_importance"].to_csv(
                output_dir / "best_model_feature_importance.csv",
                index=False,
            )

        if self.best_model_ is not None:
            joblib.dump(self.best_model_, output_dir / "best_model.joblib")

        if result.get("tuning_summary"):
            self._save_tuning_artifacts(
                result["tuning_summary"], result.get("tuning_history", {})
            )

    def explain(self) -> str:
        if not self.summary_:
            return "Modelling summary is not available yet."

        lines = [
            "Modelling Summary",
            f"- Problem type: {self.summary_['problem_type']}",
            f"- Primary metric: {self.summary_['primary_metric']}",
            f"- Candidate models: {len(self.summary_['candidate_models'])}",
            f"- Best model: {self.summary_['best_model_name']}",
            f"- Train rows: {self.summary_['train_rows']}",
            f"- Train feature count: {self.summary_['train_feature_count']}",
            f"- Test rows: {self.summary_['test_rows']}",
            f"- LLM plan status: {self.summary_['llm_plan_status']}",
            f"- Limitations flagged: {len(self.summary_.get('limitations', []))}",
        ]
        if self.tuning_summary_:
            n_ok = len([s for s in self.tuning_summary_.values() if isinstance(s, dict) and s.get("status") == "ok"])
            lines.append(f"- Tuning: {n_ok} model(s) tuned via Optuna")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2 tuning helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_trial_budget(self, n_samples: int, intensity: str) -> tuple[int, int]:
        """Return (n_trials, cv_folds_tuning) based on dataset size and intensity."""
        if intensity == "light":
            return 30, 3
        if intensity == "full":
            return 150, 5
        # auto: scale with dataset size
        if n_samples < 1_000:
            return 30, 3
        if n_samples < 10_000:
            return 80, 5
        return 150, 5

    def _primary_tuning_scorer(self, y_train: pd.Series) -> str:
        """Return the single scorer string used by Optuna cross_validate calls."""
        primary = self.config.resolved_primary_metric()
        if self.config.problem_type == "classification":
            binary = y_train.nunique() == 2
            return {
                "roc_auc": "roc_auc" if binary else "roc_auc_ovr_weighted",
                "accuracy": "accuracy",
                "f1": "f1" if binary else "f1_weighted",
            }.get(primary, "roc_auc" if binary else "roc_auc_ovr_weighted")
        return {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }.get(primary, "neg_root_mean_squared_error")

    def _primary_cv_col(self) -> str:
        """Return the leaderboard column name for the primary CV metric."""
        return {
            "roc_auc": "cv_roc_auc",
            "accuracy": "cv_accuracy",
            "f1": "cv_f1",
            "rmse": "cv_rmse",
            "mae": "cv_mae",
            "r2": "cv_r2",
        }.get(self.config.resolved_primary_metric(), "cv_roc_auc")

    def _optuna_objective(
        self,
        trial: Any,
        model_name: str,
        base_pipeline: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: Any,
        tuning_scorer: str,
    ) -> float:
        """Optuna objective: sample hyperparams, run cross_validate, return mean score."""
        space = SEARCH_SPACES.get(model_name, {})
        params: Dict[str, Any] = {}
        for param_name, spec in space.items():
            ptype = spec[0]
            if ptype == "int":
                params[param_name] = trial.suggest_int(param_name, int(spec[1]), int(spec[2]))
            elif ptype == "float":
                params[param_name] = trial.suggest_float(param_name, float(spec[1]), float(spec[2]))
            elif ptype == "float_log":
                params[param_name] = trial.suggest_float(
                    param_name, float(spec[1]), float(spec[2]), log=True
                )

        if model_name == "xgboost":
            estimator = self._build_xgb_tuning_estimator(params, y_train)
        else:
            estimator = clone(base_pipeline)
            if params:
                estimator.set_params(**params)

        cv_result = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=tuning_scorer,
            n_jobs=1,
            return_train_score=False,
            error_score=np.nan,
        )
        return float(np.nanmean(cv_result["test_score"]))

    def _tune_model(
        self,
        model_name: str,
        base_pipeline: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int,
        cv_folds_tuning: int,
    ) -> tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Run Optuna TPE search for one model. Returns (best_params, best_score, history)."""
        tuning_scorer = self._primary_tuning_scorer(y_train)

        if self.config.problem_type == "classification":
            class_counts = y_train.value_counts(dropna=False)
            min_class = int(class_counts.min())
            n_splits = max(2, min(cv_folds_tuning, min_class))
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.config.random_state
            )
        else:
            n_splits = max(2, min(cv_folds_tuning, len(y_train)))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_state)

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.config.random_state),
        )
        early_stop = _EarlyStoppingCallback(
            rounds=self.tuning_config.early_stopping_rounds,
            min_improvement=0.01,
        )

        def _objective(trial: Any) -> float:
            return self._optuna_objective(
                trial=trial,
                model_name=model_name,
                base_pipeline=base_pipeline,
                X_train=X_train,
                y_train=y_train,
                cv=cv,
                tuning_scorer=tuning_scorer,
            )

        study.optimize(_objective, n_trials=n_trials, callbacks=[early_stop], catch=(Exception,))

        if not study.trials or study.best_trial is None:
            return {}, float("nan"), []

        history = [
            {
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ]
        return study.best_trial.params, study.best_value, history

    def _build_xgb_tuning_estimator(
        self,
        params: Dict[str, Any],
        y_train: pd.Series,
    ) -> "Pipeline":
        """Build an XGBClassifier Pipeline with given params for tuning objective."""
        n_classes = y_train.nunique()
        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
        estimator = XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            random_state=self.config.random_state,
            use_label_encoder=False,
            **params,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

    def _build_tuned_estimator(
        self,
        model_name: str,
        base_pipeline: Any,
        best_params: Dict[str, Any],
        y_train: pd.Series,
    ) -> Any:
        """Rebuild the estimator with tuned params for final re-evaluation."""
        if model_name == "xgboost":
            return self._build_xgb_tuning_estimator(best_params, y_train)
        est = clone(base_pipeline)
        if best_params:
            est.set_params(**best_params)
        return est

    def _select_stage2_candidates(
        self,
        leaderboard: pd.DataFrame,
        candidates: List[CandidateModel],
    ) -> List[str]:
        """Select top-N models for Stage 2 tuning, with optional LLM override."""
        primary_col = self._primary_cv_col()
        n_top = self.tuning_config.n_top_models_to_tune
        ascending = self.config.resolved_primary_metric() in {"rmse", "mae"}

        if primary_col in leaderboard.columns:
            sorted_lb = leaderboard.sort_values(primary_col, ascending=ascending, na_position="last")
        else:
            sorted_lb = leaderboard
        top_names = sorted_lb.head(n_top)["model_name"].tolist()

        if not self.tuning_config.use_llm_stage2_selection or self.llm is None:
            return top_names

        try:
            llm_names = self._llm_stage2_selection(leaderboard, top_names)
            if llm_names:
                return llm_names
        except Exception:  # noqa: BLE001
            pass

        return top_names

    def _llm_stage2_selection(
        self,
        leaderboard: pd.DataFrame,
        top_names: List[str],
    ) -> List[str]:
        """Ask the LLM which models to tune; returns empty list on any failure."""
        if self.llm is None or HumanMessage is None or SystemMessage is None:
            return []

        display_cols = ["rank", "model_name"]
        primary_col = self._primary_cv_col()
        if primary_col in leaderboard.columns:
            display_cols.append(primary_col)
        leaderboard_str = leaderboard[display_cols].head(10).to_string(index=False)

        system_prompt = """
You are a modelling tuning advisor. Given a Stage 1 leaderboard, select which
models to tune in Stage 2. Return only valid JSON — no extra text.

JSON format:
{"selected_models": ["model_a", "model_b"], "reason": "..."}
"""
        human_prompt = f"""
Stage 1 leaderboard (top 10):
{leaderboard_str}

Available search spaces: {list(SEARCH_SPACES.keys())}

Select up to {self.tuning_config.n_top_models_to_tune} models to tune.
Only select models that appear in the leaderboard AND have a search space.
"""
        try:
            response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            plan = self._extract_json(str(response.content))
            available_names = set(leaderboard["model_name"].tolist())
            selected = [
                m for m in plan.get("selected_models", [])
                if m in available_names and m in SEARCH_SPACES
            ]
            return selected if selected else []
        except Exception:  # noqa: BLE001
            return []

    def _generate_llm_tuning_interpretation(
        self,
        tuning_summary: Dict[str, Any],
    ) -> str:
        """Generate a short LLM text summary of tuning results (optional)."""
        if self.llm is None or HumanMessage is None or SystemMessage is None:
            return ""

        summary_for_llm = {
            k: v for k, v in tuning_summary.items() if not k.startswith("_")
        }
        system_prompt = """
You are a machine learning results interpreter.
Given hyperparameter tuning results, write a brief (2-4 sentence) interpretation.
Focus on which model improved the most and whether tuning was worthwhile.
Return plain text only — no code, no markdown, no JSON.
"""
        human_prompt = f"""
Tuning results:
{json.dumps(summary_for_llm, indent=2, default=json_default)}
"""
        try:
            response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            return str(response.content).strip()
        except Exception:  # noqa: BLE001
            return ""

    def _save_tuning_artifacts(
        self,
        tuning_summary: Dict[str, Any],
        tuning_history: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "tuning_summary.json", "w", encoding="utf-8") as f:
            json.dump(tuning_summary, f, ensure_ascii=False, indent=2, default=json_default)
        with open(output_dir / "tuning_history.json", "w", encoding="utf-8") as f:
            json.dump(tuning_history, f, ensure_ascii=False, indent=2, default=json_default)