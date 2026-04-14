"""Hyperparameter-tuning modelling agent (V2) for tabular learning experiments.

V2 extends V1 with:
  - Full end-to-end regression support (classification + regression parity).
  - PlannerInput interface: accepts a JSON file from an external reasoning /
    planner agent to override task, model, tuning, and feature-selection
    settings programmatically.
  - _normalize_tuning_score: fixes Stage 2 improvement tracking for negated
    regression scorers (neg_rmse, neg_mae).

Stage 1 — quick baseline comparison across all candidates.
Stage 2 — Optuna TPE hyperparameter search on the top-N Stage 1 models,
           with adaptive trial budget, early stopping, and optional LLM
           candidate selection.

V1 artifact format is preserved; V2 adds planner_input.json to the output
directory when a PlannerInput is supplied.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace as dataclass_replace
from datetime import datetime, timezone
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
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
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
    upstream_context: Optional[Dict[str, Any]] = None

    def resolved_primary_metric(self) -> str:
        _CLF_METRICS = {"roc_auc", "f1", "accuracy", "precision", "recall"}
        _REG_METRICS  = {"rmse", "mae", "r2", "mse"}
        metric = self.primary_metric
        if metric:
            # If the LLM returned a metric that belongs to the wrong problem type,
            # silently replace it with the correct default.
            if self.problem_type == "regression" and metric in _CLF_METRICS:
                return "rmse"
            if self.problem_type == "classification" and metric in _REG_METRICS:
                return "roc_auc"
            return metric
        return "roc_auc" if self.problem_type == "classification" else "rmse"


@dataclass
class TuningConfig:
    """Configuration for Stage 2 Optuna hyperparameter search."""

    enable_tuning: bool = True
    n_top_models_to_tune: int = 3
    early_stopping_rounds: int = 10  # consecutive non-improving trials before study stops
    tuning_intensity: str = "auto"  # "auto" | "light" | "full"
    use_llm_stage2_selection: bool = True  # falls back to top-N Python when LLM unavailable


@dataclass
class PlannerInput:
    """Structured contract from an external reasoning / planner agent.

    Produced by an upstream planning component and loaded via
    ``load_planner_input(path)``.  Fields set to ``None`` do not override
    the corresponding ``ModellingConfig`` or ``TuningConfig`` setting — only
    explicitly provided values take effect.

    Supported JSON schemas
    ----------------------
    Flat (all top-level keys)::

        {
          "source": "autods_planner_v1",
          "problem_type": "regression",
          "primary_metric": "rmse",
          "candidate_models": ["xgboost_regressor"],
          "enable_tuning": true,
          "tuning_intensity": "auto"
        }

    Nested (grouped keys — also accepted)::

        {
          "source": "autods_planner_v1",
          "task": {"problem_type": "regression", "primary_metric": "rmse"},
          "models": {"candidate_models": ["xgboost_regressor"]},
          "tuning": {"enable_tuning": true, "tuning_intensity": "auto"},
          "features": {"drop_columns": ["id"]}
        }

    Flat keys take precedence over nested block keys when both are present.
    """

    # ── provenance ──────────────────────────────────────────────────────────
    source: str = "unknown"
    schema_version: str = "1.0"
    rationale: str = ""

    # ── task overrides (None = use ModellingConfig value as-is) ────────────
    problem_type: Optional[str] = None
    primary_metric: Optional[str] = None
    task_description: Optional[str] = None
    candidate_models: Optional[List[str]] = None

    # ── tuning overrides ────────────────────────────────────────────────────
    enable_tuning: Optional[bool] = None
    tuning_intensity: Optional[str] = None
    n_top_models_to_tune: Optional[int] = None

    # ── feature hints ───────────────────────────────────────────────────────
    drop_columns: Optional[List[str]] = None   # remove from X before training
    use_columns: Optional[List[str]] = None    # keep only these in X (takes precedence over drop)

    # ── pass-through context ─────────────────────────────────────────────────
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerInput":
        """Deserialise from a dict supporting both flat and nested JSON formats."""
        task_block = data.get("task", {})
        tuning_block = data.get("tuning", {})
        features_block = data.get("features", data.get("feature_hints", {}))
        models_block = data.get("models", {})

        def _flat_or_block(flat_key: str, block: Dict[str, Any]) -> Any:
            return data[flat_key] if flat_key in data else block.get(flat_key)

        def _bool_or_block(flat_key: str, block: Dict[str, Any]) -> Optional[bool]:
            """Handle booleans carefully so False is not confused with missing."""
            if flat_key in data:
                return bool(data[flat_key])
            if flat_key in block:
                return bool(block[flat_key])
            return None

        known_keys = {
            "schema_version", "source", "rationale",
            "task", "tuning", "features", "feature_hints", "models",
            "problem_type", "primary_metric", "task_description", "candidate_models",
            "enable_tuning", "tuning_intensity", "n_top_models_to_tune",
            "drop_columns", "use_columns",
        }
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            source=data.get("source", "unknown"),
            rationale=data.get("rationale", ""),
            problem_type=_flat_or_block("problem_type", task_block),
            primary_metric=_flat_or_block("primary_metric", task_block),
            task_description=_flat_or_block("task_description", task_block),
            candidate_models=_flat_or_block("candidate_models", models_block),
            enable_tuning=_bool_or_block("enable_tuning", tuning_block),
            tuning_intensity=_flat_or_block("tuning_intensity", tuning_block),
            n_top_models_to_tune=_flat_or_block("n_top_models_to_tune", tuning_block),
            drop_columns=_flat_or_block("drop_columns", features_block),
            use_columns=_flat_or_block("use_columns", features_block),
            extra={k: v for k, v in data.items() if k not in known_keys},
        )

    @classmethod
    def from_json_file(cls, path: str) -> "PlannerInput":
        """Load from a JSON file path."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def apply_to_config(self, config: ModellingConfig) -> ModellingConfig:
        """Return a copy of ``config`` with planner overrides applied."""
        overrides: Dict[str, Any] = {}
        if self.problem_type is not None:
            overrides["problem_type"] = self.problem_type
        if self.primary_metric is not None:
            overrides["primary_metric"] = self.primary_metric
        if self.task_description:
            overrides["task_description"] = self.task_description
        if self.candidate_models is not None:
            overrides["candidate_model_names"] = self.candidate_models
        return dataclass_replace(config, **overrides) if overrides else config

    def apply_to_tuning_config(self, tuning_config: TuningConfig) -> TuningConfig:
        """Return a copy of ``tuning_config`` with planner overrides applied."""
        overrides: Dict[str, Any] = {}
        if self.enable_tuning is not None:
            overrides["enable_tuning"] = self.enable_tuning
        if self.tuning_intensity is not None:
            overrides["tuning_intensity"] = self.tuning_intensity
        if self.n_top_models_to_tune is not None:
            overrides["n_top_models_to_tune"] = self.n_top_models_to_tune
        return dataclass_replace(tuning_config, **overrides) if overrides else tuning_config


def load_planner_input(path: str) -> PlannerInput:
    """Load a ``PlannerInput`` from a JSON file.  Raises ``FileNotFoundError`` if the path does not exist."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Planner input file not found: {path}")
    return PlannerInput.from_json_file(path)


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
        planner_input: Optional[PlannerInput] = None,
    ):
        self.planner_input_: Optional[PlannerInput] = planner_input
        # Apply planner overrides before freezing config objects
        if planner_input is not None:
            config = planner_input.apply_to_config(config)
            tuning_config = planner_input.apply_to_tuning_config(
                tuning_config if tuning_config is not None else TuningConfig()
            )
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
        self.threshold_optimization_: Dict[str, Any] = {}
        self.best_threshold_: Optional[float] = None
        # ── upstream context ─────────────────────────────────────────────────
        self.upstream_context_: Dict[str, Any] = dict(config.upstream_context or {})
        self._upstream_decisions_: List[Dict[str, str]] = []
        self._upstream_warnings_: List[str] = []
        self._upstream_skip_models_: set = set()
        self._upstream_skip_reasons_: Dict[str, str] = {}
        self._upstream_apply_balanced_lightgbm_: bool = False
        self._upstream_feature_names_final_: Optional[List[str]] = None
        self.stage_handoff_: Dict[str, Any] = {}

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

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        feature_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _run_start_ts = datetime.now(timezone.utc).isoformat()

        # ── Apply planner feature hints before validation ──────────────────
        if self.planner_input_ is not None:
            if self.planner_input_.use_columns:
                available = [c for c in self.planner_input_.use_columns if c in X.columns]
                if available:
                    X = X[available].copy()
            elif self.planner_input_.drop_columns:
                to_drop = [c for c in self.planner_input_.drop_columns if c in X.columns]
                if to_drop:
                    X = X.drop(columns=to_drop).copy()

        # ── Absorb upstream stage context (data-scale rules, cv_folds, etc.) ─
        self._absorb_upstream_context(n_rows=len(X), n_cols=X.shape[1])

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
                # V2 fix: normalize raw Optuna score (may be negative for neg_* scorers)
                # to the same positive scale as the leaderboard cv_* columns before
                # computing improvement.
                normalized_score = self._normalize_tuning_score(best_score)
                is_lower_better = self.config.resolved_primary_metric() in {"rmse", "mae"}
                if not (np.isnan(normalized_score) or np.isnan(stage1_cv_score)):
                    if is_lower_better:
                        # for RMSE/MAE: positive improvement means score got smaller
                        improvement = round(stage1_cv_score - normalized_score, 6)
                    else:
                        improvement = round(normalized_score - stage1_cv_score, 6)
                else:
                    improvement = None
                tuning_summary[model_name] = {
                    "status": "ok",
                    "best_tuned_cv_score": normalized_score,
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

        # ── Threshold optimisation (binary classification only) ────────────────
        best_preds = prediction_outputs.get(self.best_model_name_ or "", {})
        _is_binary = (
            self.config.problem_type == "classification"
            and y_train.nunique() == 2
            and "test_scores" in best_preds
            and y_test is not None
        )
        if _is_binary:
            self.threshold_optimization_ = self._compute_threshold_optimization(
                X_train=X_train,
                y_train=y_train,
                y_test=y_test,
                test_scores=np.asarray(best_preds["test_scores"]),
            )
            self.best_threshold_ = (
                self.threshold_optimization_.get("selected_threshold")
                if self.threshold_optimization_.get("enabled")
                else None
            )
        else:
            self.threshold_optimization_ = {"enabled": False, "reason": "not_binary_classification"}

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
            threshold_optimization=self.threshold_optimization_,
        )
        self.metadata_ = self._build_metadata(feature_metadata=feature_metadata)

        # ── Stage handoff ─────────────────────────────────────────────────────
        _models_attempted = [r["model_name"] for r in leaderboard_rows]
        _models_succeeded = [r["model_name"] for r in leaderboard_rows if "training_error" not in r]
        _training_errors = {
            r["model_name"]: r["training_error"]
            for r in leaderboard_rows
            if "training_error" in r
        }
        # Merge upstream-skipped models into the skipped-reason log
        _skipped_all = dict(self._upstream_skip_reasons_)
        _skipped_all.update(_training_errors)

        _primary_metric = self.config.resolved_primary_metric()
        _primary_cv_col = self._primary_cv_col()
        _best_score_raw = self.best_model_metrics_.get(_primary_cv_col, np.nan)
        _best_score = float(_best_score_raw) if not (
            isinstance(_best_score_raw, float) and np.isnan(_best_score_raw)
        ) else None

        # Top-3 leaderboard snapshot
        _lb_cols = ["model_name", _primary_cv_col]
        _lb_available = [c for c in _lb_cols if c in leaderboard.columns]
        _top3_rows = leaderboard[_lb_available].iloc[:3].to_dict(orient="records")
        _leaderboard_top3 = [
            {
                "model": str(row.get("model_name", "")),
                "score": float(row[_primary_cv_col]) if _primary_cv_col in row and not (
                    isinstance(row[_primary_cv_col], float) and np.isnan(row[_primary_cv_col])
                ) else None,
            }
            for row in _top3_rows
        ]

        # One-line summary for downstream LLM consumption
        _n_ok = len(_models_succeeded)
        _n_total = len(_models_attempted) + len(self._upstream_skip_models_)
        _score_str = f"{_best_score:.4f}" if _best_score is not None else "n/a"
        _summary_line = (
            f"Best model: {self.best_model_name_} "
            f"({_primary_metric}={_score_str}) on {self.config.problem_type} task; "
            f"{_n_ok}/{_n_total} models trained successfully."
        )

        self.stage_handoff_ = {
            "stage_id": "stage_4",
            "stage_name": "ModellingAgent",
            "status": "success",
            "timestamp": _run_start_ts,
            "summary": _summary_line,
            "key_outputs": {
                "models_attempted": _models_attempted,
                "models_succeeded": _models_succeeded,
                "best_model_name": self.best_model_name_,
                "best_model_score": _best_score,
                "primary_metric_used": _primary_metric,
                "leaderboard_top3": _leaderboard_top3,
                "training_skipped_reason": _skipped_all,
            },
            "decisions": list(self._upstream_decisions_),
            "warnings": list(self._upstream_warnings_),
        }

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
            "threshold_optimization": self.threshold_optimization_,
            "stage_handoff": self.stage_handoff_,
        }

        if self.config.save_artifacts:
            self._save_artifacts(result)

        return result

    def _absorb_upstream_context(self, n_rows: int, n_cols: int) -> None:
        """Translate upstream stage_handoff context into behaviour adjustments.

        Called once at the start of ``run()`` after planner feature hints are
        applied.  The method mutates internal state only — it never raises; any
        condition that cannot be satisfied is recorded as a warning instead.

        Context keys consumed
        ---------------------
        class_imbalance_ratio : float   majority-class fraction (Stage 1 output)
        n_rows                : int     row count at the time of understanding/cleaning
        n_cols                : int     feature count at the time of feature engineering
        train_size            : int     number of training rows (Stage 3 output)
        feature_names_final   : list    final feature column names (Stage 3 output)
        """
        ctx = self.upstream_context_
        if not ctx:
            return

        decisions = self._upstream_decisions_
        warnings_ = self._upstream_warnings_

        # ── class imbalance → class_weight for LightGBM ──────────────────────
        imbalance_ratio = ctx.get("class_imbalance_ratio")
        if (
            imbalance_ratio is not None
            and float(imbalance_ratio) > 0.9
            and self.config.problem_type == "classification"
        ):
            self._upstream_apply_balanced_lightgbm_ = True
            decisions.append({
                "decision": "Enable class_weight='balanced' for LightGBM classifier",
                "reason": (
                    f"Upstream class_imbalance_ratio={float(imbalance_ratio):.2f} > 0.9; "
                    "LR / RF / SVM already use class_weight='balanced' by default"
                ),
            })
            warnings_.append(
                f"Severe class imbalance (majority class ratio {float(imbalance_ratio):.2f}). "
                "LightGBM will use class_weight='balanced'. "
                "XGBoost does not support class_weight directly — consider providing "
                "scale_pos_weight via PlannerInput if XGBoost is selected as the best model."
            )

        # ── data scale → model skip list ─────────────────────────────────────
        effective_rows = int(ctx.get("n_rows") or n_rows)

        if effective_rows < 50:
            for m in ("svm_rbf", "svr_rbf"):
                self._upstream_skip_models_.add(m)
                reason = (
                    f"Skipped by upstream context: dataset too small "
                    f"(n_rows={effective_rows} < 50) — kernel SVM unstable with very few samples"
                )
                self._upstream_skip_reasons_[m] = reason
                decisions.append({"decision": f"Skip {m}", "reason": reason})

        if effective_rows > 50_000:
            for m in ("svm_rbf", "svr_rbf"):
                if m not in self._upstream_skip_models_:
                    self._upstream_skip_models_.add(m)
                    reason = (
                        f"Skipped by upstream context: dataset too large "
                        f"(n_rows={effective_rows} > 50 000) — kernel SVM training time impractical"
                    )
                    self._upstream_skip_reasons_[m] = reason
                    decisions.append({"decision": f"Skip {m}", "reason": reason})

        # ── train size → CV folds cap ─────────────────────────────────────────
        upstream_train_size = ctx.get("train_size")
        if upstream_train_size is not None and int(upstream_train_size) < 150:
            old_folds = self.resolved_cv_folds_
            self.resolved_cv_folds_ = min(self.resolved_cv_folds_, 3)
            if self.resolved_cv_folds_ != old_folds:
                decisions.append({
                    "decision": f"Reduce CV folds from {old_folds} to {self.resolved_cv_folds_}",
                    "reason": (
                        f"Upstream train_size={upstream_train_size} < 150; "
                        "capped at 3 folds to avoid empty-fold errors with small datasets"
                    ),
                })
                warnings_.append(
                    f"Upstream train_size={upstream_train_size} is small; "
                    f"CV folds reduced to {self.resolved_cv_folds_}."
                )

        # ── feature names alignment ──────────────────────────────────────────
        upstream_features = ctx.get("feature_names_final")
        if upstream_features:
            self._upstream_feature_names_final_ = list(upstream_features)
            decisions.append({
                "decision": "Reference upstream feature_names_final for feature importance alignment",
                "reason": (
                    f"Stage 3 provided {len(self._upstream_feature_names_final_)} final feature names; "
                    "used to sanity-check feature importance column alignment"
                ),
            })

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
            filtered = [model for model in models if model.name in requested_set]
            if filtered:
                models = filtered
            else:
                # None of the requested names matched available models — fall back to
                # all defaults rather than crashing.  This happens when the LLM returns
                # names for the wrong problem type (e.g. classification names for a
                # regression task) or names that don't exist in the registry.
                print(
                    f"  [ModellingAgent] ⚠ Requested models {list(requested_names)} "
                    f"not found in {self.config.problem_type} registry — using all defaults."
                )

        # Filter models skipped due to upstream context (data-scale rules)
        if self._upstream_skip_models_:
            models = [m for m in models if m.name not in self._upstream_skip_models_]

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
            lgbm_kwargs: Dict[str, Any] = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "random_state": self.config.random_state,
                "verbose": -1,
            }
            if self._upstream_apply_balanced_lightgbm_:
                lgbm_kwargs["class_weight"] = "balanced"
            models.append(
                CandidateModel(
                    name="lightgbm",
                    estimator=self._make_numeric_pipeline(LGBMClassifier(**lgbm_kwargs)),
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

    def _scoring_metrics(self, y_train: pd.Series) -> Dict[str, str]:
        if self.config.problem_type == "classification":
            n_classes = y_train.nunique()
            binary = n_classes == 2
            # P0: multi-class requires averaging; binary uses default (no average kwarg needed)
            roc_auc_scorer = "roc_auc" if binary else "roc_auc_ovr_weighted"
            precision_scorer = "precision" if binary else "precision_weighted"
            recall_scorer = "recall" if binary else "recall_weighted"
            f1_scorer = "f1" if binary else "f1_weighted"
            return {
                "accuracy": "accuracy",
                "precision": precision_scorer,
                "recall": recall_scorer,
                "f1": f1_scorer,
                "roc_auc": roc_auc_scorer,
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
        avg = "binary" if binary else "weighted"
        metrics = {
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
            "test_recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
            "test_f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
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
        threshold_optimization: Optional[Dict[str, Any]] = None,
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
            if threshold_optimization and threshold_optimization.get("enabled"):
                diagnostics["threshold_optimization"] = threshold_optimization
        else:
            residuals = pd.Series(y_test).reset_index(drop=True) - pd.Series(y_pred)
            diagnostics["residual_summary"] = {
                "mean": float(residuals.mean()),
                "std": float(residuals.std(ddof=0)),
                "min": float(residuals.min()),
                "max": float(residuals.max()),
            }

        return diagnostics

    def _compute_threshold_optimization(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        test_scores: np.ndarray,
    ) -> Dict[str, Any]:
        """Binary classification threshold optimisation (literature-standard, OOF-based).

        Uses out-of-fold (OOF) probabilities from the best model to fit the
        decision threshold, eliminating test-set leakage in threshold selection.

        Four methods are computed and all results are reported:

        Youden's J  — argmax(Sensitivity + Specificity − 1) via ROC curve.
                      Maximises information gain; standard in medical diagnostics
                      (Youden, 1950; Fluss et al., 2005).
        F1-optimal  — argmax F₁ via precision-recall curve (exact, no grid search).
                      Preferred when class imbalance is present (Davis & Goadrich 2006).
        G-mean      — argmax √(TPR × TNR) via ROC curve.
                      Geometric mean; robust to severe imbalance (Kubat & Matwin 1997).
        MCC-optimal — argmax Matthews Correlation Coefficient via candidate thresholds.
                      Single best metric for imbalanced binary problems
                      (Chicco & Jurman, BMC Genomics 2020).

        The *selected* threshold is the one that directly optimises
        ``self.config.resolved_primary_metric()``:
          roc_auc / accuracy → Youden's J
          f1 / precision / recall → F1-optimal

        Returns
        -------
        Dict with keys: enabled, oof_n_samples, primary_metric, selected_method,
        selected_threshold, methods, test_metrics_at_default_0.5,
        test_metrics_at_optimal.
        """
        # ── Step 1: OOF probabilities via cross_val_predict ────────────────
        cv = self._cross_validator(y_train)
        try:
            oof_probs = cross_val_predict(
                clone(self.best_model_),
                X_train,
                y_train,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )
        except Exception as exc:  # noqa: BLE001
            return {"enabled": False, "reason": f"OOF CV failed: {exc}"}

        oof_scores: np.ndarray = (
            oof_probs[:, 1] if oof_probs.ndim == 2 else oof_probs.ravel()
        )
        y_train_arr = np.asarray(y_train)

        # ── Step 2: per-method optimal thresholds ─────────────────────────
        methods: Dict[str, Any] = {}

        # Youden's J  (ROC curve)
        try:
            fpr, tpr, roc_thresh = roc_curve(y_train_arr, oof_scores)
            j = tpr - fpr
            j_idx = int(np.argmax(j))
            methods["youden_j"] = {
                "threshold": float(roc_thresh[j_idx]),
                "sensitivity": float(tpr[j_idx]),
                "specificity": float(1.0 - fpr[j_idx]),
                "j_statistic": float(j[j_idx]),
                "reference": "Youden (1950); Fluss et al. (2005)",
            }
        except Exception as exc:  # noqa: BLE001
            methods["youden_j"] = {"error": str(exc)}

        # F1-optimal  (precision-recall curve — exact, no brute-force grid)
        try:
            prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_train_arr, oof_scores)
            # precision_recall_curve returns n+1 precision/recall points for n thresholds
            denom = prec_arr[:-1] + rec_arr[:-1]
            f1_arr = np.where(
                denom > 0,
                2.0 * prec_arr[:-1] * rec_arr[:-1] / denom,
                0.0,
            )
            if len(f1_arr) > 0:
                f1_idx = int(np.argmax(f1_arr))
                methods["f1_optimal"] = {
                    "threshold": float(pr_thresh[f1_idx]),
                    "f1": float(f1_arr[f1_idx]),
                    "precision": float(prec_arr[f1_idx]),
                    "recall": float(rec_arr[f1_idx]),
                    "reference": "Davis & Goadrich (2006)",
                }
            else:
                methods["f1_optimal"] = {"error": "empty PR curve"}
        except Exception as exc:  # noqa: BLE001
            methods["f1_optimal"] = {"error": str(exc)}

        # G-mean  (ROC curve)
        try:
            g_mean_arr = np.sqrt(tpr * np.maximum(0.0, 1.0 - fpr))
            g_idx = int(np.argmax(g_mean_arr))
            methods["g_mean"] = {
                "threshold": float(roc_thresh[g_idx]),
                "g_mean": float(g_mean_arr[g_idx]),
                "sensitivity": float(tpr[g_idx]),
                "specificity": float(1.0 - fpr[g_idx]),
                "reference": "Kubat & Matwin (1997)",
            }
        except Exception as exc:  # noqa: BLE001
            methods["g_mean"] = {"error": str(exc)}

        # MCC-optimal  (candidate-threshold scan — deduplicated + bounded)
        try:
            candidate_thresh = np.unique(oof_scores)
            if len(candidate_thresh) > 500:
                candidate_thresh = np.percentile(oof_scores, np.linspace(1, 99, 300))
            mcc_vals = np.array(
                [
                    matthews_corrcoef(y_train_arr, (oof_scores >= t).astype(int))
                    for t in candidate_thresh
                ]
            )
            mcc_idx = int(np.argmax(mcc_vals))
            methods["mcc_optimal"] = {
                "threshold": float(candidate_thresh[mcc_idx]),
                "mcc": float(mcc_vals[mcc_idx]),
                "reference": "Chicco & Jurman (BMC Genomics 2020)",
            }
        except Exception as exc:  # noqa: BLE001
            methods["mcc_optimal"] = {"error": str(exc)}

        # ── Step 3: select primary threshold ──────────────────────────────
        primary = self.config.resolved_primary_metric()
        method_map = {
            "roc_auc":  "youden_j",
            "accuracy": "youden_j",
            "f1":       "f1_optimal",
            "precision":"f1_optimal",
            "recall":   "f1_optimal",
        }
        selected_method = method_map.get(primary, "youden_j")
        selected_entry = methods.get(selected_method, {})
        if "error" in selected_entry:
            # fallback to Youden's J
            selected_method = "youden_j"
            selected_entry = methods.get("youden_j", {})
        selected_threshold = float(selected_entry.get("threshold", 0.5))

        # ── Step 4: test-set metrics at default 0.5 vs optimal threshold ──
        y_test_arr = np.asarray(y_test)

        def _test_metrics_at(t: float) -> Dict[str, Any]:
            y_pred_t = (test_scores >= t).astype(int)
            return {
                "threshold": round(float(t), 6),
                "accuracy":  round(float(accuracy_score(y_test_arr, y_pred_t)), 6),
                "precision": round(float(precision_score(y_test_arr, y_pred_t, zero_division=0)), 6),
                "recall":    round(float(recall_score(y_test_arr, y_pred_t, zero_division=0)), 6),
                "f1":        round(float(f1_score(y_test_arr, y_pred_t, zero_division=0)), 6),
                "mcc":       round(float(matthews_corrcoef(y_test_arr, y_pred_t)), 6),
            }

        at_default = _test_metrics_at(0.5)
        at_optimal = _test_metrics_at(selected_threshold)

        # delta shows per-metric absolute improvement at optimal vs default
        delta = {
            k: round(at_optimal[k] - at_default[k], 6)
            for k in ("accuracy", "precision", "recall", "f1", "mcc")
        }

        return {
            "enabled": True,
            "oof_n_samples": int(len(oof_scores)),
            "primary_metric": primary,
            "selected_method": selected_method,
            "selected_threshold": round(selected_threshold, 6),
            "methods": methods,
            "test_metrics_at_default_0.5": at_default,
            "test_metrics_at_optimal": at_optimal,
            "delta_optimal_vs_default": delta,
        }

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
            "planner_input_source": self.planner_input_.source if self.planner_input_ is not None else None,
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
            "planner_input_source": self.planner_input_.source if self.planner_input_ is not None else None,
            "planner_input_available": self.planner_input_ is not None,
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

        if self.planner_input_ is not None:
            with open(output_dir / "planner_input.json", "w", encoding="utf-8") as file:
                json.dump(
                    asdict(self.planner_input_), file, ensure_ascii=False, indent=2, default=json_default
                )

        if self.threshold_optimization_.get("enabled"):
            with open(output_dir / "threshold_optimization.json", "w", encoding="utf-8") as file:
                json.dump(
                    self.threshold_optimization_, file, ensure_ascii=False, indent=2, default=json_default
                )

        if self.stage_handoff_:
            with open(output_dir / "stage_handoff.json", "w", encoding="utf-8") as file:
                json.dump(
                    self.stage_handoff_, file, ensure_ascii=False, indent=2, default=json_default
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
        if self.planner_input_ is not None:
            lines.append(f"- Planner input: {self.planner_input_.source}")
        if self.best_threshold_ is not None:
            method = self.threshold_optimization_.get("selected_method", "n/a")
            delta_f1 = self.threshold_optimization_.get("delta_optimal_vs_default", {}).get("f1", 0.0)
            lines.append(
                f"- Optimal threshold: {self.best_threshold_:.4f}"
                f" (method: {method}, ΔF1 vs 0.5: {delta_f1:+.4f})"
            )
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2 tuning helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_tuning_score(self, raw_optuna_score: float) -> float:
        """Convert a raw Optuna objective value to the same scale as leaderboard cv_* columns.

        Optuna always maximizes.  For regression metrics stored as negated values
        (neg_rmse, neg_mae), the raw study.best_value is negative.  The
        leaderboard stores the absolute value (positive RMSE / MAE).
        roc_auc / r2 / accuracy / f1 are already in a natural range — returned as-is.
        """
        if self.config.resolved_primary_metric() in {"rmse", "mae"}:
            return abs(raw_optuna_score)
        return raw_optuna_score

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