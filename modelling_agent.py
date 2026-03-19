"""Baseline modelling agent for downstream tabular learning experiments.

This module prioritises reproducible baseline comparison over broad AutoML
coverage. The current implementation is intentionally scoped to the first
project deliverable: binary classification with preserved interfaces for later
problem families.
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

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


class ModellingAgent:
    def __init__(self, config: ModellingConfig):
        self.config = config
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
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        feature_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._validate_inputs(X_train, y_train, X_test, y_test)

        self.llm_plan_ = self._generate_llm_plan(X_train, y_train, feature_metadata)
        candidates = self._get_candidate_models()
        leaderboard_rows: List[Dict[str, Any]] = []
        fitted_models: Dict[str, Any] = {}
        prediction_outputs: Dict[str, Dict[str, List[Any]]] = {}

        for candidate in candidates:
            row, fitted_model, predictions = self._evaluate_candidate(
                candidate=candidate,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            leaderboard_rows.append(row)
            fitted_models[candidate.name] = fitted_model
            prediction_outputs[candidate.name] = predictions

        leaderboard = pd.DataFrame(leaderboard_rows)
        leaderboard = self._rank_leaderboard(leaderboard)
        self.leaderboard_ = leaderboard

        best_row = leaderboard.iloc[0].to_dict()
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
        }

        if self.config.save_artifacts:
            self._save_artifacts(result)

        return result

    def _validate_inputs(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame],
        y_test: Optional[pd.Series],
    ) -> None:
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data must not be empty.")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of rows.")

        if X_test is not None and not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame when provided.")
        if y_test is not None and not isinstance(y_test, pd.Series):
            raise TypeError("y_test must be a pandas Series when provided.")
        if X_test is not None and y_test is not None and len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have the same number of rows.")

        if self.config.problem_type not in self.config.supported_future_problem_families:
            raise ValueError(f"Unsupported problem type: {self.config.problem_type}")

        if self.config.problem_type in {"unsupervised", "nlp"}:
            raise NotImplementedError(
                f"Problem type '{self.config.problem_type}' is reserved for a later project phase."
            )
        if self.config.problem_type == "classification" and y_train.nunique(dropna=True) < 2:
            raise ValueError("Classification requires at least two target classes in y_train.")

    def _generate_llm_plan(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.llm is None:
            return {
                "planner_status": "disabled_or_unavailable",
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

    def _classification_models(self) -> List[CandidateModel]:
        models: List[CandidateModel] = [
            CandidateModel(
                name="logistic_regression",
                estimator=Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=1000,
                                class_weight="balanced",
                                random_state=self.config.random_state,
                            ),
                        ),
                    ]
                ),
            ),
            CandidateModel(
                name="random_forest",
                estimator=Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=300,
                                class_weight="balanced",
                                random_state=self.config.random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ),
            CandidateModel(
                name="svm_rbf",
                estimator=Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", SVC(probability=True, class_weight="balanced", random_state=self.config.random_state)),
                    ]
                ),
            ),
        ]

        if XGBClassifier is not None:
            models.append(
                CandidateModel(
                    name="xgboost",
                    estimator=Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                XGBClassifier(
                                    n_estimators=300,
                                    max_depth=6,
                                    learning_rate=0.05,
                                    subsample=0.9,
                                    colsample_bytree=0.9,
                                    objective="binary:logistic",
                                    eval_metric="logloss",
                                    random_state=self.config.random_state,
                                ),
                            ),
                        ]
                    ),
                )
            )

        if LGBMClassifier is not None:
            models.append(
                CandidateModel(
                    name="lightgbm",
                    estimator=Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                LGBMClassifier(
                                    n_estimators=300,
                                    learning_rate=0.05,
                                    random_state=self.config.random_state,
                                    verbose=-1,
                                ),
                            ),
                        ]
                    ),
                )
            )

        return models

    def _regression_models(self) -> List[CandidateModel]:
        models: List[CandidateModel] = [
            CandidateModel(
                name="ridge_regression",
                estimator=Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", Ridge(random_state=self.config.random_state)),
                    ]
                ),
            ),
            CandidateModel(
                name="random_forest_regressor",
                estimator=Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=300,
                                random_state=self.config.random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            ),
            CandidateModel(
                name="svr_rbf",
                estimator=Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", SVR()),
                    ]
                ),
            ),
        ]

        if XGBRegressor is not None:
            models.append(
                CandidateModel(
                    name="xgboost_regressor",
                    estimator=Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                XGBRegressor(
                                    n_estimators=300,
                                    max_depth=6,
                                    learning_rate=0.05,
                                    subsample=0.9,
                                    colsample_bytree=0.9,
                                    objective="reg:squarederror",
                                    random_state=self.config.random_state,
                                ),
                            ),
                        ]
                    ),
                )
            )

        if LGBMRegressor is not None:
            models.append(
                CandidateModel(
                    name="lightgbm_regressor",
                    estimator=Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                LGBMRegressor(
                                    n_estimators=300,
                                    learning_rate=0.05,
                                    random_state=self.config.random_state,
                                    verbose=-1,
                                ),
                            ),
                        ]
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
        scoring = self._scoring_metrics()
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
                predictions["test_scores"] = np.asarray(y_score).tolist() if y_score is not None else []
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

    def _scoring_metrics(self) -> Dict[str, str]:
        if self.config.problem_type == "classification":
            return {
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
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
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
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
        metrics = {
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "test_roc_auc": np.nan,
        }
        if y_score is not None and len(np.unique(y_true)) == 2:
            metrics["test_roc_auc"] = float(roc_auc_score(y_true, y_score))
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

        sort_column_map = {
            "roc_auc": "test_roc_auc" if "test_roc_auc" in leaderboard.columns else "cv_roc_auc",
            "accuracy": "test_accuracy" if "test_accuracy" in leaderboard.columns else "cv_accuracy",
            "f1": "test_f1" if "test_f1" in leaderboard.columns else "cv_f1",
            "rmse": "test_rmse" if "test_rmse" in leaderboard.columns else "cv_rmse",
            "mae": "test_mae" if "test_mae" in leaderboard.columns else "cv_mae",
            "r2": "test_r2" if "test_r2" in leaderboard.columns else "cv_r2",
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

        with open(output_dir / "modelling_summary.json", "w", encoding="utf-8") as file:
            json.dump(result["summary"], file, ensure_ascii=False, indent=2)

        with open(output_dir / "modelling_metadata.json", "w", encoding="utf-8") as file:
            json.dump(result["metadata"], file, ensure_ascii=False, indent=2)

        with open(output_dir / "llm_plan.json", "w", encoding="utf-8") as file:
            json.dump(result["llm_plan"], file, ensure_ascii=False, indent=2)

        with open(output_dir / "best_model_metrics.json", "w", encoding="utf-8") as file:
            json.dump(result["best_model_metrics"], file, ensure_ascii=False, indent=2)

        with open(output_dir / "diagnostics.json", "w", encoding="utf-8") as file:
            json.dump(result["diagnostics"], file, ensure_ascii=False, indent=2)

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
        return "\n".join(lines)