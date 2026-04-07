"""
AutoDS Evaluation Agent

This module implements the Evaluation Agent for the AutoDS multi-agent workflow.

The Evaluation Agent is responsible for consuming modelling artifacts,
performing benchmark comparison across candidate models, validating
best-model selection consistency, and generating structured technical
evaluation outputs for downstream reporting.

Supported task types:
- binary classification
- multiclass classification
- regression

Design principles:
- deterministic and programmatic evaluation
- task-aware evaluation logic
- structured outputs for downstream agents
- extensibility for future task types
"""

from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd


METRIC_DIRECTION = {

# Metric optimisation direction map.
# True  -> higher metric values are better
# False -> lower metric values are better

    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
    "macro_precision": True,
    "macro_recall": True,
    "macro_f1": True,
    "weighted_precision": True,
    "weighted_recall": True,
    "weighted_f1": True,
    "micro_precision": True,
    "micro_recall": True,
    "micro_f1": True,
    "balanced_accuracy": True,
    "r2": True,
    "explained_variance": True,
    "mae": False,
    "mse": False,
    "rmse": False,
    "median_absolute_error": False,
}


CLASSIFICATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "balanced_accuracy",
}

REGRESSION_METRICS = {
    "mae",
    "mse",
    "rmse",
    "r2",
    "explained_variance",
    "median_absolute_error",
}


@dataclass
class EvaluationConfig:
    modelling_output_dir: str = "model_outputs"
    output_dir: str = "evaluation_outputs"
    save_artifacts: bool = True


def is_higher_better(metric_name: Optional[str]) -> bool:
    if not metric_name:
        return True
    return METRIC_DIRECTION.get(metric_name, True)


def _normalise_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    alias_groups = {
        "y_true": ["y_true", "test_truth", "truth", "actual", "actuals", "label_true", "target_true"],
        "y_pred": ["y_pred", "test_predictions", "prediction", "predictions", "pred", "label_pred", "target_pred"],
        "y_proba": ["y_proba", "test_scores", "score", "scores", "probability", "probabilities", "proba"],
    }

    existing_cols = list(df.columns)
    lower_to_original = {str(c).lower(): c for c in existing_cols}

    for canonical, aliases in alias_groups.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            original = lower_to_original.get(alias.lower())
            if original is not None:
                rename_map[original] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def safe_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


class BaseTaskEvaluator(ABC):
    '''
    Abstract base class for task-specific evaluators.

    This class defines the shared interface used by all evaluation subtypes.
    Each concrete evaluator is responsible for producing structured benchmark summaries, 
    best-model selection evidence, task-specific metric evaluation, and structured error analysis
    for downstream reporting.

    Subclasses currently include:
    - BinaryClassificationEvaluator
    - MulticlassClassificationEvaluator
    - RegressionEvaluator
    - GenericClassificationEvaluator
    """
    '''
    def __init__(
        # Shared modelling artifacts passed from the main Evaluation Agent.

        self,
        leaderboard: pd.DataFrame,
        best_model_metrics: Dict[str, Any],
        diagnostics: Dict[str, Any],
        modelling_summary: Dict[str, Any],
        best_model_predictions: Optional[pd.DataFrame] = None,
        best_model_feature_importance: Optional[pd.DataFrame] = None,
    ):
        self.leaderboard = leaderboard
        self.best_model_metrics = best_model_metrics
        self.diagnostics = diagnostics
        self.modelling_summary = modelling_summary
        self.best_model_predictions = _normalise_prediction_columns(best_model_predictions.copy()) if best_model_predictions is not None else None
        self.best_model_feature_importance = best_model_feature_importance

    def _get_primary_metric(self) -> Optional[str]:
        return self.modelling_summary.get("primary_metric")

    def _get_metric_column(self) -> Optional[str]:
        # Resolve the leaderboard column corresponding to the selected primary metric.

        primary_metric = self._get_primary_metric()
        if not primary_metric:
            return None
        metric_column = f"test_{primary_metric}"
        return metric_column if metric_column in self.leaderboard.columns else None

    def _sort_leaderboard(self) -> pd.DataFrame:
        if self.leaderboard.empty:
            return self.leaderboard.copy()
        metric_column = self._get_metric_column()
        if metric_column is None:
            return self.leaderboard.reset_index(drop=True)
        ascending = not is_higher_better(self._get_primary_metric())
        return self.leaderboard.sort_values(by=metric_column, ascending=ascending).reset_index(drop=True)

    def _metric_columns(self) -> List[str]:
        metric_columns = [c for c in self.leaderboard.columns if c.startswith("cv_") or c.startswith("test_")]
        return metric_columns

    def _comparison_table(self) -> List[Dict[str, Any]]:
        base_columns = [c for c in ["rank", "model_name"] if c in self.leaderboard.columns]
        metric_columns = self._metric_columns()
        selected_columns = base_columns + metric_columns
        if not selected_columns:
            return []
        return self.leaderboard[selected_columns].to_dict(orient="records")

    def _build_best_model_evaluation_common(self) -> Dict[str, Any]:
        result = {
            "model_name": self.best_model_metrics.get("model_name"),
            "metrics": self.best_model_metrics,
            "diagnostics": self.diagnostics,
        }

        if self.best_model_predictions is not None and not self.best_model_predictions.empty:
            result["prediction_output_available"] = True
            result["prediction_row_count"] = int(len(self.best_model_predictions))
            result["prediction_columns"] = list(self.best_model_predictions.columns)
        else:
            result["prediction_output_available"] = False
            result["prediction_row_count"] = 0
            result["prediction_columns"] = []

        if self.best_model_feature_importance is not None and not self.best_model_feature_importance.empty:
            result["feature_importance_available"] = True
            result["feature_importance_top_rows"] = self.best_model_feature_importance.head(10).to_dict(
                orient="records"
            )
        else:
            result["feature_importance_available"] = False
            result["feature_importance_top_rows"] = []

        return result

    def _build_selection_evidence_common(self) -> Dict[str, Any]:
        primary_metric = self.modelling_summary.get("primary_metric")
        best_model_name = self.modelling_summary.get("best_model_name")

        if self.leaderboard.empty or primary_metric is None or best_model_name is None:
            return {
                "selected_model": best_model_name,
                "selection_metric": primary_metric,
                "selection_metric_value": None,
                "selection_rank": None,
            }

        metric_column = f"test_{primary_metric}"
        selection_metric_value = None
        selection_rank = None

        if "model_name" in self.leaderboard.columns:
            matched = self.leaderboard[self.leaderboard["model_name"] == best_model_name]
            if not matched.empty:
                if metric_column in matched.columns:
                    selection_metric_value = safe_float(matched.iloc[0][metric_column])
                if "rank" in matched.columns:
                    rank_value = matched.iloc[0]["rank"]
                    selection_rank = None if pd.isna(rank_value) else int(rank_value)

        return {
            "selected_model": best_model_name,
            "selection_metric": primary_metric,
            "selection_metric_value": selection_metric_value,
            "selection_rank": selection_rank,
        }

    def _build_benchmark_overview_common(self) -> Dict[str, Any]:
        primary_metric = self.modelling_summary.get("primary_metric")
        best_model_name = self.modelling_summary.get("best_model_name")
        ranked = self._sort_leaderboard()

        top_ranked_model = None
        second_best_model = None
        top_model_margin_over_second = None
        metric_column = self._get_metric_column()

        if not ranked.empty and "model_name" in ranked.columns:
            top_ranked_model = str(ranked.iloc[0]["model_name"])
            if len(ranked) > 1:
                second_best_model = str(ranked.iloc[1]["model_name"])
                if metric_column and metric_column in ranked.columns:
                    top_value = ranked.iloc[0][metric_column]
                    second_value = ranked.iloc[1][metric_column]
                    if pd.notna(top_value) and pd.notna(second_value):
                        top_model_margin_over_second = float(abs(top_value - second_value))

        return {
            "candidate_model_count": int(len(self.leaderboard)),
            "primary_metric": primary_metric,
            "top_ranked_model": top_ranked_model,
            "second_best_model": second_best_model,
            "top_model_margin_over_second": top_model_margin_over_second,
            "selected_best_model": best_model_name,
            "comparison_table": self._comparison_table(),
        }

    @abstractmethod
    def build_benchmark_overview(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_best_model_selection_evidence(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_best_model_evaluation(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_error_analysis(self) -> Dict[str, Any]:
        raise NotImplementedError


class BinaryClassificationEvaluator(BaseTaskEvaluator):

    """
    Task-specific evaluator for binary classification problems.

    This evaluator supports:
    - benchmark comparison
    - best-model selection evidence
    - confusion-matrix-based error analysis
    - false positive / false negative inspection
    """

    def build_benchmark_overview(self) -> Dict[str, Any]:
        result = self._build_benchmark_overview_common()
        result["task_variant"] = "binary_classification"
        return result

    def build_best_model_selection_evidence(self) -> Dict[str, Any]:
        result = self._build_selection_evidence_common()
        result["task_variant"] = "binary_classification"
        return result

    def build_best_model_evaluation(self) -> Dict[str, Any]:
        result = self._build_best_model_evaluation_common()
        result["task_variant"] = "binary_classification"
        return result

    def build_error_analysis(self) -> Dict[str, Any]:
        if self.best_model_predictions is None or self.best_model_predictions.empty:
            return {"available": False, "reason": "best_model_predictions.csv not found or empty"}

        df = self.best_model_predictions.copy()
        required_cols = {"y_true", "y_pred"}
        if not required_cols.issubset(df.columns):
            return {
                "available": False,
                "reason": f"Required columns missing. Found columns: {list(df.columns)}",
            }

        y_true = df["y_true"]
        y_pred = df["y_pred"]

        unique_labels = sorted(set(pd.Series(y_true).dropna().tolist()) | set(pd.Series(y_pred).dropna().tolist()))
        if len(unique_labels) != 2:
            return {
                "available": False,
                "reason": f"Binary evaluator received {len(unique_labels)} labels: {unique_labels}",
            }

        negative_label, positive_label = unique_labels[0], unique_labels[1]
        tp = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
        tn = int(((y_true == negative_label) & (y_pred == negative_label)).sum())
        fp = int(((y_true == negative_label) & (y_pred == positive_label)).sum())
        fn = int(((y_true == positive_label) & (y_pred == negative_label)).sum())

        accuracy = float((y_true == y_pred).mean())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "available": True,
            "task_variant": "binary_classification",
            "labels": {
                "negative_label": negative_label,
                "positive_label": positive_label,
            },
            "confusion_matrix": {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            },
            "classification_metrics_from_predictions": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            "error_summary": {
                "false_positives": fp,
                "false_negatives": fn,
                "dominant_error_type": (
                    "false_positives" if fp > fn else "false_negatives" if fn > fp else "balanced"
                ),
            },
        }


class MulticlassClassificationEvaluator(BaseTaskEvaluator):

    """
    Task-specific evaluator for multiclass classification problems.

    This evaluator supports:
    - benchmark comparison across candidate models
    - per-class metric analysis
    - multiclass confusion matrix generation
    - identification of the most confused class pairs
    - identification of the weakest class by performance
    """

    def build_benchmark_overview(self) -> Dict[str, Any]:
        result = self._build_benchmark_overview_common()
        result["task_variant"] = "multiclass_classification"
        return result

    def build_best_model_selection_evidence(self) -> Dict[str, Any]:
        result = self._build_selection_evidence_common()
        result["task_variant"] = "multiclass_classification"
        return result

    def build_best_model_evaluation(self) -> Dict[str, Any]:
        result = self._build_best_model_evaluation_common()
        result["task_variant"] = "multiclass_classification"
        return result

    def build_error_analysis(self) -> Dict[str, Any]:
        if self.best_model_predictions is None or self.best_model_predictions.empty:
            return {"available": False, "reason": "best_model_predictions.csv not found or empty"}

        df = self.best_model_predictions.copy()
        required_cols = {"y_true", "y_pred"}
        if not required_cols.issubset(df.columns):
            return {
                "available": False,
                "reason": f"Required columns missing. Found columns: {list(df.columns)}",
            }

        y_true = pd.Series(df["y_true"])
        y_pred = pd.Series(df["y_pred"])
        labels = sorted(set(y_true.dropna().tolist()) | set(y_pred.dropna().tolist()))
        if len(labels) < 3:
            return {
                "available": False,
                "reason": f"Multiclass evaluator requires at least 3 labels. Found: {labels}",
            }

        confusion = pd.crosstab(
            pd.Categorical(y_true, categories=labels),
            pd.Categorical(y_pred, categories=labels),
            dropna=False,
        )
        confusion.index = [str(x) for x in labels]
        confusion.columns = [str(x) for x in labels]

        total = int(confusion.to_numpy().sum())
        correct = int(np.trace(confusion.to_numpy()))
        accuracy = float(correct / total) if total > 0 else 0.0

        per_class_metrics: List[Dict[str, Any]] = []
        confusion_pairs: List[Dict[str, Any]] = []

        tp_total = 0
        fp_total = 0
        fn_total = 0
        weighted_precision_sum = 0.0
        weighted_recall_sum = 0.0
        weighted_f1_sum = 0.0
        class_support_total = 0

        for label in labels:
            label_str = str(label)
            tp = int(confusion.loc[label_str, label_str])
            row_sum = int(confusion.loc[label_str].sum())
            col_sum = int(confusion[label_str].sum())
            fn = row_sum - tp
            fp = col_sum - tp
            support = row_sum

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            tp_total += tp
            fp_total += fp
            fn_total += fn
            weighted_precision_sum += precision * support
            weighted_recall_sum += recall * support
            weighted_f1_sum += f1 * support
            class_support_total += support

            per_class_metrics.append(
                {
                    "class": label,
                    "support": support,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

            for pred_label in labels:
                if pred_label == label:
                    continue
                count = int(confusion.loc[label_str, str(pred_label)])
                if count > 0:
                    confusion_pairs.append(
                        {
                            "true_class": label,
                            "predicted_class": pred_label,
                            "count": count,
                        }
                    )

        macro_precision = float(np.mean([m["precision"] for m in per_class_metrics])) if per_class_metrics else 0.0
        macro_recall = float(np.mean([m["recall"] for m in per_class_metrics])) if per_class_metrics else 0.0
        macro_f1 = float(np.mean([m["f1"] for m in per_class_metrics])) if per_class_metrics else 0.0
        weighted_precision = float(weighted_precision_sum / class_support_total) if class_support_total > 0 else 0.0
        weighted_recall = float(weighted_recall_sum / class_support_total) if class_support_total > 0 else 0.0
        weighted_f1 = float(weighted_f1_sum / class_support_total) if class_support_total > 0 else 0.0
        micro_precision = float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
        micro_recall = float(tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = (
            float(2 * micro_precision * micro_recall / (micro_precision + micro_recall))
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        worst_class = min(per_class_metrics, key=lambda x: x["f1"]) if per_class_metrics else None
        most_confused_pairs = sorted(confusion_pairs, key=lambda x: x["count"], reverse=True)[:5]

        return {
            "available": True,
            "task_variant": "multiclass_classification",
            "labels": labels,
            "confusion_matrix": confusion.to_dict(orient="index"),
            "overall_metrics_from_predictions": {
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1": weighted_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
            },
            "per_class_metrics": per_class_metrics,
            "error_summary": {
                "most_confused_class_pairs": most_confused_pairs,
                "worst_class_by_f1": worst_class["class"] if worst_class else None,
                "worst_class_f1": worst_class["f1"] if worst_class else None,
            },
        }


class RegressionEvaluator(BaseTaskEvaluator):

    """
    Task-specific evaluator for regression problems.

    This evaluator supports:
    - benchmark comparison across candidate models
    - best-model selection evidence
    - residual-based error analysis
    - overestimation / underestimation inspection
    - worst-prediction analysis
    """

    def build_benchmark_overview(self) -> Dict[str, Any]:
        result = self._build_benchmark_overview_common()
        result["task_variant"] = "regression"
        return result

    def build_best_model_selection_evidence(self) -> Dict[str, Any]:
        result = self._build_selection_evidence_common()
        result["task_variant"] = "regression"
        return result

    def build_best_model_evaluation(self) -> Dict[str, Any]:
        result = self._build_best_model_evaluation_common()
        result["task_variant"] = "regression"
        return result

    def build_error_analysis(self) -> Dict[str, Any]:
        if self.best_model_predictions is None or self.best_model_predictions.empty:
            return {"available": False, "reason": "best_model_predictions.csv not found or empty"}

        df = self.best_model_predictions.copy()
        required_cols = {"y_true", "y_pred"}
        if not required_cols.issubset(df.columns):
            return {
                "available": False,
                "reason": f"Required columns missing. Found columns: {list(df.columns)}",
            }

        y_true = pd.to_numeric(df["y_true"], errors="coerce")
        y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
        valid_mask = ~(y_true.isna() | y_pred.isna())
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            return {"available": False, "reason": "No valid numeric rows found for regression analysis"}

        residual = y_true - y_pred
        abs_error = residual.abs()
        squared_error = residual.pow(2)
        mae = float(abs_error.mean())
        mse = float(squared_error.mean())
        rmse = float(np.sqrt(mse))

        ss_res = float(squared_error.sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

        enriched = pd.DataFrame({
            "y_true": y_true.values,
            "y_pred": y_pred.values,
            "residual": residual.values,
            "absolute_error": abs_error.values,
            "squared_error": squared_error.values,
        })
        if "sample_id" in df.columns:
            enriched["sample_id"] = df.loc[valid_mask, "sample_id"].values
        enriched = enriched.sort_values(by="absolute_error", ascending=False)

        overestimation_count = int((residual < 0).sum())
        underestimation_count = int((residual > 0).sum())
        near_zero_count = int((residual == 0).sum())

        return {
            "available": True,
            "task_variant": "regression",
            "regression_metrics_from_predictions": {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "mean_residual": float(residual.mean()),
                "median_absolute_error": float(abs_error.median()),
                "max_absolute_error": float(abs_error.max()),
            },
            "error_summary": {
                "overestimation_count": overestimation_count,
                "underestimation_count": underestimation_count,
                "exact_prediction_count": near_zero_count,
                "dominant_error_direction": (
                    "overestimation"
                    if overestimation_count > underestimation_count
                    else "underestimation"
                    if underestimation_count > overestimation_count
                    else "balanced"
                ),
            },
            "worst_predictions_top5": enriched.head(5).to_dict(orient="records"),
        }


class GenericClassificationEvaluator(BaseTaskEvaluator):

    """
    Fallback evaluator for classification tasks when the subtype
    cannot be confidently identified as binary or multiclass.

    This evaluator provides safe generic classification analysis
    without assuming a specific class structure.
    """
    
    def build_benchmark_overview(self) -> Dict[str, Any]:
        result = self._build_benchmark_overview_common()
        result["task_variant"] = "generic_classification"
        return result

    def build_best_model_selection_evidence(self) -> Dict[str, Any]:
        result = self._build_selection_evidence_common()
        result["task_variant"] = "generic_classification"
        return result

    def build_best_model_evaluation(self) -> Dict[str, Any]:
        result = self._build_best_model_evaluation_common()
        result["task_variant"] = "generic_classification"
        return result

    def build_error_analysis(self) -> Dict[str, Any]:
        if self.best_model_predictions is None or self.best_model_predictions.empty:
            return {"available": False, "reason": "best_model_predictions.csv not found or empty"}

        df = self.best_model_predictions.copy()
        required_cols = {"y_true", "y_pred"}
        if not required_cols.issubset(df.columns):
            return {
                "available": False,
                "reason": f"Required columns missing. Found columns: {list(df.columns)}",
            }

        y_true = pd.Series(df["y_true"])
        y_pred = pd.Series(df["y_pred"])
        labels = sorted(set(y_true.dropna().tolist()) | set(y_pred.dropna().tolist()))
        accuracy = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
        return {
            "available": True,
            "task_variant": "generic_classification",
            "label_count": len(labels),
            "labels": labels,
            "accuracy": accuracy,
            "note": "Fell back to generic classification evaluator because task variant could not be resolved confidently.",
        }


class EvaluationAgent:
    """
    Evaluation Agent

    Responsibility:
    - Consume standardised modelling artifacts
    - Benchmark candidate models using exported modelling results
    - Compare candidate models under a common benchmark
    - Validate model-selection consistency across exported artifacts
    - Present technical evidence for best-model selection
    - Perform task-aware prediction-level error analysis for the selected best model
    - Save structured evaluation outputs
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.leaderboard_: pd.DataFrame = pd.DataFrame()
        self.best_model_metrics_: Dict[str, Any] = {}
        self.diagnostics_: Dict[str, Any] = {}
        self.modelling_summary_: Dict[str, Any] = {}
        self.modelling_metadata_: Dict[str, Any] = {}
        self.best_model_predictions_: Optional[pd.DataFrame] = None
        self.best_model_feature_importance_: Optional[pd.DataFrame] = None
        self.summary_: Dict[str, Any] = {}
        self.task_variant_: Optional[str] = None

    def run(self) -> Dict[str, Any]:
        modelling_dir = Path(self.config.modelling_output_dir)

        self.leaderboard_ = self._load_required_csv(modelling_dir / "leaderboard.csv")
        self.best_model_metrics_ = self._load_required_json(modelling_dir / "best_model_metrics.json")
        self.diagnostics_ = self._load_required_json(modelling_dir / "diagnostics.json")
        self.modelling_summary_ = self._load_required_json(modelling_dir / "modelling_summary.json")

        self.modelling_metadata_ = self._load_optional_json(modelling_dir / "modelling_metadata.json") or {}
        self.best_model_predictions_ = _normalise_prediction_columns(
            self._load_optional_csv(modelling_dir / "best_model_predictions.csv")
        ) if (modelling_dir / "best_model_predictions.csv").exists() else None
        self.best_model_feature_importance_ = self._load_optional_csv(
            modelling_dir / "best_model_feature_importance.csv"
        )

        evaluator = self._resolve_evaluator()

        summary = {
            "problem_type": self.modelling_summary_.get("problem_type"),
            "task_variant": self.task_variant_,
            "primary_metric": self.modelling_summary_.get("primary_metric"),
            "best_model_name": self.modelling_summary_.get("best_model_name"),
            "benchmark_overview": evaluator.build_benchmark_overview(),
            "consistency_checks": self._validate_modelling_outputs(),
            "best_model_selection_evidence": evaluator.build_best_model_selection_evidence(),
            "best_model_evaluation": evaluator.build_best_model_evaluation(),
            "error_analysis": evaluator.build_error_analysis(),
            "limitations": self.modelling_summary_.get("limitations", []),
        }

        self.summary_ = summary

        if self.config.save_artifacts:
            self._save_results(summary)

        return summary

    def _load_required_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_optional_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_required_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Required CSV file not found: {path}")
        return pd.read_csv(path)

    def _load_optional_csv(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        return pd.read_csv(path)

    def _infer_task_variant(self) -> str:
        explicit_task_variant = self.modelling_summary_.get("task_variant")
        if explicit_task_variant:
            return str(explicit_task_variant)

        problem_type = self.modelling_summary_.get("problem_type")
        primary_metric = self.modelling_summary_.get("primary_metric")

        if problem_type == "regression":
            return "regression"

        if problem_type == "classification":
            if primary_metric in {"macro_f1", "weighted_f1", "macro_precision", "macro_recall"}:
                return "multiclass_classification"
            if self.best_model_predictions_ is not None and not self.best_model_predictions_.empty:
                if {"y_true", "y_pred"}.issubset(self.best_model_predictions_.columns):
                    y_true = pd.Series(self.best_model_predictions_["y_true"])
                    y_pred = pd.Series(self.best_model_predictions_["y_pred"])
                    labels = sorted(set(y_true.dropna().tolist()) | set(y_pred.dropna().tolist()))
                    if len(labels) <= 2:
                        return "binary_classification"
                    return "multiclass_classification"
            return "generic_classification"

        if primary_metric in REGRESSION_METRICS:
            return "regression"
        if primary_metric in CLASSIFICATION_METRICS:
            return "generic_classification"
        return "generic_classification"

    def _resolve_evaluator(self) -> BaseTaskEvaluator:
        self.task_variant_ = self._infer_task_variant()
        registry = {
            "binary_classification": BinaryClassificationEvaluator,
            "multiclass_classification": MulticlassClassificationEvaluator,
            "regression": RegressionEvaluator,
            "generic_classification": GenericClassificationEvaluator,
        }
        evaluator_cls = registry.get(self.task_variant_, GenericClassificationEvaluator)
        return evaluator_cls(
            leaderboard=self.leaderboard_,
            best_model_metrics=self.best_model_metrics_,
            diagnostics=self.diagnostics_,
            modelling_summary=self.modelling_summary_,
            best_model_predictions=self.best_model_predictions_,
            best_model_feature_importance=self.best_model_feature_importance_,
        )

    def _validate_modelling_outputs(self) -> Dict[str, Any]:
        checks: Dict[str, Any] = {}

        primary_metric = self.modelling_summary_.get("primary_metric")
        best_model_name = self.modelling_summary_.get("best_model_name")
        problem_type = self.modelling_summary_.get("problem_type")

        checks["problem_type_present"] = problem_type is not None
        checks["task_variant_resolved"] = self.task_variant_
        checks["primary_metric_present"] = primary_metric is not None
        checks["best_model_name_present"] = best_model_name is not None
        checks["leaderboard_available"] = not self.leaderboard_.empty

        if self.leaderboard_.empty:
            checks["leaderboard_top_model"] = None
            checks["best_model_matches_leaderboard_rank1"] = False
            checks["primary_metric_column_present"] = False
            return checks

        if "model_name" in self.leaderboard_.columns:
            leaderboard_top_model = str(self.leaderboard_.iloc[0]["model_name"])
        else:
            leaderboard_top_model = None
        checks["leaderboard_top_model"] = leaderboard_top_model
        checks["best_model_matches_leaderboard_rank1"] = best_model_name == leaderboard_top_model

        metric_column = f"test_{primary_metric}" if primary_metric else None
        if metric_column and metric_column in self.leaderboard_.columns:
            checks["primary_metric_column_present"] = True
            ascending = not is_higher_better(primary_metric)
            ranked = self.leaderboard_.sort_values(by=metric_column, ascending=ascending).reset_index(drop=True)
            metric_top_model = str(ranked.iloc[0]["model_name"])
            checks["top_model_by_primary_metric"] = metric_top_model
            checks["best_model_matches_primary_metric_ranking"] = best_model_name == metric_top_model
            checks["metric_sort_direction"] = "descending" if is_higher_better(primary_metric) else "ascending"
        else:
            checks["primary_metric_column_present"] = False
            checks["top_model_by_primary_metric"] = None
            checks["best_model_matches_primary_metric_ranking"] = False
            checks["metric_sort_direction"] = None

        checks["best_model_metrics_match_summary"] = self.best_model_metrics_.get("model_name") == best_model_name
        checks["diagnostics_match_summary"] = self.diagnostics_.get("best_model_name") == best_model_name

        if self.best_model_predictions_ is not None and not self.best_model_predictions_.empty:
            checks["prediction_file_available"] = True
            checks["prediction_required_columns_present"] = {"y_true", "y_pred"}.issubset(
                self.best_model_predictions_.columns
            )
        else:
            checks["prediction_file_available"] = False
            checks["prediction_required_columns_present"] = False

        return checks

    def _save_results(self, summary: Dict[str, Any]) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if not self.leaderboard_.empty:
            self.leaderboard_.to_csv(output_dir / "evaluation_comparison_table.csv", index=False)

    def get_minimal_summary(self) -> Dict[str, Any]:
        if not self.summary_:
            return {"message": "No evaluation summary available. Run the agent first."}

        benchmark = self.summary_.get("benchmark_overview", {})
        evidence = self.summary_.get("best_model_selection_evidence", {})

        return {
            "task_variant": self.summary_.get("task_variant"),
            "primary_metric": self.summary_.get("primary_metric"),
            "best_model_name": self.summary_.get("best_model_name"),
            "candidate_model_count": benchmark.get("candidate_model_count"),
            "top_ranked_model": benchmark.get("top_ranked_model"),
            "selection_metric_value": evidence.get("selection_metric_value"),
        }
