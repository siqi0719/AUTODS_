from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

@dataclass
class EvaluationConfig:
    modelling_output_dir: str = "model_outputs"
    output_dir: str = "evaluation_outputs"
    save_artifacts: bool = True

class EvaluationAgent:
    """
    Evaluation Agent

    Responsibility:
    - Consume standardised modelling artifacts
    - Benchmark candidate models using exported modelling results
    - Summarise model comparison
    - Present technical evidence for best model selection
    - Save structured evaluation outputs
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.leaderboard_: pd.DataFrame = pd.DataFrame()
        self.best_model_metrics_: Dict[str, Any] = {}
        self.diagnostics_: Dict[str, Any] = {}
        self.modelling_summary_: Dict[str, Any] = {}
        self.modelling_metadata_: Dict[str, Any] = {}
        self.best_model_predictions_: pd.DataFrame = pd.DataFrame()
        self.best_model_feature_importance_: Optional[pd.DataFrame] = None
        self.summary_: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """
        Main entry point for the Evaluation Agent.
        Reads modelling artifacts, validates consistency and returns
        structured evaluation outputs.
        """
        modelling_dir = Path(self.config.modelling_output_dir)

        self.leaderboard_ = self._load_required_csv(modelling_dir / "leaderboard.csv")
        self.best_model_metrics_ = self._load_required_json(modelling_dir / "best_model_metrics.json")
        self.diagnostics_ = self._load_required_json(modelling_dir / "diagnostics.json")
        self.modelling_summary_ = self._load_required_json(modelling_dir / "modelling_summary.json")

        self.modelling_metadata_ = self._load_optional_json(modelling_dir / "modelling_metadata.json") or {}
        self.best_model_predictions_ = self._load_optional_csv(modelling_dir / "best_model_predictions.csv")
        self.best_model_feature_importance_ = self._load_optional_csv(
            modelling_dir / "best_model_feature_importance.csv"
        )

        summary = {
            "problem_type": self.modelling_summary_.get("problem_type"),
            "primary_metric": self.modelling_summary_.get("primary_metric"),
            "best_model_name": self.modelling_summary_.get("best_model_name"),
            "benchmark_overview": self._build_benchmark_overview(),
            "best_model_selection_evidence": self._build_best_model_selection_evidence(),
            "best_model_evaluation": self._build_best_model_evaluation(),
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

    def _validate_modelling_outputs(self) -> Dict[str, Any]:
        """
        Validate that the modelling outputs are internally consistent.
        """
        checks: Dict[str, Any] = {}

        primary_metric = self.modelling_summary_.get("primary_metric")
        best_model_name = self.modelling_summary_.get("best_model_name")

        checks["primary_metric_present"] = primary_metric is not None
        checks["best_model_name_present"] = best_model_name is not None
        checks["leaderboard_available"] = not self.leaderboard_.empty

        if self.leaderboard_.empty:
            checks["leaderboard_top_model"] = None
            checks["best_model_matches_leaderboard_rank1"] = False
            checks["primary_metric_column_present"] = False
            return checks

        leaderboard_top_model = str(self.leaderboard_.iloc[0]["model_name"])
        checks["leaderboard_top_model"] = leaderboard_top_model
        checks["best_model_matches_leaderboard_rank1"] = best_model_name == leaderboard_top_model

        metric_column = f"test_{primary_metric}" if primary_metric else None
        if metric_column and metric_column in self.leaderboard_.columns:
            checks["primary_metric_column_present"] = True

            ranked = self.leaderboard_.sort_values(by=metric_column, ascending=False).reset_index(drop=True)
            metric_top_model = str(ranked.iloc[0]["model_name"])
            checks["top_model_by_primary_metric"] = metric_top_model
            checks["best_model_matches_primary_metric_ranking"] = best_model_name == metric_top_model
        else:
            checks["primary_metric_column_present"] = False
            checks["top_model_by_primary_metric"] = None
            checks["best_model_matches_primary_metric_ranking"] = False

        checks["best_model_metrics_match_summary"] = (
            self.best_model_metrics_.get("model_name") == best_model_name
        )
        checks["diagnostics_match_summary"] = (
            self.diagnostics_.get("best_model_name") == best_model_name
        )

        return checks
    
    def _build_benchmark_overview(self) -> Dict[str, Any]:
        """
        Build a benchmark overview from leaderboard.csv.
        """
        primary_metric = self.modelling_summary_.get("primary_metric")
        best_model_name = self.modelling_summary_.get("best_model_name")

        columns_to_keep = [
            "rank",
            "model_name",
            "cv_accuracy",
            "cv_precision",
            "cv_recall",
            "cv_f1",
            "cv_roc_auc",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
        ]

        available_columns = [col for col in columns_to_keep if col in self.leaderboard_.columns]
        benchmark_table = self.leaderboard_[available_columns].to_dict(orient="records")

        top_ranked_model = None
        if not self.leaderboard_.empty and "model_name" in self.leaderboard_.columns:
            top_ranked_model = str(self.leaderboard_.iloc[0]["model_name"])


        return {
            "candidate_model_count": int(len(self.leaderboard_)),
            "primary_metric": primary_metric,
            "top_ranked_model": top_ranked_model,
            "selected_best_model": best_model_name,
            "comparison_table": benchmark_table,
        }
       
    def _build_best_model_evaluation(self) -> Dict[str, Any]:
        """
        Build a technical evaluation summary for the selected best model.
        """
        result = {
            "model_name": self.best_model_metrics_.get("model_name"),
            "metrics": self.best_model_metrics_,
            "diagnostics": self.diagnostics_,
        }

        if self.best_model_predictions_ is not None and not self.best_model_predictions_.empty:
            result["prediction_output_available"] = True
            result["prediction_row_count"] = int(len(self.best_model_predictions_))
            result["prediction_columns"] = list(self.best_model_predictions_.columns)
        else:
            result["prediction_output_available"] = False
            result["prediction_row_count"] = 0
            result["prediction_columns"] = []

        if self.best_model_feature_importance_ is not None and not self.best_model_feature_importance_.empty:
            result["feature_importance_available"] = True
            result["feature_importance_top_rows"] = self.best_model_feature_importance_.head(10).to_dict(
                orient="records"
            )
        else:
            result["feature_importance_available"] = False
            result["feature_importance_top_rows"] = []

        return result
    
    def _build_best_model_selection_evidence(self) -> Dict[str, Any]:
        """
        Build technical evidence showing why the selected model is considered the best model.
        """
        primary_metric = self.modelling_summary_.get("primary_metric")
        best_model_name = self.modelling_summary_.get("best_model_name")

        if self.leaderboard_.empty or primary_metric is None or best_model_name is None:
            return {
                "selected_model": best_model_name,
                "selection_metric": primary_metric,
                "selection_metric_value": None,
                "selection_rank": None,
            }
  
        metric_column = f"test_{primary_metric}"
        selection_metric_value = None
        selection_rank = None

        if "model_name" in self.leaderboard_.columns:
            matched = self.leaderboard_[self.leaderboard_["model_name"] == best_model_name]

            if not matched.empty:
                if metric_column in matched.columns:
                    value = matched.iloc[0][metric_column]
                    selection_metric_value = None if pd.isna(value) else float(value)

                if "rank" in matched.columns:
                    rank_value = matched.iloc[0]["rank"]
                    selection_rank = None if pd.isna(rank_value) else int(rank_value)

        return {
            "selected_model": best_model_name,
            "selection_metric": primary_metric,
            "selection_metric_value": selection_metric_value,
            "selection_rank": selection_rank,
        }

    def _save_results(self, summary: Dict[str, Any]) -> None:
        """
        Save structured evaluation outputs.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if not self.leaderboard_.empty:
            self.leaderboard_.to_csv(output_dir / "evaluation_comparison_table.csv", index=False)

    def get_minimal_summary(self) -> Dict[str, Any]:
        """
        Return a minimal technical summary strictly within the evaluation boundary.
        """
        if not self.summary_:
            raise ValueError("Evaluation has not been run yet.")
        
        return {
            "primary_metric": self.summary_.get("primary_metric"),
            "best_model_name": self.summary_.get("best_model_name"),
            "candidate_model_count": self.summary_.get("benchmark_overview", {}).get("candidate_model_count"),
            "top_ranked_model": self.summary_.get("benchmark_overview", {}).get("top_ranked_model"),
            "best_model_selection_evidence": self.summary_.get("best_model_selection_evidence"),
        }