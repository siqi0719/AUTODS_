from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


class VerificationError(Exception):
    """Raised when pipeline verification fails and execution should stop."""
    pass


@dataclass
class ValidationIssue:
    level: str   # info, warning, error
    step: str
    message: str


class VerificationAgent:
    """
    File-based Verification Agent for AUTODS.

    Responsibilities:
    1. Validate each step's output artifacts
    2. Validate handoff consistency between steps
    3. Stop pipeline if critical errors exist
    4. Optionally call an LLM to generate a readable verification report
    """

    def __init__(
        self,
        pipeline_root: str | Path,
        llm_report_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        fail_on_error: bool = True,
    ) -> None:
        self.pipeline_root = Path(pipeline_root)
        self.llm_report_fn = llm_report_fn
        self.fail_on_error = fail_on_error
        self.issues: List[ValidationIssue] = []

    # -----------------------------
    # basic helpers
    # -----------------------------
    def reset(self) -> None:
        self.issues = []

    def _add_issue(self, level: str, step: str, message: str) -> None:
        self.issues.append(ValidationIssue(level=level, step=step, message=message))

    def _step_path(self, step_folder: str) -> Path:
        return self.pipeline_root / step_folder

    def _require_file(self, step: str, path: Path) -> bool:
        if not path.exists():
            self._add_issue("error", step, f"Missing required file: {path.name}")
            return False
        if path.is_dir():
            self._add_issue("error", step, f"Expected file but found directory: {path.name}")
            return False
        self._add_issue("info", step, f"Found file: {path.name}")
        return True

    def _load_json(self, step: str, path: Path) -> Optional[Dict[str, Any]]:
        if not self._require_file(step, path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._add_issue("info", step, f"JSON parsed successfully: {path.name}")
            return data
        except Exception as e:
            self._add_issue("error", step, f"Invalid JSON in {path.name}: {e}")
            return None

    def _load_csv(self, step: str, path: Path) -> Optional[pd.DataFrame]:
        if not self._require_file(step, path):
            return None
        try:
            df = pd.read_csv(path)
            self._add_issue("info", step, f"CSV loaded successfully: {path.name}, shape={df.shape}")
            if df.empty:
                self._add_issue("error", step, f"CSV is empty: {path.name}")
            return df
        except Exception as e:
            self._add_issue("error", step, f"Failed to read CSV {path.name}: {e}")
            return None

    def _check_non_empty_text_file(self, step: str, path: Path) -> None:
        if not self._require_file(step, path):
            return
        try:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                self._add_issue("error", step, f"Text file is empty: {path.name}")
            else:
                self._add_issue("info", step, f"Text file is non-empty: {path.name}")
        except Exception as e:
            self._add_issue("error", step, f"Cannot read text file {path.name}: {e}")

    def _require_joblib(self, step: str, path: Path) -> None:
        if self._require_file(step, path):
            self._add_issue("info", step, f"Artifact exists: {path.name}")

    def _check_json_keys(
        self,
        step: str,
        data: Optional[Dict[str, Any]],
        required_keys: List[str],
        file_name: str
    ) -> None:
        if data is None:
            return
        for key in required_keys:
            if key not in data:
                self._add_issue("warning", step, f"{file_name} missing suggested key: '{key}'")

    # -----------------------------
    # step validations
    # -----------------------------
    def validate_understanding(self) -> None:
        step = "01_understanding"
        root = self._step_path(step)

        files = [
            "data_profile.json",
            "data_quality_report.json",
            "data_understanding_metadata.json",
            "data_understanding_summary.json",
            "target_analysis.json",
        ]

        loaded = {}
        for name in files:
            loaded[name] = self._load_json(step, root / name)

        self._check_json_keys(step, loaded["data_understanding_metadata.json"], ["dataset_name"], "data_understanding_metadata.json")
        self._check_json_keys(step, loaded["data_understanding_summary.json"], ["summary"], "data_understanding_summary.json")
        self._check_json_keys(step, loaded["target_analysis.json"], ["target_column"], "target_analysis.json")

    def validate_cleaning(self) -> None:
        step = "02_cleaning"
        root = self._step_path(step)

        cleaned_df = self._load_csv(step, root / "cleaned_data.csv")
        cleaning_report = self._load_json(step, root / "cleaning_report.json")

        self._check_json_keys(step, cleaning_report, ["cleaning_actions"], "cleaning_report.json")

        if cleaned_df is not None and cleaned_df.shape[1] == 0:
            self._add_issue("error", step, "cleaned_data.csv has zero columns.")

    def validate_feature_engineering(self) -> None:
        step = "03_feature_engineering"
        root = self._step_path(step)

        feature_metadata = self._load_json(step, root / "feature_metadata.json")
        feature_summary = self._load_json(step, root / "feature_summary.json")
        llm_plan = self._load_json(step, root / "llm_plan.json")

        self._require_joblib(step, root / "preprocessor.joblib")

        X_engineered = self._load_csv(step, root / "X_engineered.csv")
        X_train = self._load_csv(step, root / "X_train.csv")
        X_test = self._load_csv(step, root / "X_test.csv")
        y = self._load_csv(step, root / "y.csv")
        y_train = self._load_csv(step, root / "y_train.csv")
        y_test = self._load_csv(step, root / "y_test.csv")

        self._check_json_keys(step, feature_metadata, ["feature_count"], "feature_metadata.json")
        self._check_json_keys(step, feature_summary, ["summary"], "feature_summary.json")
        self._check_json_keys(step, llm_plan, ["plan"], "llm_plan.json")

        if X_train is not None and y_train is not None and len(X_train) != len(y_train):
            self._add_issue("error", step, f"X_train rows ({len(X_train)}) != y_train rows ({len(y_train)})")

        if X_test is not None and y_test is not None and len(X_test) != len(y_test):
            self._add_issue("error", step, f"X_test rows ({len(X_test)}) != y_test rows ({len(y_test)})")

        if X_engineered is not None and X_engineered.shape[0] == 0:
            self._add_issue("error", step, "X_engineered.csv has zero rows.")

        if y is not None and y.shape[1] == 0:
            self._add_issue("error", step, "y.csv has zero columns.")

    def validate_modelling(self) -> None:
        step = "04_modelling"
        root = self._step_path(step)

        self._require_joblib(step, root / "best_model.joblib")

        feature_importance = self._load_csv(step, root / "best_model_feature_importance.csv")
        metrics = self._load_json(step, root / "best_model_metrics.json")
        predictions = self._load_csv(step, root / "best_model_predictions.csv")
        diagnostics = self._load_json(step, root / "diagnostics.json")
        leaderboard = self._load_csv(step, root / "leaderboard.csv")
        llm_plan = self._load_json(step, root / "llm_plan.json")
        metadata = self._load_json(step, root / "modelling_metadata.json")
        summary = self._load_json(step, root / "modelling_summary.json")

        self._check_json_keys(step, metrics, ["best_model"], "best_model_metrics.json")
        self._check_json_keys(step, diagnostics, ["status"], "diagnostics.json")
        self._check_json_keys(step, metadata, ["task_type"], "modelling_metadata.json")
        self._check_json_keys(step, summary, ["summary"], "modelling_summary.json")
        self._check_json_keys(step, llm_plan, ["plan"], "llm_plan.json")

        if predictions is not None and predictions.empty:
            self._add_issue("error", step, "best_model_predictions.csv is empty.")

        if leaderboard is not None and leaderboard.empty:
            self._add_issue("error", step, "leaderboard.csv is empty.")

        if feature_importance is not None and feature_importance.empty:
            self._add_issue("warning", step, "best_model_feature_importance.csv is empty.")

    def validate_evaluation(self) -> None:
        step = "05_evaluation"
        root = self._step_path(step)

        comparison = self._load_csv(step, root / "evaluation_comparison_table.csv")
        summary = self._load_json(step, root / "evaluation_summary.json")

        self._check_json_keys(step, summary, ["summary"], "evaluation_summary.json")

        if comparison is not None and comparison.empty:
            self._add_issue("error", step, "evaluation_comparison_table.csv is empty.")

    def validate_reports(self) -> None:
        step = "06_reports"
        root = self._step_path(step)

        report_input = self._load_json(step, root / "pipeline_report_input.json")
        report_json = self._load_json(step, root / "report.json")
        self._check_non_empty_text_file(step, root / "report.md")

        self._check_json_keys(step, report_input, ["pipeline"], "pipeline_report_input.json")
        self._check_json_keys(step, report_json, ["report"], "report.json")

    # -----------------------------
    # handoff validations
    # -----------------------------
    def validate_handoffs(self) -> None:
        step = "handoff"

        cleaned = self._load_csv(step, self._step_path("02_cleaning") / "cleaned_data.csv")
        X_engineered = self._load_csv(step, self._step_path("03_feature_engineering") / "X_engineered.csv")

        if cleaned is not None and X_engineered is not None:
            if len(cleaned) != len(X_engineered):
                self._add_issue(
                    "warning",
                    step,
                    f"Row count changed from cleaned_data ({len(cleaned)}) to X_engineered ({len(X_engineered)}). Check whether this is expected."
                )

        X_test = self._load_csv(step, self._step_path("03_feature_engineering") / "X_test.csv")
        y_test = self._load_csv(step, self._step_path("03_feature_engineering") / "y_test.csv")
        preds = self._load_csv(step, self._step_path("04_modelling") / "best_model_predictions.csv")

        if X_test is not None and y_test is not None and len(X_test) != len(y_test):
            self._add_issue("error", step, f"X_test rows ({len(X_test)}) != y_test rows ({len(y_test)})")

        if preds is not None and y_test is not None and len(preds) != len(y_test):
            self._add_issue("error", step, f"Predictions rows ({len(preds)}) != y_test rows ({len(y_test)})")

        metrics = self._load_json(step, self._step_path("04_modelling") / "best_model_metrics.json")
        evaluation_summary = self._load_json(step, self._step_path("05_evaluation") / "evaluation_summary.json")
        report_input = self._load_json(step, self._step_path("06_reports") / "pipeline_report_input.json")

        if metrics is None:
            self._add_issue("error", step, "Evaluation step cannot proceed because modelling metrics are missing.")

        if evaluation_summary is None:
            self._add_issue("warning", step, "evaluation_summary.json missing or invalid.")

        if report_input is None:
            self._add_issue("error", step, "Report generation input missing.")

    # -----------------------------
    # public methods
    # -----------------------------
    def validate_all(self) -> Dict[str, Any]:
        self.reset()
        self.validate_understanding()
        self.validate_cleaning()
        self.validate_feature_engineering()
        self.validate_modelling()
        self.validate_evaluation()
        self.validate_reports()
        self.validate_handoffs()

        report = self.build_report()

        if self.fail_on_error and report["summary"]["error_count"] > 0:
            raise VerificationError("Verification failed. Critical pipeline errors detected.")

        return report

    def build_report(self) -> Dict[str, Any]:
        return {
            "is_valid": not any(i.level == "error" for i in self.issues),
            "issues": [asdict(i) for i in self.issues],
            "summary": {
                "error_count": sum(1 for i in self.issues if i.level == "error"),
                "warning_count": sum(1 for i in self.issues if i.level == "warning"),
                "info_count": sum(1 for i in self.issues if i.level == "info"),
            },
        }

    def verify_or_raise(self, validate_fn, step_name: str) -> Dict[str, Any]:
        self.reset()
        validate_fn()
        report = self.build_report()
        if self.fail_on_error and report["summary"]["error_count"] > 0:
            raise VerificationError(f"{step_name} failed verification.")
        return report

    def generate_llm_verification_report(self, report: Dict[str, Any]) -> str:
        if self.llm_report_fn is None:
            return self.generate_plain_text_report(report)

        try:
            return self.llm_report_fn(report)
        except Exception as e:
            self._add_issue("warning", "llm_report", f"LLM report generation failed: {e}")
            return self.generate_plain_text_report(self.build_report())

    def generate_plain_text_report(self, report: Dict[str, Any]) -> str:
        lines = []
        lines.append("=== AUTODS Verification Report ===")
        lines.append(f"Pipeline Valid: {report['is_valid']}")
        lines.append(f"Errors: {report['summary']['error_count']}")
        lines.append(f"Warnings: {report['summary']['warning_count']}")
        lines.append(f"Infos: {report['summary']['info_count']}")
        lines.append("")
        for issue in report["issues"]:
            lines.append(f"[{issue['level'].upper()}] {issue['step']} - {issue['message']}")
        return "\n".join(lines)