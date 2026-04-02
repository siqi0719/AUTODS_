#!/usr/bin/env python3
"""
Multi-agent report generation system.

This module receives structured JSON from upstream teammates or agents and
produces technical and business-facing Markdown reports.
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from dotenv import load_dotenv

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
    LANGCHAIN_IMPORT_ERROR = None
except Exception as exc:
    StrOutputParser = None
    ChatPromptTemplate = None
    ChatOpenAI = None
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_IMPORT_ERROR = exc


load_dotenv()


ReportInput = Union[str, os.PathLike[str], Mapping[str, Any]]


@dataclass
class ReportGeneratorConfig:
    output_dir: str = "reports"
    llm_model: str = "gpt-4o-mini"
    technical_temperature: float = 0.3
    business_temperature: float = 0.5
    report_language: str = "en"
    require_llm: bool = False
    force_template_mode: bool = False
    business_include_technical_context: bool = False


class MultiAgentReportGenerator:
    """Generates technical and business reports from structured JSON inputs."""

    AGENT_NAME = "AUTODS_REPORT_GENERATOR"
    NORMALIZED_REPORT_SECTIONS = [
        "meta",
        "data_understanding",
        "data_cleaning",
        "feature_engineering",
        "modeling",
        "evaluation",
        "business_context",
    ]
    SOURCE_NEW_SCHEMA_SECTIONS = [
        "project_info",
        "dataset_summary",
        "pipeline_trace",
        "model_results",
        "interpretability",
        "risk_scoring",
        "business_constraints",
    ]
    LANGUAGE_ALIASES = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "cn": "Chinese",
        "en": "English",
        "en-us": "English",
        "en-gb": "English",
    }

    def __init__(
        self,
        config: Optional[ReportGeneratorConfig] = None,
        openai_api_key: Optional[str] = None,
    ):
        default_config = ReportGeneratorConfig(
            output_dir="reports",
            llm_model=os.getenv("OPENAI_MODEL", ReportGeneratorConfig.llm_model),
            report_language=os.getenv(
                "REPORT_LANGUAGE", ReportGeneratorConfig.report_language
            ),
        )
        self.config = config or default_config

        if self.config.require_llm and self.config.force_template_mode:
            raise ValueError(
                "force_template_mode and require_llm cannot both be enabled."
            )

        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.langchain_available = LANGCHAIN_AVAILABLE
        self.generation_mode = "template"
        self.llm_enabled = self.langchain_available and not self._is_placeholder_key(
            self.api_key
        )
        self._llm_disabled_reason: Optional[str] = None
        self._mode_announced = False

        self.llm_technical = None
        self.llm_business = None
        self.technical_chain = None
        self.business_chain = None

        if self.config.force_template_mode:
            self.llm_enabled = False
            self._llm_disabled_reason = "Template mode forced by configuration."
            return

        if not self.llm_enabled:
            if not self.langchain_available:
                self._llm_disabled_reason = (
                    "LangChain LLM dependencies are unavailable "
                    f"({LANGCHAIN_IMPORT_ERROR})."
                )
            else:
                self._llm_disabled_reason = "No valid OPENAI_API_KEY detected."
            if self.config.require_llm:
                if not self.langchain_available:
                    raise ValueError(
                        "LLM mode requires langchain-core and langchain-openai. "
                        f"Import error: {LANGCHAIN_IMPORT_ERROR}"
                    )
                raise ValueError(
                    "Set a valid OPENAI_API_KEY or disable require_llm to use template mode."
                )
            return

        try:
            self._setup_chains()
            self.generation_mode = "llm"
        except Exception as exc:
            self.llm_enabled = False
            self._llm_disabled_reason = (
                "LLM initialization failed "
                f"({exc.__class__.__name__}: {exc})."
            )
            if self.config.require_llm:
                raise ValueError(self._llm_disabled_reason) from exc

    @staticmethod
    def _is_placeholder_key(key: str) -> bool:
        if not key:
            return True
        normalized = key.strip().lower()
        return (
            normalized.endswith("_here")
            or "your_" in normalized
            or "api_key_here" in normalized
        )

    def _setup_chains(self) -> None:
        if not self.langchain_available:
            raise RuntimeError(
                "LangChain dependencies are unavailable, so LLM-backed report "
                "generation cannot be initialized."
            )
        technical_prompt = ChatPromptTemplate.from_messages(
            [("human", self._get_technical_report_prompt())]
        )
        business_prompt = ChatPromptTemplate.from_messages(
            [("human", self._get_business_translation_prompt())]
        )

        self.llm_technical = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.technical_temperature,
            openai_api_key=self.api_key,
        )
        self.llm_business = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.business_temperature,
            openai_api_key=self.api_key,
        )

        self.technical_chain = (
            technical_prompt | self.llm_technical | StrOutputParser()
        )
        self.business_chain = business_prompt | self.llm_business | StrOutputParser()

    def _announce_generation_mode(self) -> None:
        if self._mode_announced:
            return
        if self.llm_enabled:
            print(
                f"Using LLM-backed report generation with model '{self.config.llm_model}'."
            )
        else:
            reason = self._llm_disabled_reason or "No valid OPENAI_API_KEY detected."
            print(f"{reason} Using built-in template report generation.")
        self._mode_announced = True

    def validate_input_json(self, json_data: Dict[str, Any]) -> None:
        """Validate the normalized JSON structure required by the report layer."""
        missing_sections = [
            section
            for section in self.NORMALIZED_REPORT_SECTIONS
            if section not in json_data
        ]
        if missing_sections:
            raise ValueError(
                "Input JSON is missing required sections: "
                + ", ".join(missing_sections)
                + ". Upstream agents must provide the normalized report schema."
            )

    @classmethod
    def _has_required_sections(
        cls, json_data: Dict[str, Any], required_sections: list[str]
    ) -> bool:
        return all(section in json_data for section in required_sections)

    @classmethod
    def _is_normalized_schema(cls, json_data: Dict[str, Any]) -> bool:
        return cls._has_required_sections(json_data, cls.NORMALIZED_REPORT_SECTIONS)

    @classmethod
    def _is_source_new_schema(cls, json_data: Dict[str, Any]) -> bool:
        return cls._has_required_sections(json_data, cls.SOURCE_NEW_SCHEMA_SECTIONS)

    @classmethod
    def _detect_schema_kind(cls, json_data: Dict[str, Any]) -> Optional[str]:
        is_normalized = cls._is_normalized_schema(json_data)
        is_source_new = cls._is_source_new_schema(json_data)

        if is_normalized:
            return "normalized"
        if is_source_new:
            return "source_new"
        return None

    @staticmethod
    def _coerce_list(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _extract_metric_value(
        metrics: Dict[str, Any], metric_name: str
    ) -> Optional[float]:
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _compute_class_imbalance_ratio(
        class_distribution: Dict[str, Any]
    ) -> Optional[float]:
        values = [
            float(value)
            for value in class_distribution.values()
            if isinstance(value, (int, float)) and float(value) > 0
        ]
        if len(values) < 2:
            return None
        return round(max(values) / min(values), 3)

    @staticmethod
    def _normalize_confusion_matrix(confusion_matrix: Dict[str, Any]) -> Dict[str, Any]:
        if not confusion_matrix:
            return {}
        if isinstance(confusion_matrix, list):
            return {"matrix": confusion_matrix}
        if not isinstance(confusion_matrix, dict):
            return {"value": confusion_matrix}
        if {"tn", "fp", "fn", "tp"}.issubset(confusion_matrix.keys()):
            return {
                "true_negative": confusion_matrix.get("tn"),
                "false_positive": confusion_matrix.get("fp"),
                "false_negative": confusion_matrix.get("fn"),
                "true_positive": confusion_matrix.get("tp"),
            }
        return confusion_matrix

    @staticmethod
    def _normalize_operations(actions: Any) -> list:
        normalized = []
        for index, item in enumerate(
            MultiAgentReportGenerator._coerce_list(actions), start=1
        ):
            if isinstance(item, dict):
                normalized.append(
                    {
                        "operation": item.get("operation", f"step_{index}"),
                        "detail": item.get("detail", item.get("action", "")),
                        "affected_columns": item.get("affected_columns", ""),
                    }
                )
            else:
                normalized.append(
                    {
                        "operation": f"step_{index}",
                        "detail": str(item),
                        "affected_columns": "",
                    }
                )
        return normalized

    def _normalize_new_schema(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        project_info = json_data.get("project_info", {})
        dataset_summary = json_data.get("dataset_summary", {})
        pipeline_trace = json_data.get("pipeline_trace", {})
        model_results = json_data.get("model_results", {})
        interpretability = json_data.get("interpretability", {})
        risk_scoring = json_data.get("risk_scoring", {})
        business_constraints = json_data.get("business_constraints", {})
        reporting_preferences = json_data.get("reporting_preferences", {})

        candidate_models = self._coerce_list(model_results.get("candidate_models"))
        selected_model = model_results.get("selected_model", {})
        selected_metrics = selected_model.get("metrics", {})
        primary_metric = (
            pipeline_trace.get("evaluation_setup", {}).get("primary_metric")
            or model_results.get("primary_metric")
            or "f1_score"
        )
        primary_score = self._extract_metric_value(selected_metrics, primary_metric)

        class_distribution = dataset_summary.get("class_distribution", {})
        feature_importance = self._coerce_list(
            interpretability.get("feature_importance")
            or interpretability.get("feature_importances")
        )

        normalized_models = []
        for rank, model in enumerate(candidate_models, start=1):
            metrics = model.get("metrics", {})
            normalized_models.append(
                {
                    "name": model.get("model_name", model.get("name", f"Model {rank}")),
                    "rank": model.get("rank", rank),
                    **metrics,
                }
            )

        business_goal = project_info.get("business_objective")
        if isinstance(project_info.get("success_criteria"), list):
            success_summary = "; ".join(
                str(item) for item in project_info["success_criteria"]
            )
        else:
            success_summary = None

        normalized = {
            "meta": {
                "schema_version": json_data.get("schema_version", "2.0"),
                "pipeline_id": project_info.get(
                    "project_id", project_info.get("project_name", "team_project")
                ),
                "dataset_name": dataset_summary.get(
                    "dataset_name", project_info.get("project_name", "Unknown Project")
                ),
                "dataset_source": dataset_summary.get("data_source"),
                "target_variable": project_info.get("target_variable"),
                "task_type": project_info.get("problem_type"),
                "project_theme": project_info.get("project_name"),
                "project_description": project_info.get("target_definition"),
                "timestamp": json_data.get("generated_at"),
                "models_evaluated": len(candidate_models),
            },
            "data_understanding": {
                "n_rows": dataset_summary.get("num_rows"),
                "n_cols": dataset_summary.get("num_features"),
                "n_rows_after_cleaning": dataset_summary.get(
                    "num_rows_after_cleaning",
                    dataset_summary.get("num_rows"),
                ),
                "feature_types": dataset_summary.get("feature_types", {}),
                "class_distribution": class_distribution,
                "class_imbalance_ratio": dataset_summary.get(
                    "class_imbalance_ratio"
                )
                or self._compute_class_imbalance_ratio(class_distribution),
                "missing_values_summary": dataset_summary.get(
                    "missing_value_summary", {}
                ),
                "key_insights": self._coerce_list(dataset_summary.get("key_insights")),
            },
            "data_cleaning": {
                "operations_performed": self._normalize_operations(
                    pipeline_trace.get("data_cleaning", {}).get("actions")
                ),
                "outliers_detected": self._coerce_list(
                    pipeline_trace.get("data_cleaning", {}).get("outliers_detected")
                ),
                "data_quality_score": dataset_summary.get("data_quality_score"),
                "quality_notes": dataset_summary.get("data_quality_notes")
                or pipeline_trace.get("data_cleaning", {}).get("notes"),
            },
            "feature_engineering": {
                "features_created": self._coerce_list(
                    pipeline_trace.get("feature_engineering", {}).get(
                        "features_created"
                    )
                ),
                "features_dropped": self._coerce_list(
                    pipeline_trace.get("feature_engineering", {}).get(
                        "features_dropped"
                    )
                ),
                "encoding_applied": pipeline_trace.get("feature_engineering", {}).get(
                    "encoding_applied",
                    {},
                ),
                "feature_importances": feature_importance,
                "final_feature_count": dataset_summary.get(
                    "num_features_after_engineering",
                    dataset_summary.get("num_features"),
                ),
                "key_insights": self._coerce_list(interpretability.get("key_insights")),
            },
            "modeling": {
                "best_model": {
                    "name": selected_model.get(
                        "model_name", selected_model.get("name")
                    ),
                    "params": selected_model.get("params", {}),
                    "training_time_seconds": selected_model.get(
                        "training_time_seconds"
                    ),
                    "optimization_method": pipeline_trace.get(
                        "model_selection", {}
                    ).get("selection_strategy"),
                },
                "models_compared": normalized_models,
                "selection_reason": selected_model.get("selection_reason")
                or model_results.get("selection_reason"),
            },
            "evaluation": {
                "primary_metric": primary_metric,
                "primary_score": primary_score,
                "metrics": selected_metrics,
                "confusion_matrix": self._normalize_confusion_matrix(
                    selected_model.get("confusion_matrix", {})
                ),
                "cv_scores": self._coerce_list(
                    selected_model.get("cv_scores") or model_results.get("cv_scores")
                ),
                "cv_mean": selected_model.get("cv_mean")
                or model_results.get("cv_mean"),
                "cv_std": selected_model.get("cv_std")
                or model_results.get("cv_std"),
                "baseline_score": model_results.get("baseline_score"),
                "improvement_over_baseline": model_results.get(
                    "improvement_over_baseline"
                ),
                "key_insights": self._coerce_list(model_results.get("key_insights")),
            },
            "business_context": {
                "industry": project_info.get("industry"),
                "use_case": business_goal,
                "target_audience": project_info.get("target_audience")
                or ", ".join(
                    str(item)
                    for item in self._coerce_list(project_info.get("stakeholders"))
                ),
                "stakeholders": self._coerce_list(project_info.get("stakeholders")),
                "business_goal": business_goal,
                "project_objective": project_info.get("target_definition"),
                "report_language": reporting_preferences.get("language"),
                "report_detail_level": reporting_preferences.get(
                    "business_report_format"
                ),
                "decision_threshold": risk_scoring.get("risk_threshold"),
                "action_budget": business_constraints.get("action_budget"),
                "action_cost_per_case": business_constraints.get(
                    "action_cost_per_case"
                ),
                "value_per_case": business_constraints.get("value_per_case"),
                "available_actions": self._coerce_list(
                    business_constraints.get("available_actions")
                ),
                "max_priority_cases": self._first_non_empty(
                    business_constraints.get("max_priority_cases"),
                    risk_scoring.get("risk_summary", {}).get("total_high_risk"),
                ),
                "preferred_strategy": business_constraints.get("preferred_strategy"),
                "success_criteria": success_summary,
            },
            "project_info": project_info,
            "dataset_summary": dataset_summary,
            "pipeline_trace": pipeline_trace,
            "model_results": model_results,
            "interpretability": interpretability,
            "risk_scoring": risk_scoring,
            "business_constraints": business_constraints,
            "reporting_preferences": reporting_preferences,
        }
        return normalized

    def _normalize_input_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(json_data, dict):
            raise TypeError("Input data must be a JSON object.")

        schema_kind = self._detect_schema_kind(json_data)

        if schema_kind == "normalized":
            normalized = json_data
        elif schema_kind == "source_new":
            normalized = self._normalize_new_schema(json_data)
        else:
            raise ValueError(
                "Input JSON does not match a supported schema. Supported schemas are "
                "normalized report format "
                "(meta/data_understanding/data_cleaning/feature_engineering/modeling/evaluation/business_context) "
                "and source pipeline format "
                "(project_info/dataset_summary/...)."
            )

        self.validate_input_json(normalized)
        return normalized

    def _load_json_file(self, json_file_path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
        json_path = Path(json_file_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        return self._normalize_input_json(json_data)

    def _load_input_json(self, input_data: ReportInput) -> Dict[str, Any]:
        if isinstance(input_data, Mapping):
            return self._normalize_input_json(dict(input_data))
        if isinstance(input_data, (str, os.PathLike)):
            return self._load_json_file(input_data)
        raise TypeError(
            "input_data must be either a JSON file path or an already loaded dictionary."
        )

    @classmethod
    def _describe_report_language(cls, report_language: Optional[str]) -> str:
        if not report_language:
            return "English"
        normalized = str(report_language).strip().lower()
        return cls.LANGUAGE_ALIASES.get(normalized, str(report_language).strip())

    def _resolve_report_language(self, json_data: Dict[str, Any]) -> str:
        business_context = json_data.setdefault("business_context", {})
        report_language = (
            business_context.get("report_language") or self.config.report_language
        )
        business_context["report_language"] = report_language
        return str(report_language)

    @staticmethod
    def _format_optional_context(
        label: str, value: Optional[object]
    ) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            if not value:
                return None
            value = ", ".join(str(item) for item in value)
        value_str = str(value).strip()
        if not value_str:
            return None
        return f"- {label}: {value_str}"

    def _build_project_context(self, json_data: Dict[str, Any]) -> str:
        meta = json_data.get("meta", {})
        business_context = json_data.get("business_context", {})

        context_lines = [
            self._format_optional_context("Dataset name", meta.get("dataset_name")),
            self._format_optional_context("Dataset source", meta.get("dataset_source")),
            self._format_optional_context("Task type", meta.get("task_type")),
            self._format_optional_context(
                "Target variable", meta.get("target_variable")
            ),
            self._format_optional_context("Project theme", meta.get("project_theme")),
            self._format_optional_context(
                "Project description", meta.get("project_description")
            ),
            self._format_optional_context("Use case", business_context.get("use_case")),
            self._format_optional_context("Industry", business_context.get("industry")),
            self._format_optional_context(
                "Target audience", business_context.get("target_audience")
            ),
            self._format_optional_context(
                "Stakeholders", business_context.get("stakeholders")
            ),
            self._format_optional_context(
                "Report language",
                self._describe_report_language(
                    business_context.get("report_language")
                ),
            ),
            self._format_optional_context(
                "Business goal", business_context.get("business_goal")
            ),
            self._format_optional_context(
                "Project objective", business_context.get("project_objective")
            ),
        ]
        usable_lines = [line for line in context_lines if line]
        if not usable_lines:
            return (
                "- No extra project context was provided. Infer the topic strictly "
                "from the JSON content."
            )
        return "\n".join(usable_lines)

    def _build_prompt_payload(self, json_data: Dict[str, Any]) -> Dict[str, str]:
        report_language_name = self._describe_report_language(
            self._resolve_report_language(json_data)
        )
        return {
            "json_data": json.dumps(json_data, ensure_ascii=False, indent=2),
            "project_context": self._build_project_context(json_data),
            "report_language_name": report_language_name,
        }

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None or value == "":
            return "N/A"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            if value.is_integer():
                return f"{int(value):,}"
            formatted = f"{value:,.3f}".rstrip("0").rstrip(".")
            return formatted
        if isinstance(value, list):
            if not value:
                return "N/A"
            return ", ".join(str(item) for item in value)
        if isinstance(value, dict):
            if not value:
                return "N/A"
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _format_percent(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "N/A"
        return f"{value * 100:.1f}%"

    @staticmethod
    def _titleize_metric_name(metric_name: str) -> str:
        return metric_name.replace("_", " ").title()

    @staticmethod
    def _markdown_bullets(items: list[str]) -> str:
        if not items:
            return "- N/A"
        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
        if denominator == 0:
            return None
        return numerator / denominator

    def _top_feature_summaries(
        self, feature_importances: list[Any], limit: int = 5
    ) -> list[str]:
        summaries = []
        for item in feature_importances[:limit]:
            if isinstance(item, dict):
                feature = item.get("feature", "Unknown feature")
                importance = item.get("importance")
                direction = item.get("direction")
                detail = f"{feature} ({self._format_value(importance)})"
                if direction:
                    detail += f": {direction}"
                summaries.append(detail)
            else:
                summaries.append(str(item))
        return summaries

    def _build_metric_table(
        self, metrics: Dict[str, Any], left_header: str, right_header: str
    ) -> str:
        if not metrics:
            return f"| {left_header} | {right_header} |\n| --- | --- |\n| N/A | N/A |"
        rows = [f"| {left_header} | {right_header} |", "| --- | --- |"]
        for metric_name, value in metrics.items():
            rows.append(
                f"| {self._titleize_metric_name(str(metric_name))} | {self._format_value(value)} |"
            )
        return "\n".join(rows)

    def _build_confusion_matrix_table(
        self, confusion_matrix: Dict[str, Any], left_header: str, right_header: str
    ) -> str:
        if not confusion_matrix:
            return f"| {left_header} | {right_header} |\n| --- | --- |\n| N/A | N/A |"
        rows = [f"| {left_header} | {right_header} |", "| --- | --- |"]
        for metric_name, value in confusion_matrix.items():
            rows.append(
                f"| {self._titleize_metric_name(str(metric_name))} | {self._format_value(value)} |"
            )
        return "\n".join(rows)

    @staticmethod
    def _first_non_empty(*values: Any) -> Any:
        for value in values:
            if value not in (None, "", [], {}):
                return value
        return None

    def _build_chart_recommendations(self, task_type: str) -> list[str]:
        task = (task_type or "").lower()
        recommendations = ["Top-10 feature importance bar chart"]
        if "classification" in task:
            recommendations.extend(
                [
                    "Confusion matrix heatmap",
                    "ROC curve",
                    "Precision-recall curve",
                ]
            )
        elif "regression" in task:
            recommendations.extend(
                [
                    "Residual plot",
                    "Predicted vs actual scatter plot",
                    "Error distribution plot",
                ]
            )
        else:
            recommendations.append("Learning curve or validation curve")
        return recommendations

    def _build_template_technical_recommendations(
        self, json_data: Dict[str, Any], chinese: bool
    ) -> list[str]:
        data_understanding = json_data.get("data_understanding", {})
        evaluation = json_data.get("evaluation", {})
        recommendations = []

        imbalance_ratio = data_understanding.get("class_imbalance_ratio")
        if isinstance(imbalance_ratio, (int, float)) and imbalance_ratio > 1.5:
            recommendations.append(
                "Review threshold tuning and class-balancing strategies to protect minority-class recall."
            )

        cv_std = evaluation.get("cv_std")
        if isinstance(cv_std, (int, float)) and cv_std > 0.02:
            recommendations.append(
                "Investigate model stability because cross-validation variance is relatively high."
            )

        missing_values = data_understanding.get("missing_values_summary", {})
        if missing_values not in ({}, None):
            recommendations.append(
                "Keep monitoring missing-value patterns and data quality drift in production feeds."
            )

        recommendations.append(
            "Establish a monitoring plan for feature drift, score drift, and periodic model refresh."
        )
        recommendations.append(
            "Collect additional explanatory variables to improve model robustness and actionability."
        )
        return recommendations

    def _build_template_technical_report(
        self, json_data: Dict[str, Any], report_language_name: str
    ) -> str:
        meta = json_data.get("meta", {})
        data_understanding = json_data.get("data_understanding", {})
        data_cleaning = json_data.get("data_cleaning", {})
        feature_engineering = json_data.get("feature_engineering", {})
        modeling = json_data.get("modeling", {})
        evaluation = json_data.get("evaluation", {})

        dataset_name = meta.get("dataset_name", "Unknown dataset")
        target_variable = meta.get("target_variable", "unknown target")
        task_type = meta.get("task_type", "unknown task")
        best_model = modeling.get("best_model", {})
        best_model_name = best_model.get("name", "N/A")
        primary_metric = evaluation.get("primary_metric", "primary metric")
        primary_score = evaluation.get("primary_score")
        top_features = self._top_feature_summaries(
            self._coerce_list(feature_engineering.get("feature_importances"))
        )
        charts = self._build_chart_recommendations(task_type)
        recommendations = self._build_template_technical_recommendations(
            json_data, chinese=False
        )
        summary_intro = (
            "This technical report was generated with the built-in template mode. "
            f"The project analyses **{dataset_name}** for **{target_variable}** using a **{task_type}** workflow. "
            f"The selected model is **{best_model_name}**, and the primary metric **{primary_metric}** reached **{self._format_value(primary_score)}**."
        )
        overview_lines = [
            f"Dataset size: {self._format_value(data_understanding.get('n_rows'))} rows and {self._format_value(data_understanding.get('n_cols'))} columns.",
            f"Rows after cleaning: {self._format_value(data_understanding.get('n_rows_after_cleaning'))}.",
            f"Class imbalance ratio: {self._format_value(data_understanding.get('class_imbalance_ratio'))}.",
            f"Data quality score: {self._format_value(data_cleaning.get('data_quality_score'))}.",
            f"Quality notes: {self._format_value(data_cleaning.get('quality_notes'))}.",
        ]
        feature_lines = [
            f"Features created: {self._format_value(feature_engineering.get('features_created'))}.",
            f"Features dropped: {self._format_value(feature_engineering.get('features_dropped'))}.",
            f"Encoding or transformation steps: {self._format_value(feature_engineering.get('encoding_applied'))}.",
            "Top feature signals:",
            self._markdown_bullets(
                top_features or ["No feature-importance details were provided."]
            ),
        ]
        model_lines = [
            f"Selected model: **{best_model_name}**.",
            f"Selection reason: {self._format_value(modeling.get('selection_reason'))}.",
            f"Optimization or selection strategy: {self._format_value(best_model.get('optimization_method'))}.",
            f"Training time (seconds): {self._format_value(best_model.get('training_time_seconds'))}.",
        ]
        performance_title = "Metric"
        performance_value = "Value"
        confusion_title = "Confusion Matrix Item"
        confusion_value = "Value"
        chart_heading = "### Recommended Visualizations"
        rec_heading = "### Improvement Recommendations"
        section_titles = [
            "# Executive Summary",
            "## 1. Data Overview and Quality Assessment",
            "## 2. Feature Engineering Analysis",
            "## 3. Model Comparison and Selection",
            "## 4. Performance Evaluation",
            "## 5. Technical Recommendations and Improvement Directions",
        ]

        report_sections = [
            section_titles[0],
            "",
            summary_intro,
            "",
            section_titles[1],
            "",
            self._markdown_bullets(overview_lines),
            "",
            self._markdown_bullets(
                [
                    str(item)
                    for item in self._coerce_list(data_understanding.get("key_insights"))
                ]
            ),
            "",
            section_titles[2],
            "",
            "\n".join(feature_lines),
            "",
            self._markdown_bullets(
                [
                    str(item)
                    for item in self._coerce_list(feature_engineering.get("key_insights"))
                ]
            ),
            "",
            section_titles[3],
            "",
            self._markdown_bullets(model_lines),
            "",
            section_titles[4],
            "",
            self._build_metric_table(
                evaluation.get("metrics", {}),
                performance_title,
                performance_value,
            ),
            "",
            self._build_confusion_matrix_table(
                evaluation.get("confusion_matrix", {}),
                confusion_title,
                confusion_value,
            ),
            "",
            self._markdown_bullets(
                [
                    str(item)
                    for item in self._coerce_list(evaluation.get("key_insights"))
                ]
            ),
            "",
            section_titles[5],
            "",
            rec_heading,
            "",
            self._markdown_bullets(recommendations),
            "",
            chart_heading,
            "",
            self._markdown_bullets(charts),
        ]
        return "\n".join(report_sections).strip() + "\n"

    @staticmethod
    def _build_business_technical_context(technical_report: Optional[str]) -> str:
        if not technical_report or not technical_report.strip():
            return ""
        return "\n".join(
            [
                "[Optional technical report context]",
                technical_report.strip(),
            ]
        )

    def _build_action_table(
        self,
        rows: list[Dict[str, str]],
        headers: list[str],
    ) -> str:
        table_lines = [
            f"| {' | '.join(headers)} |",
            f"| {' | '.join(['---'] * len(headers))} |",
        ]
        for row in rows:
            table_lines.append(
                "| "
                + " | ".join(row.get(header, "N/A") for header in headers)
                + " |"
            )
        return "\n".join(table_lines)

    def _build_business_action_rows(
        self, json_data: Dict[str, Any], chinese: bool
    ) -> list[Dict[str, str]]:
        business_context = json_data.get("business_context", {})
        business_constraints = json_data.get("business_constraints", {})
        risk_scoring = json_data.get("risk_scoring", {})

        stakeholders = self._coerce_list(business_context.get("stakeholders"))
        owner = (
            str(stakeholders[0])
            if stakeholders
            else business_context.get("target_audience", "Business operations lead")
        )
        threshold = self._format_value(business_context.get("decision_threshold"))
        max_cases = self._format_value(
            self._first_non_empty(
                business_context.get("max_priority_cases"),
                business_constraints.get("max_priority_cases"),
                risk_scoring.get("risk_summary", {}).get("total_high_risk"),
            )
        )
        resources = self._format_value(
            self._first_non_empty(
                business_context.get("available_actions"),
                business_constraints.get("available_actions"),
                business_context.get("action_budget"),
                business_constraints.get("action_budget"),
            )
        )

        return [
            {
                "Priority": "High",
                "Action": f"Launch the first intervention wave for cases above threshold {threshold} and prioritize the top {max_cases} cases.",
                "Owner": owner,
                "Timeline": "1-2 weeks",
                "Expected Result": "Focus limited operational capacity on the highest-value high-risk cases.",
                "Required Resources": resources,
            },
            {
                "Priority": "Medium",
                "Action": "Create a human-review and feedback loop so intervention outcomes feed back into model monitoring.",
                "Owner": owner,
                "Timeline": "2-4 weeks",
                "Expected Result": "Improve explainability, operational trust, and the evidence base for the next model refresh.",
                "Required Resources": "Review workflow, outcome tracking sheet, recurring governance meeting",
            },
        ]

    def _build_roi_section(self, json_data: Dict[str, Any], chinese: bool) -> str:
        business_context = json_data.get("business_context", {})
        risk_scoring = json_data.get("risk_scoring", {})
        business_constraints = json_data.get("business_constraints", {})

        case_count = self._first_non_empty(
            business_context.get("max_priority_cases"),
            business_constraints.get("max_priority_cases"),
            risk_scoring.get("risk_summary", {}).get("total_high_risk"),
        )
        action_cost = self._first_non_empty(
            business_context.get("action_cost_per_case"),
            business_constraints.get("action_cost_per_case"),
        )
        value_per_case = self._first_non_empty(
            business_context.get("value_per_case"),
            business_constraints.get("value_per_case"),
        )

        missing = []
        if not isinstance(case_count, (int, float)):
            missing.append("case_count")
        if not isinstance(action_cost, (int, float)):
            missing.append("action_cost_per_case")
        if not isinstance(value_per_case, (int, float)):
            missing.append("value_per_case")

        if missing:
            return (
                "Exact ROI cannot be calculated because the following inputs are missing: "
                + ", ".join(missing)
                + ". Add them to the JSON payload for a precise estimate."
            )

        investment = float(case_count) * float(action_cost)
        expected_return = float(case_count) * float(value_per_case)
        net_benefit = expected_return - investment
        roi = self._safe_divide(net_benefit, investment)

        headers = ["Metric", "Value"]
        rows = {
            "Intervention Cases": self._format_value(case_count),
            "Estimated Investment": self._format_value(investment),
            "Expected Return": self._format_value(expected_return),
            "Net Benefit": self._format_value(net_benefit),
            "ROI": self._format_percent(roi) if roi is not None else "N/A",
        }
        return self._build_metric_table(rows, headers[0], headers[1])

    def _build_risk_notes(self, json_data: Dict[str, Any], chinese: bool) -> list[str]:
        risk_assessment = json_data.get("risk_assessment", {})
        model_limitations = self._coerce_list(risk_assessment.get("model_limitations"))
        ethical_considerations = self._coerce_list(
            risk_assessment.get("ethical_considerations")
        )
        notes = [str(item) for item in model_limitations + ethical_considerations]

        if not notes:
            notes = [
                "Monitor data quality, model drift, and operational adoption risks during rollout.",
                "Confirm privacy, fairness, and governance controls before using predictions in production.",
            ]
        return notes

    def _build_implementation_roadmap(
        self, json_data: Dict[str, Any], chinese: bool
    ) -> list[str]:
        business_context = json_data.get("business_context", {})
        use_case = business_context.get("use_case", "the target use case")

        return [
            f"Phase 1 (1-2 weeks): align on goals, thresholds, and review ownership for {use_case}.",
            "Phase 2 (2-4 weeks): run the first intervention pilot and capture outcome feedback.",
            "Phase 3 (ongoing): monitor performance, review ROI, and refresh the model on a regular cadence.",
        ]

    def _build_template_business_report(
        self,
        json_data: Dict[str, Any],
        report_language_name: str,
        technical_report: Optional[str] = None,
    ) -> str:
        meta = json_data.get("meta", {})
        business_context = json_data.get("business_context", {})
        evaluation = json_data.get("evaluation", {})

        dataset_name = meta.get("dataset_name", "Unknown dataset")
        use_case = business_context.get("use_case", "the target use case")
        primary_metric = evaluation.get("primary_metric", "primary metric")
        primary_score = evaluation.get("primary_score")
        decision_threshold = business_context.get("decision_threshold")
        action_rows = self._build_business_action_rows(json_data, chinese=False)
        risk_notes = self._build_risk_notes(json_data, chinese=False)
        roadmap = self._build_implementation_roadmap(json_data, chinese=False)
        technical_context_note = technical_report.strip() if technical_report else ""
        headers = [
            "Priority",
            "Action",
            "Owner",
            "Timeline",
            "Expected Result",
            "Required Resources",
        ]
        summary_intro = (
            f"This business report translates the **{dataset_name}** model output into action for **{use_case}**. "
            f"The primary metric **{primary_metric}** is **{self._format_value(primary_score)}**, and the recommended first-wave decisions should use threshold **{self._format_value(decision_threshold)}**."
        )
        if technical_context_note:
            summary_intro += " An existing technical report was used as additional context."
        section_titles = [
            "# Executive Summary",
            "## 1. Key Findings",
            "## 2. Immediate Action Recommendations",
            "## 3. ROI Analysis",
            "## 4. Risk Notes",
            "## 5. Implementation Roadmap",
        ]
        key_findings = [
            f"Business goal: {self._format_value(business_context.get('business_goal'))}.",
            f"Target audience: {self._format_value(business_context.get('target_audience'))}.",
            f"Primary metric {primary_metric}: {self._format_value(primary_score)}.",
            f"Decision threshold: {self._format_value(decision_threshold)}.",
            f"Preferred strategy: {self._format_value(business_context.get('preferred_strategy'))}.",
        ]

        action_table = self._build_action_table(action_rows, headers)
        report_sections = [
            section_titles[0],
            "",
            summary_intro,
            "",
            section_titles[1],
            "",
            self._markdown_bullets(key_findings),
            "",
            section_titles[2],
            "",
            action_table,
            "",
            section_titles[3],
            "",
            self._build_roi_section(json_data, chinese=False),
            "",
            section_titles[4],
            "",
            self._markdown_bullets(risk_notes),
            "",
            section_titles[5],
            "",
            self._markdown_bullets(roadmap),
        ]
        return "\n".join(report_sections).strip() + "\n"

    def _generate_technical_report_content(self, json_data: Dict[str, Any]) -> str:
        payload = self._build_prompt_payload(json_data)
        report_language_name = payload["report_language_name"]

        if self.technical_chain is None:
            return self._build_template_technical_report(json_data, report_language_name)

        try:
            return self.technical_chain.invoke(payload)
        except Exception as exc:
            if self.config.require_llm:
                raise
            print(
                f"Technical LLM generation failed: {exc}. Falling back to template mode."
            )
            return self._build_template_technical_report(json_data, report_language_name)

    def _generate_business_report_content(
        self,
        json_data: Dict[str, Any],
        technical_report: Optional[str] = None,
    ) -> str:
        payload = self._build_prompt_payload(json_data)
        report_language_name = payload["report_language_name"]
        payload["technical_report_context"] = self._build_business_technical_context(
            technical_report
        )

        if self.business_chain is None:
            return self._build_template_business_report(
                json_data,
                report_language_name,
                technical_report=technical_report,
            )

        try:
            return self.business_chain.invoke(payload)
        except Exception as exc:
            if self.config.require_llm:
                raise
            print(
                f"Business LLM generation failed: {exc}. Falling back to template mode."
            )
            return self._build_template_business_report(
                json_data,
                report_language_name,
                technical_report=technical_report,
            )

    def _resolve_business_technical_context_report(
        self,
        json_data: Dict[str, Any],
        technical_report: Optional[str],
        include_technical_context: bool,
    ) -> Optional[str]:
        if not include_technical_context:
            return None
        if technical_report and technical_report.strip():
            return technical_report
        return self._generate_technical_report_content(json_data)

    def _save_text_report(self, filename_prefix: str, content: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{filename_prefix}_{timestamp}.md"
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        return str(path)

    def _save_reports(
        self, technical_report: str, business_report: str
    ) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        technical_path = self.output_dir / f"technical_report_{timestamp}.md"
        business_path = self.output_dir / f"business_report_{timestamp}.md"
        combined_markdown_path = self.output_dir / "report.md"
        report_json_path = self.output_dir / "report.json"

        with open(technical_path, "w", encoding="utf-8") as file:
            file.write(technical_report)
        with open(business_path, "w", encoding="utf-8") as file:
            file.write(business_report)
        with open(combined_markdown_path, "w", encoding="utf-8") as file:
            file.write(self._build_combined_report_markdown(technical_report, business_report))
        with open(report_json_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "status": "success",
                    "agent_name": self.AGENT_NAME,
                    "generation_mode": self.generation_mode,
                    "generated_at": datetime.now().isoformat(),
                    "technical_report": technical_report,
                    "business_report": business_report,
                    "saved_paths": {
                        "technical_report": str(technical_path),
                        "business_report": str(business_path),
                        "combined_markdown": str(combined_markdown_path),
                        "report_json": str(report_json_path),
                    },
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved technical report to: {technical_path}")
        print(f"Saved business report to: {business_path}")
        print(f"Saved combined markdown report to: {combined_markdown_path}")
        print(f"Saved report payload to: {report_json_path}")
        return {
            "technical_report": str(technical_path),
            "business_report": str(business_path),
            "combined_markdown": str(combined_markdown_path),
            "report_json": str(report_json_path),
        }

    @staticmethod
    def _build_combined_report_markdown(
        technical_report: str, business_report: str
    ) -> str:
        return "\n".join(
            [
                "# AutoDS Report Package",
                "",
                "## Technical Report",
                "",
                technical_report.strip(),
                "",
                "---",
                "",
                "## Business Report",
                "",
                business_report.strip(),
                "",
            ]
        )

    def run(
        self,
        input_data: ReportInput,
        save_reports: bool = True,
        business_include_technical_context: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Unified main entry point for report generation."""
        return self._generate_reports_impl(
            input_data=input_data,
            save_reports=save_reports,
            business_include_technical_context=business_include_technical_context,
        )

    def generate_reports(
        self,
        input_data: ReportInput,
        save_reports: bool = True,
        business_include_technical_context: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible alias for run()."""
        return self.run(
            input_data=input_data,
            save_reports=save_reports,
            business_include_technical_context=business_include_technical_context,
        )

    def _generate_reports_impl(
        self,
        input_data: ReportInput,
        save_reports: bool = True,
        business_include_technical_context: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self._announce_generation_mode()
        try:
            json_data = self._load_input_json(input_data)
            print("Generating technical and business reports...")
            include_technical_context = (
                self.config.business_include_technical_context
                if business_include_technical_context is None
                else business_include_technical_context
            )
            technical_report = self._generate_technical_report_content(json_data)
            business_technical_context = (
                self._resolve_business_technical_context_report(
                    json_data,
                    technical_report=technical_report,
                    include_technical_context=include_technical_context,
                )
            )
            business_report = self._generate_business_report_content(
                json_data,
                technical_report=business_technical_context,
            )
            saved_paths = (
                self._save_reports(technical_report, business_report)
                if save_reports
                else None
            )
            return {
                "status": "success",
                "technical_report": technical_report,
                "business_report": business_report,
                "saved_paths": saved_paths,
                "agent_name": self.AGENT_NAME,
                "generation_mode": self.generation_mode,
                "business_used_technical_context": include_technical_context,
            }
        except Exception as exc:
            return {
                "status": "failure",
                "technical_report": None,
                "business_report": None,
                "saved_paths": None,
                "agent_name": self.AGENT_NAME,
                "generation_mode": self.generation_mode,
                "business_used_technical_context": False,
                "error": str(exc),
            }

    def generate_technical_report_only(
        self,
        input_data: ReportInput,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        self._announce_generation_mode()
        try:
            json_data = self._load_input_json(input_data)
            technical_report = self._generate_technical_report_content(json_data)
            saved_paths = None

            if save_report:
                path = self._save_text_report("technical_report", technical_report)
                print(f"Saved technical report to: {path}")
                saved_paths = {"technical_report": path}

            return {
                "status": "success",
                "technical_report": technical_report,
                "saved_paths": saved_paths,
                "agent_name": self.AGENT_NAME,
                "generation_mode": self.generation_mode,
            }
        except Exception as exc:
            return {
                "status": "failure",
                "technical_report": None,
                "saved_paths": None,
                "agent_name": self.AGENT_NAME,
                "generation_mode": self.generation_mode,
                "error": str(exc),
            }

    def generate_business_report_only(
        self,
        input_data: ReportInput,
        technical_report: Optional[str] = None,
        include_technical_context: Optional[bool] = None,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        self._announce_generation_mode()
        try:
            json_data = self._load_input_json(input_data)
            use_technical_context = (
                self.config.business_include_technical_context
                if include_technical_context is None
                else include_technical_context
            )
            business_technical_context = (
                self._resolve_business_technical_context_report(
                    json_data,
                    technical_report=technical_report,
                    include_technical_context=use_technical_context,
                )
            )
            business_report = self._generate_business_report_content(
                json_data,
                technical_report=business_technical_context,
            )
            saved_paths = None

            if save_report:
                path = self._save_text_report("business_report", business_report)
                print(f"Saved business report to: {path}")
                saved_paths = {"business_report": path}

            return {
                "status": "success",
                "business_report": business_report,
                "saved_paths": saved_paths,
                "agent_name": self.AGENT_NAME,
                "generation_mode": self.generation_mode,
                "business_used_technical_context": use_technical_context,
            }
        except Exception as exc:
            return {
                "status": "failure",
                "business_report": None,
                "saved_paths": None,
                "agent_name": self.AGENT_NAME,
                "generation_mode": self.generation_mode,
                "business_used_technical_context": False,
                "error": str(exc),
            }

    def _get_technical_report_prompt(self) -> str:
        return """You are a senior data scientist with more than 10 years of machine learning project experience. Your job is to write a professional technical report from structured project results.

Generate a technical report based on the project context and JSON data below.

[Project context]
{project_context}

[JSON data]
{json_data}

[Report requirements]

0. Topic adaptation:
   - Infer the project theme, industry, and task strictly from the project context and JSON content.
   - Do not assume any specific domain, user group, or use case unless the JSON clearly says so.
   - For classification tasks, explain metrics such as accuracy, F1, recall, precision, class imbalance, and the confusion matrix.
   - For regression tasks, focus on MAE, RMSE, R2, error range, and business impact.
   - For other task types, prioritize the core metrics that actually appear in the JSON instead of forcing a fixed template.
   - Use the entity names from the JSON for the target, risk group, and business objects.

1. Structure:
   - # Executive Summary
   - ## 1. Data Overview and Quality Assessment
   - ## 2. Feature Engineering Analysis
   - ## 3. Model Comparison and Selection
   - ## 4. Performance Evaluation
   - ## 5. Technical Recommendations and Improvement Directions

2. Writing style:
   - Professional, precise, and data-driven
   - Use concrete numbers instead of vague descriptions
   - Explain technical terms clearly while staying professional
   - State both model strengths and limitations objectively

3. Metric interpretation:
   - For classification, explain accuracy, F1 score, precision, recall, confusion matrix, and class imbalance effects
   - For regression, explain MAE, RMSE, R2, and the business meaning of prediction error
   - For all tasks, explain the practical meaning and predictive logic of the top five important features or variables
   - If cross-validation results are available, assess model stability

4. Chart recommendations:
   Describe which charts should be produced by the downstream visualization layer, for example:
   - Top-10 feature importance bar chart
   - For classification: confusion matrix heatmap, ROC curve, PR curve
   - For regression: residual plot, predicted vs actual scatter plot, error distribution plot
   - Learning curve or validation curve

5. Technical recommendations:
   - Possible model improvement directions
   - Feature engineering opportunities
   - Useful future data collection ideas
   - Monitoring and model update strategy

6. Output language:
   - Write the entire report in {report_language_name}.
   - Do not mix languages.
   - If the requested language is English, every heading, paragraph, table label, and bullet point must be in English.

[Output format]
Return plain Markdown only.
Do not wrap the answer in a code block.
Start directly with `# Executive Summary`.
"""

    def _get_business_translation_prompt(self) -> str:
        return """You are a senior business analyst who is skilled at turning data science results into action plans that business teams can execute directly.

Translate the technical analysis into practical business recommendations based on the project context and JSON data.

[Project context]
{project_context}

[Optional technical report context]
{technical_report_context}

[JSON data]
{json_data}

[Report requirements]

0. Topic adaptation:
   - Infer the business scenario strictly from the project context and JSON.
   - Do not assume any specific domain, user group, or workflow unless the JSON clearly says so.
   - The names of people, teams, departments, and processes must match the current project theme.
   - If the JSON provides `industry`, `use_case`, `target_audience`, or `stakeholders`, use them.
   - Treat the optional technical report context as supplemental input only, not as a required dependency.
   - If the optional technical report context is empty or unavailable, infer the business recommendations directly from the JSON and project context.

1. Structure:
   - # Executive Summary
   - ## 1. Key Findings
   - ## 2. Immediate Action Recommendations
   - ## 3. ROI Analysis
   - ## 4. Risk Notes
   - ## 5. Implementation Roadmap

2. Writing style:
   - Written for non-technical managers
   - Avoid unnecessary technical jargon
   - Use business language that matches the project theme
   - Each recommendation should include owner, timeline, expected result, and required resources
   - Use concrete numbers whenever possible

3. Business translation examples:
   - Technical metric -> explain what it means for business decisions
   - Important feature -> explain why it should be monitored operationally
   - Recall gap -> explain what may still be missed and where manual review is needed
   - Regression error -> explain whether the prediction error is acceptable in the real use case

4. ROI analysis:
   - If the JSON contains cost, benefit, or conversion-value fields, present investment, expected return, net benefit, and ROI in a table.
   - If the JSON does not contain enough information, say that exact ROI cannot be calculated and list the missing inputs.

5. Recommendation format:
   - Priority
   - Action
   - Owner
   - Timeline
   - Expected result
   - Required resources

6. Risk notes:
   - Business implications of model limitations
   - Ethics, compliance, or privacy concerns
   - Implementation challenges such as data access, manual review cost, or user adoption

7. Output language:
   - Write the entire report in {report_language_name}.
   - Do not mix languages.
   - If the requested language is English, every heading, paragraph, table label, and bullet point must be in English.

[Output format]
Return plain Markdown only.
Do not wrap the answer in a code block.
Start directly with `# Executive Summary`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read upstream JSON output and generate technical and business reports."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="example_pipeline_output.json",
        help="Path to the input JSON file produced by upstream teammates or agents.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory where generated reports will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="LLM model name to use when an API key is available.",
    )
    parser.add_argument(
        "--technical-temperature",
        type=float,
        default=0.3,
        help="Temperature used for technical report generation.",
    )
    parser.add_argument(
        "--business-temperature",
        type=float,
        default=0.5,
        help="Temperature used for business report generation.",
    )
    parser.add_argument(
        "--report-language",
        type=str,
        default=os.getenv("REPORT_LANGUAGE", "en"),
        help="Default report language when the JSON does not override it.",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail instead of using template mode when a valid LLM configuration is unavailable.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force built-in template generation and skip all LLM initialization.",
    )
    parser.add_argument(
        "--business-use-technical-report",
        action="store_true",
        help="Pass the generated technical report into business report generation.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "technical", "business"],
        default="both",
        help="Generation mode: both, technical, or business.",
    )

    args = parser.parse_args()

    config = ReportGeneratorConfig(
        output_dir=args.output_dir,
        llm_model=args.model,
        technical_temperature=args.technical_temperature,
        business_temperature=args.business_temperature,
        report_language=args.report_language,
        require_llm=args.require_llm,
        force_template_mode=args.no_llm,
        business_include_technical_context=args.business_use_technical_report,
    )
    generator = MultiAgentReportGenerator(config=config)

    if args.mode == "both":
        result = generator.run(
            args.json,
            business_include_technical_context=args.business_use_technical_report,
        )
        if result["status"] != "success":
            raise SystemExit(f"Report generation failed: {result.get('error', 'Unknown error')}")
    elif args.mode == "technical":
        result = generator.generate_technical_report_only(args.json)
        if result["status"] != "success":
            raise SystemExit(
                f"Technical report generation failed: {result.get('error', 'Unknown error')}"
            )
    else:
        result = generator.generate_business_report_only(
            args.json,
            include_technical_context=args.business_use_technical_report,
        )
        if result["status"] != "success":
            raise SystemExit(
                f"Business report generation failed: {result.get('error', 'Unknown error')}"
            )

    print("Report generation completed.")


if __name__ == "__main__":
    main()


ReportGenerator = MultiAgentReportGenerator

__all__ = [
    "ReportGeneratorConfig",
    "ReportGenerator",
    "MultiAgentReportGenerator",
]
