#!/usr/bin/env python3
"""
Multi-agent report generation system.

This module receives structured JSON from upstream teammates or agents and
produces technical and business-facing Markdown reports.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field, replace as dataclass_replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from dotenv import load_dotenv
from utils import load_project_env, reexec_with_project_venv


if __name__ == "__main__":
    reexec_with_project_venv(__file__)
load_project_env(__file__)

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

try:
    from openai_compatible_chat import OpenAICompatibleChatClient

    OPENAI_COMPATIBLE_CHAT_AVAILABLE = True
    OPENAI_COMPATIBLE_CHAT_IMPORT_ERROR = None
except Exception as exc:
    OpenAICompatibleChatClient = None
    OPENAI_COMPATIBLE_CHAT_AVAILABLE = False
    OPENAI_COMPATIBLE_CHAT_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


@dataclass
class DirectPromptChain:
    prompt_template: str
    client: Any

    def invoke(self, payload: Dict[str, str]) -> str:
        prompt = self.prompt_template.format(**payload)
        response = self.client.invoke([{"role": "user", "content": prompt}])
        return str(getattr(response, "content", response))


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


@dataclass
class ReportPlannerInput:
    """Structured report-planning contract from an upstream planner."""

    source: str = "unknown"
    schema_version: str = "1.0"
    rationale: str = ""

    report_language: Optional[str] = None
    llm_model: Optional[str] = None
    technical_temperature: Optional[float] = None
    business_temperature: Optional[float] = None
    business_include_technical_context: Optional[bool] = None

    use_case: Optional[str] = None
    industry: Optional[str] = None
    target_audience: Optional[str] = None
    stakeholders: Optional[List[str]] = None
    business_goal: Optional[str] = None
    project_objective: Optional[str] = None

    required_sections: Optional[List[str]] = None
    technical_instructions: Optional[List[str]] = None
    business_instructions: Optional[List[str]] = None
    planner_review: Optional[Dict[str, Any]] = None
    planner_plan: Optional[Dict[str, Any]] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _coerce_list(value: Any) -> Optional[List[Any]]:
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportPlannerInput":
        report_block = data.get("report", {})
        business_block = data.get("business_context", data.get("business", {}))
        instructions_block = data.get("instructions", {})
        planner_block = data.get("planner", {})
        review_block = data.get("planner_review", planner_block.get("review", {}))
        plan_block = data.get("planner_plan", planner_block.get("plan", {}))

        def _flat_or_block(flat_key: str, *blocks: Dict[str, Any]) -> Any:
            if flat_key in data:
                return data[flat_key]
            for block in blocks:
                if isinstance(block, dict) and flat_key in block:
                    return block.get(flat_key)
            return None

        def _bool_or_block(flat_key: str, *blocks: Dict[str, Any]) -> Optional[bool]:
            if flat_key in data:
                return bool(data[flat_key])
            for block in blocks:
                if isinstance(block, dict) and flat_key in block:
                    return bool(block[flat_key])
            return None

        technical_instructions = cls._coerce_list(
            _flat_or_block("technical_instructions", instructions_block, report_block)
        )
        if technical_instructions is None and isinstance(instructions_block, dict):
            technical_instructions = cls._coerce_list(
                instructions_block.get("technical")
            )

        business_instructions = cls._coerce_list(
            _flat_or_block("business_instructions", instructions_block, report_block)
        )
        if business_instructions is None and isinstance(instructions_block, dict):
            business_instructions = cls._coerce_list(
                instructions_block.get("business")
            )

        known_keys = {
            "schema_version",
            "source",
            "rationale",
            "report",
            "business_context",
            "business",
            "instructions",
            "planner",
            "planner_review",
            "planner_plan",
            "report_language",
            "llm_model",
            "technical_temperature",
            "business_temperature",
            "business_include_technical_context",
            "use_case",
            "industry",
            "target_audience",
            "stakeholders",
            "business_goal",
            "project_objective",
            "required_sections",
            "technical_instructions",
            "business_instructions",
        }

        planner_review = dict(review_block) if isinstance(review_block, dict) else None
        planner_plan = dict(plan_block) if isinstance(plan_block, dict) else None

        return cls(
            schema_version=data.get("schema_version", "1.0"),
            source=data.get("source", "unknown"),
            rationale=data.get("rationale", ""),
            report_language=_flat_or_block("report_language", report_block),
            llm_model=_flat_or_block("llm_model", report_block),
            technical_temperature=_flat_or_block(
                "technical_temperature", report_block
            ),
            business_temperature=_flat_or_block(
                "business_temperature", report_block
            ),
            business_include_technical_context=_bool_or_block(
                "business_include_technical_context", report_block
            ),
            use_case=_flat_or_block("use_case", business_block),
            industry=_flat_or_block("industry", business_block),
            target_audience=_flat_or_block("target_audience", business_block),
            stakeholders=cls._coerce_list(_flat_or_block("stakeholders", business_block)),
            business_goal=_flat_or_block("business_goal", business_block),
            project_objective=_flat_or_block("project_objective", business_block),
            required_sections=cls._coerce_list(
                _flat_or_block("required_sections", report_block, instructions_block)
            ),
            technical_instructions=technical_instructions,
            business_instructions=business_instructions,
            planner_review=planner_review,
            planner_plan=planner_plan,
            extra={k: v for k, v in data.items() if k not in known_keys},
        )

    @classmethod
    def from_json_file(cls, path: Union[str, os.PathLike[str]]) -> "ReportPlannerInput":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls.from_dict(data)

    def apply_to_config(self, config: ReportGeneratorConfig) -> ReportGeneratorConfig:
        overrides: Dict[str, Any] = {}
        if self.report_language:
            overrides["report_language"] = self.report_language
        if self.llm_model:
            overrides["llm_model"] = self.llm_model
        if self.technical_temperature is not None:
            overrides["technical_temperature"] = self.technical_temperature
        if self.business_temperature is not None:
            overrides["business_temperature"] = self.business_temperature
        if self.business_include_technical_context is not None:
            overrides["business_include_technical_context"] = (
                self.business_include_technical_context
            )
        return dataclass_replace(config, **overrides) if overrides else config

    def merge_into_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(json_data)
        business_context = dict(merged.get("business_context") or {})
        planner_review = dict(merged.get("planner_review") or {})
        planner_plan = dict(merged.get("planner_plan") or {})
        report_planner = dict(merged.get("report_planner") or {})

        if self.use_case is not None:
            business_context["use_case"] = self.use_case
        if self.industry is not None:
            business_context["industry"] = self.industry
        if self.target_audience is not None:
            business_context["target_audience"] = self.target_audience
        if self.stakeholders is not None:
            business_context["stakeholders"] = self.stakeholders
        if self.business_goal is not None:
            business_context["business_goal"] = self.business_goal
        if self.project_objective is not None:
            business_context["project_objective"] = self.project_objective
        if self.report_language is not None:
            business_context["report_language"] = self.report_language

        if self.planner_review:
            planner_review.update(self.planner_review)
        if self.planner_plan:
            planner_plan.update(self.planner_plan)
        if self.rationale and "review_text" not in planner_review:
            planner_review["review_text"] = self.rationale

        report_planner.update(
            {
                "source": self.source,
                "schema_version": self.schema_version,
            }
        )
        if self.rationale:
            report_planner["rationale"] = self.rationale
        if self.required_sections is not None:
            report_planner["required_sections"] = self.required_sections
        if self.technical_instructions is not None:
            report_planner["technical_instructions"] = self.technical_instructions
        if self.business_instructions is not None:
            report_planner["business_instructions"] = self.business_instructions
        if self.extra:
            report_planner["extra"] = self.extra

        merged["business_context"] = business_context
        if planner_review:
            merged["planner_review"] = planner_review
        if planner_plan:
            merged["planner_plan"] = planner_plan
        merged["report_planner"] = report_planner
        return merged


def load_report_planner_input(path: Union[str, os.PathLike[str]]) -> ReportPlannerInput:
    planner_path = Path(path)
    if not planner_path.exists():
        raise FileNotFoundError(f"Planner input file not found: {planner_path}")
    planner_input = ReportPlannerInput.from_json_file(planner_path)
    extra = dict(planner_input.extra)
    extra.setdefault("loaded_from_path", str(planner_path.resolve()))
    return dataclass_replace(planner_input, extra=extra)


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
        planner_input: Optional[ReportPlannerInput] = None,
    ):
        default_config = ReportGeneratorConfig(
            output_dir="reports",
            llm_model=os.getenv("OPENAI_MODEL", ReportGeneratorConfig.llm_model),
            report_language=os.getenv(
                "REPORT_LANGUAGE", ReportGeneratorConfig.report_language
            ),
        )
        resolved_config = config or default_config
        self.planner_input_: Optional[ReportPlannerInput] = planner_input
        if planner_input is not None:
            resolved_config = planner_input.apply_to_config(resolved_config)
        self.config = resolved_config

        if self.config.require_llm and self.config.force_template_mode:
            raise ValueError(
                "force_template_mode and require_llm cannot both be enabled."
            )

        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.langchain_available = LANGCHAIN_AVAILABLE
        self.generation_mode = "template"
        # LLM is enabled when LangChain is available AND at least one API key exists
        _has_key = (
            bool(os.getenv("DASHSCOPE_API_KEY", ""))
            or not self._is_placeholder_key(self.api_key)
        )
        self.llm_enabled = self.langchain_available and _has_key
        self._llm_disabled_reason: Optional[str] = None
        self._mode_announced = False

        self.llm_technical = None
        self.llm_business = None
        self.technical_chain = None
        self.business_chain = None
        self.llm_backend = "template"

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
                self._llm_disabled_reason = "No valid API key detected (set DASHSCOPE_API_KEY or OPENAI_API_KEY)."
            if self.config.require_llm:
                if not self.langchain_available:
                    raise ValueError(
                        "LLM mode requires langchain-core and langchain-openai. "
                        f"Import error: {LANGCHAIN_IMPORT_ERROR}"
                    )
                raise ValueError(
                    "Set DASHSCOPE_API_KEY (Qwen) or OPENAI_API_KEY, or disable require_llm."
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
        langchain_error: Optional[Exception] = None
        technical_prompt_text = self._get_technical_report_prompt()
        business_prompt_text = self._get_business_translation_prompt()

        if self.langchain_available:
            try:
                technical_prompt = ChatPromptTemplate.from_messages(
                    [("human", technical_prompt_text)]
                )
                business_prompt = ChatPromptTemplate.from_messages(
                    [("human", business_prompt_text)]
                )

                from utils import build_chat_llm
                self.llm_technical = build_chat_llm(
                    model=self.config.llm_model,
                    temperature=self.config.technical_temperature,
                )
                self.llm_business = build_chat_llm(
                    model=self.config.llm_model,
                    temperature=self.config.business_temperature,
                )
                if self.llm_technical is None or self.llm_business is None:
                    raise RuntimeError("No LLM available (set DASHSCOPE_API_KEY or OPENAI_API_KEY).")

                self.technical_chain = (
                    technical_prompt | self.llm_technical | StrOutputParser()
                )
                self.business_chain = (
                    business_prompt | self.llm_business | StrOutputParser()
                )
                self.llm_backend = "langchain"
                return
            except Exception as exc:
                langchain_error = exc
        else:
            langchain_error = RuntimeError(
                "LangChain dependencies are unavailable "
                f"({LANGCHAIN_IMPORT_ERROR})."
            )

        if not OPENAI_COMPATIBLE_CHAT_AVAILABLE:
            raise RuntimeError(
                "Unable to initialize LLM chains. "
                f"LangChain backend failed: {langchain_error}. "
                "OpenAI-compatible fallback client is unavailable "
                f"({OPENAI_COMPATIBLE_CHAT_IMPORT_ERROR})."
            )

        base_url = os.getenv("OPENAI_BASE_URL")
        timeout_raw = os.getenv("OPENAI_TIMEOUT", "30")
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = 30.0

        self.llm_technical = OpenAICompatibleChatClient(
            model=self.config.llm_model,
            api_key=self.api_key,
            base_url=base_url,
            temperature=self.config.technical_temperature,
            timeout=timeout,
        )
        self.llm_business = OpenAICompatibleChatClient(
            model=self.config.llm_model,
            api_key=self.api_key,
            base_url=base_url,
            temperature=self.config.business_temperature,
            timeout=timeout,
        )
        self.technical_chain = DirectPromptChain(
            technical_prompt_text,
            self.llm_technical,
        )
        self.business_chain = DirectPromptChain(
            business_prompt_text,
            self.llm_business,
        )
        self.llm_backend = "openai_compatible"

    def _announce_generation_mode(self) -> None:
        if self._mode_announced:
            return
        if self.llm_enabled:
            print(
                "Using LLM-backed report generation with model "
                f"'{self.config.llm_model}' via backend '{self.llm_backend}'."
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
            normalized = self._normalize_legacy_normalized_schema(json_data)
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

        if self.planner_input_ is not None:
            normalized = self.planner_input_.merge_into_json(normalized)

        self.validate_input_json(normalized)
        return normalized

    def _normalize_legacy_normalized_schema(
        self, json_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        normalized = dict(json_data)

        meta = dict(normalized.get("meta") or {})
        data_understanding = dict(normalized.get("data_understanding") or {})
        data_cleaning = dict(normalized.get("data_cleaning") or {})
        feature_engineering = dict(normalized.get("feature_engineering") or {})
        modeling = dict(normalized.get("modeling") or {})
        evaluation = dict(normalized.get("evaluation") or {})
        business_context = dict(normalized.get("business_context") or {})
        planner_plan = dict(normalized.get("planner_plan") or {})

        understanding_result = (
            data_understanding.get("result")
            if isinstance(data_understanding.get("result"), dict)
            else {}
        )
        data_profile = understanding_result.get("data_profile", {})
        data_quality_report = understanding_result.get("data_quality_report", {})
        target_analysis = understanding_result.get("target_analysis", {})
        understanding_summary = understanding_result.get(
            "data_understanding_summary", {}
        )
        diagnostics = modeling.get("diagnostics", {})
        best_model_metrics = modeling.get("best_model_metrics", {})
        evaluation_selection = evaluation.get("best_model_selection_evidence", {})
        best_model_evaluation = evaluation.get("best_model_evaluation", {})
        best_model_eval_metrics = best_model_evaluation.get("metrics", {})
        best_model_eval_diagnostics = best_model_evaluation.get("diagnostics", {})

        if "project_theme" not in meta and meta.get("project_name"):
            meta["project_theme"] = meta.get("project_name")
        if "pipeline_id" not in meta and meta.get("project_name"):
            meta["pipeline_id"] = str(meta.get("project_name")).lower().replace(" ", "_")
        if "dataset_source" not in meta and data_understanding.get("output_dir"):
            meta["dataset_source"] = data_understanding.get("output_dir")
        if "target_variable" not in meta:
            meta["target_variable"] = self._first_non_empty(
                target_analysis.get("target_column"),
                feature_engineering.get("target_column"),
                planner_plan.get("target_column"),
            )
        if "task_type" not in meta:
            meta["task_type"] = self._first_non_empty(
                target_analysis.get("problem_type"),
                feature_engineering.get("problem_type"),
                modeling.get("problem_type"),
                evaluation.get("problem_type"),
                planner_plan.get("problem_type"),
            )
        if "project_description" not in meta:
            meta["project_description"] = self._first_non_empty(
                understanding_summary.get("executive_summary"),
                business_context.get("project_objective"),
                business_context.get("business_goal"),
            )

        if "n_rows" not in data_understanding and "total_samples" in data_understanding:
            data_understanding["n_rows"] = data_understanding.get("total_samples")
        if "n_cols" not in data_understanding and "feature_count" in data_understanding:
            data_understanding["n_cols"] = data_understanding.get("feature_count")
        if (
            "class_distribution" not in data_understanding
            and "target_distribution" in data_understanding
        ):
            data_understanding["class_distribution"] = data_understanding.get(
                "target_distribution"
            )
        if (
            "missing_values_summary" not in data_understanding
            and "missing_values" in data_understanding
        ):
            missing_values = data_understanding.get("missing_values")
            if isinstance(missing_values, dict):
                data_understanding["missing_values_summary"] = missing_values
            elif missing_values not in (None, ""):
                data_understanding["missing_values_summary"] = {
                    "total_missing_values": missing_values
                }
        if "n_rows" not in data_understanding and isinstance(data_profile.get("shape"), dict):
            data_understanding["n_rows"] = data_profile.get("shape", {}).get("rows")
        if "n_cols" not in data_understanding and isinstance(data_profile.get("shape"), dict):
            data_understanding["n_cols"] = data_profile.get("shape", {}).get("columns")
        if "feature_types" not in data_understanding and data_profile.get("feature_types"):
            data_understanding["feature_types"] = data_profile.get("feature_types")
        if (
            "class_distribution" not in data_understanding
            and target_analysis.get("class_distribution")
        ):
            data_understanding["class_distribution"] = target_analysis.get(
                "class_distribution"
            )
        if (
            "class_imbalance_ratio" not in data_understanding
            and target_analysis.get("imbalance_ratio_max_over_min") is not None
        ):
            data_understanding["class_imbalance_ratio"] = target_analysis.get(
                "imbalance_ratio_max_over_min"
            )
        if (
            "missing_values_summary" not in data_understanding
            and data_quality_report.get("missing_values")
        ):
            data_understanding["missing_values_summary"] = data_quality_report.get(
                "missing_values"
            )
        if "key_insights" not in data_understanding and understanding_summary.get(
            "major_findings"
        ):
            data_understanding["key_insights"] = understanding_summary.get(
                "major_findings"
            )

        if not data_cleaning.get("quality_notes"):
            cleaned_rows = data_cleaning.get("cleaned_rows")
            rows_removed = data_cleaning.get("rows_removed")
            retention_rate = data_cleaning.get("retention_rate")
            anomalies_removed = data_cleaning.get("anomalies_removed")
            quality_parts = []
            if cleaned_rows not in (None, ""):
                quality_parts.append(f"Rows after cleaning: {cleaned_rows}")
            if rows_removed not in (None, ""):
                quality_parts.append(f"Rows removed: {rows_removed}")
            if retention_rate not in (None, ""):
                quality_parts.append(f"Retention rate: {retention_rate}")
            if anomalies_removed not in (None, ""):
                quality_parts.append(
                    f"Anomaly handling summary: {anomalies_removed}"
                )
            if quality_parts:
                data_cleaning["quality_notes"] = ". ".join(quality_parts) + "."

        if (
            "final_feature_count" not in feature_engineering
            and "engineered_features" in feature_engineering
        ):
            feature_engineering["final_feature_count"] = feature_engineering.get(
                "engineered_features"
            )
        if (
            "features_created" not in feature_engineering
            and feature_engineering.get("final_feature_count") not in (None, "")
        ):
            feature_engineering["features_created"] = [
                f"Final feature count: {feature_engineering.get('final_feature_count')}"
            ]
        if (
            "features_created" not in feature_engineering
            and "engineered_features" in feature_engineering
        ):
            feature_engineering["features_created"] = [
                f"Engineered features count: {feature_engineering.get('engineered_features')}"
            ]
        if (
            "features_created" not in feature_engineering
            and feature_engineering.get("llm_actions_applied")
        ):
            feature_engineering["features_created"] = feature_engineering.get(
                "llm_actions_applied"
            )
        if (
            "features_dropped" not in feature_engineering
            and isinstance(feature_engineering.get("dropped_columns"), dict)
        ):
            feature_engineering["features_dropped"] = feature_engineering.get(
                "dropped_columns", {}
            ).get("general_drop", [])
        if (
            "encoding_applied" not in feature_engineering
            and feature_engineering.get("used_columns")
        ):
            feature_engineering["encoding_applied"] = feature_engineering.get(
                "used_columns"
            )
        if (
            "feature_importances" not in feature_engineering
            and modeling.get("best_model_feature_importance")
        ):
            feature_engineering["feature_importances"] = [
                {
                    "feature": item.get("feature_name", item.get("feature")),
                    "importance": item.get("importance"),
                }
                for item in self._coerce_list(
                    modeling.get("best_model_feature_importance")
                )
                if isinstance(item, dict)
            ]
        if "key_insights" not in feature_engineering and feature_engineering.get(
            "llm_actions_count"
        ) is not None:
            feature_engineering["key_insights"] = [
                f"LLM feature actions applied: {feature_engineering.get('llm_actions_count')}"
            ]

        best_model = modeling.get("best_model")
        if isinstance(best_model, str):
            best_model_name = self._first_non_empty(
                modeling.get("best_model_name"),
                evaluation.get("best_model_name"),
            )
            if best_model_name not in (None, ""):
                modeling["best_model"] = {"name": best_model_name}
            else:
                modeling["best_model"] = {"name": best_model}
        elif best_model in (None, ""):
            best_model_name = self._first_non_empty(
                modeling.get("best_model_name"),
                evaluation.get("best_model"),
            )
            modeling["best_model"] = (
                {"name": best_model_name} if best_model_name not in (None, "") else {}
            )
        elif isinstance(best_model, dict) and "name" not in best_model:
            best_model_name = self._first_non_empty(
                modeling.get("best_model_name"),
                evaluation.get("best_model_name"),
            )
            if best_model_name not in (None, ""):
                modeling["best_model"]["name"] = best_model_name

        if not modeling.get("models_compared") and modeling.get("leaderboard"):
            modeling["models_compared"] = modeling.get("leaderboard")
        if not modeling.get("selection_reason") and evaluation_selection:
            modeling["selection_reason"] = (
                f"Selected by {evaluation_selection.get('selection_metric')} with score "
                f"{evaluation_selection.get('selection_metric_value')} at rank "
                f"{evaluation_selection.get('selection_rank')}."
            )

        if "primary_metric" not in evaluation:
            evaluation["primary_metric"] = self._first_non_empty(
                modeling.get("primary_metric"),
                planner_plan.get("primary_metric"),
            )
        if "primary_score" not in evaluation and evaluation_selection.get(
            "selection_metric_value"
        ) is not None:
            evaluation["primary_score"] = evaluation_selection.get(
                "selection_metric_value"
            )
        if not evaluation.get("metrics"):
            evaluation["metrics"] = self._first_non_empty(
                best_model_eval_metrics,
                best_model_metrics,
                {},
            )
        if not evaluation.get("confusion_matrix"):
            evaluation["confusion_matrix"] = self._first_non_empty(
                best_model_eval_diagnostics.get("confusion_matrix"),
                diagnostics.get("confusion_matrix"),
                {},
            )
        if not evaluation.get("key_insights") and evaluation.get("limitations") is not None:
            evaluation["key_insights"] = evaluation.get("limitations")

        if meta.get("models_evaluated") in (None, "") and modeling.get(
            "models_trained"
        ) not in (None, ""):
            meta["models_evaluated"] = modeling.get("models_trained")
        if meta.get("models_evaluated") in (None, "") and evaluation.get(
            "benchmark_overview", {}
        ).get("candidate_model_count") is not None:
            meta["models_evaluated"] = evaluation.get("benchmark_overview", {}).get(
                "candidate_model_count"
            )

        normalized["meta"] = meta
        normalized["data_understanding"] = data_understanding
        normalized["data_cleaning"] = data_cleaning
        normalized["feature_engineering"] = feature_engineering
        normalized["modeling"] = modeling
        normalized["evaluation"] = evaluation
        normalized["business_context"] = business_context
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
        planner_review = json_data.get("planner_review", {})
        planner_plan = json_data.get("planner_plan", {})
        report_planner = json_data.get("report_planner", {})

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
            self._format_optional_context("Planner source", report_planner.get("source")),
            self._format_optional_context(
                "Planner rationale", report_planner.get("rationale")
            ),
            self._format_optional_context(
                "Planner required sections",
                report_planner.get("required_sections"),
            ),
            self._format_optional_context(
                "Planner review", planner_review.get("review_text")
            ),
            self._format_optional_context(
                "Planner primary metric", planner_plan.get("primary_metric")
            ),
            self._format_optional_context(
                "Planner reasoning", planner_plan.get("reasoning")
            ),
        ]
        usable_lines = [line for line in context_lines if line]
        if not usable_lines:
            return (
                "- No extra project context was provided. Infer the topic strictly "
                "from the JSON content."
            )
        return "\n".join(usable_lines)

    def _build_planner_instruction_block(
        self, json_data: Dict[str, Any], audience: str
    ) -> str:
        report_planner = json_data.get("report_planner", {})
        required_sections = self._coerce_list(report_planner.get("required_sections"))
        instruction_items = self._coerce_list(
            report_planner.get(f"{audience}_instructions")
        )

        lines = []
        if required_sections:
            joined_sections = ", ".join(str(item) for item in required_sections)
            lines.append(f"- Required sections: {joined_sections}")
        lines.extend(
            f"- {str(item).strip()}"
            for item in instruction_items
            if str(item).strip()
        )
        if not lines:
            return "- No additional planner instructions were provided for this report."
        return "\n".join(lines)

    def _build_prompt_payload(self, json_data: Dict[str, Any]) -> Dict[str, str]:
        report_language_name = self._describe_report_language(
            self._resolve_report_language(json_data)
        )
        return {
            "json_data": json.dumps(json_data, ensure_ascii=False, indent=2),
            "project_context": self._build_project_context(json_data),
            "technical_planner_instructions": self._build_planner_instruction_block(
                json_data, "technical"
            ),
            "business_planner_instructions": self._build_planner_instruction_block(
                json_data, "business"
            ),
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
        confusion_matrix = self._normalize_confusion_matrix(confusion_matrix)
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
        planner_review = json_data.get("planner_review", {})
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
        recommendations.extend(
            str(item)
            for item in self._coerce_list(planner_review.get("recommendations"))
            if str(item).strip()
        )
        return self._deduplicate_items(recommendations)

    @staticmethod
    def _deduplicate_items(items: list[str]) -> list[str]:
        seen = set()
        deduplicated = []
        for item in items:
            normalized = str(item).strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(normalized)
        return deduplicated

    def _build_planner_highlights(self, json_data: Dict[str, Any]) -> list[str]:
        planner_review = json_data.get("planner_review", {})
        planner_plan = json_data.get("planner_plan", {})
        report_planner = json_data.get("report_planner", {})

        highlights = [
            str(item)
            for item in self._coerce_list(planner_review.get("key_findings"))
            if str(item).strip()
        ]

        primary_metric = planner_plan.get("primary_metric")
        if primary_metric:
            highlights.append(f"Planner primary metric: {primary_metric}.")

        reasoning = planner_plan.get("reasoning")
        if reasoning:
            highlights.append(f"Planner reasoning: {reasoning}")

        rationale = report_planner.get("rationale")
        if rationale:
            highlights.append(f"Planner rationale: {rationale}")

        required_sections = self._coerce_list(report_planner.get("required_sections"))
        if required_sections:
            highlights.append(
                "Planner requested sections: "
                + ", ".join(str(item) for item in required_sections)
                + "."
            )

        return self._deduplicate_items(highlights)

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
        best_model_raw = modeling.get("best_model", {})
        best_model = (
            best_model_raw
            if isinstance(best_model_raw, dict)
            else {"name": best_model_raw}
            if best_model_raw not in (None, "")
            else {}
        )
        best_model_name = best_model.get("name", "N/A")
        primary_metric = evaluation.get("primary_metric", "primary metric")
        primary_score = evaluation.get("primary_score")
        top_features = self._top_feature_summaries(
            self._coerce_list(feature_engineering.get("feature_importances"))
        )
        planner_highlights = self._build_planner_highlights(json_data)
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
            self._markdown_bullets(
                planner_highlights
                or ["No planner-specific findings were provided."]
            ),
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
        planner_highlights = self._build_planner_highlights(json_data)
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
        key_findings.extend(planner_highlights)

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

    @staticmethod
    def _sanitize_report_markdown(report: str) -> str:
        """Remove trailing assistant-style follow-up offers from report output."""
        normalized = report.rstrip()
        if not normalized:
            return report

        conversational_starters = (
            "if you want,",
            "if you'd like,",
            "if you would like,",
            "i can ",
            "let me know if you want",
            "let me know if you'd like",
        )

        paragraphs = normalized.split("\n\n")
        while paragraphs:
            last_paragraph = paragraphs[-1].strip()
            if not last_paragraph:
                paragraphs.pop()
                continue
            lowered = last_paragraph.casefold()
            if any(lowered.startswith(prefix) for prefix in conversational_starters):
                paragraphs.pop()
                continue
            break

        cleaned = "\n\n".join(paragraphs).strip()
        return (cleaned + "\n") if cleaned else ""

    def _generate_technical_report_content(self, json_data: Dict[str, Any]) -> str:
        payload = self._build_prompt_payload(json_data)
        report_language_name = payload["report_language_name"]

        if self.technical_chain is None:
            return self._sanitize_report_markdown(
                self._build_template_technical_report(json_data, report_language_name)
            )

        try:
            return self._sanitize_report_markdown(self.technical_chain.invoke(payload))
        except Exception as exc:
            if self.config.require_llm:
                raise
            print(
                f"Technical LLM generation failed: {exc}. Falling back to template mode."
            )
            return self._sanitize_report_markdown(
                self._build_template_technical_report(json_data, report_language_name)
            )

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
            return self._sanitize_report_markdown(
                self._build_template_business_report(
                    json_data,
                    report_language_name,
                    technical_report=technical_report,
                )
            )

        try:
            return self._sanitize_report_markdown(self.business_chain.invoke(payload))
        except Exception as exc:
            if self.config.require_llm:
                raise
            print(
                f"Business LLM generation failed: {exc}. Falling back to template mode."
            )
            return self._sanitize_report_markdown(
                self._build_template_business_report(
                    json_data,
                    report_language_name,
                    technical_report=technical_report,
                )
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

        with open(technical_path, "w", encoding="utf-8") as file:
            file.write(technical_report)
        with open(business_path, "w", encoding="utf-8") as file:
            file.write(business_report)

        print(f"Saved technical report to: {technical_path}")
        print(f"Saved business report to: {business_path}")
        return {
            "technical_report": str(technical_path),
            "business_report": str(business_path),
            "llm_backend": self.llm_backend,
        }

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
                "llm_backend": self.llm_backend,
                "planner_input": asdict(self.planner_input_)
                if self.planner_input_ is not None
                else None,
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
                "llm_backend": self.llm_backend,
                "planner_input": asdict(self.planner_input_)
                if self.planner_input_ is not None
                else None,
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
                "llm_backend": self.llm_backend,
                "planner_input": asdict(self.planner_input_)
                if self.planner_input_ is not None
                else None,
            }
        except Exception as exc:
            return {
                "status": "failure",
                "technical_report": None,
                "saved_paths": None,
                "agent_name": self.AGENT_NAME,
                "generation_mode": self.generation_mode,
                "llm_backend": self.llm_backend,
                "planner_input": asdict(self.planner_input_)
                if self.planner_input_ is not None
                else None,
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
                "llm_backend": self.llm_backend,
                "planner_input": asdict(self.planner_input_)
                if self.planner_input_ is not None
                else None,
                "business_used_technical_context": use_technical_context,
            }
        except Exception as exc:
            return {
                "status": "failure",
                "business_report": None,
                "saved_paths": None,
                "agent_name": self.AGENT_NAME,
                "generation_mode": self.generation_mode,
                "llm_backend": self.llm_backend,
                "planner_input": asdict(self.planner_input_)
                if self.planner_input_ is not None
                else None,
                "business_used_technical_context": False,
                "error": str(exc),
            }

    def _get_technical_report_prompt(self) -> str:
        return """You are a senior data scientist with more than 10 years of machine learning project experience. Your job is to write a professional technical report from structured project results.

Generate a technical report based on the project context and JSON data below.

[Project context]
{project_context}

[Planner instructions for the technical report]
{technical_planner_instructions}

[JSON data]
{json_data}

[Report requirements]

0. Topic adaptation:
   - Infer the project theme, industry, and task strictly from the project context and JSON content.
   - Do not assume any specific domain, user group, or use case unless the JSON clearly says so.
   - If `planner_review`, `planner_plan`, or `report_planner` exists, use them as steering context, but let concrete metrics and field values in the JSON take priority when there is any conflict.
   - For classification tasks, explain metrics such as accuracy, F1, recall, precision, class imbalance, and the confusion matrix.
   - For regression tasks, focus on MAE, RMSE, R2, error range, and business impact.
   - For other task types, prioritize the core metrics that actually appear in the JSON instead of forcing a fixed template.
   - Use the entity names from the JSON for the target, risk group, and business objects.
   - Follow any required sections or report-specific instructions in the planner block when they do not conflict with the actual JSON evidence.

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
   - Do not use stock sentences that could apply to any project; every key finding and recommendation must map back to the provided JSON.
   - Do not ask the reader follow-up questions.
   - Do not offer extra help, future optional work, or next-step services outside the report itself.
   - Do not use conversational phrases such as `If you want`, `I can`, `let me know`, or `once you confirm`.

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
End the report immediately after the final required section.
"""

    def _get_business_translation_prompt(self) -> str:
        return """You are a senior business analyst who is skilled at turning data science results into action plans that business teams can execute directly.

Translate the technical analysis into practical business recommendations based on the project context and JSON data.

[Project context]
{project_context}

[Planner instructions for the business report]
{business_planner_instructions}

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
   - If `planner_review`, `planner_plan`, or `report_planner` exists, use them as business-steering context, but prefer concrete metrics and explicit field values from the JSON over generic planner wording when they differ.
   - Treat the optional technical report context as supplemental input only, not as a required dependency.
   - If the optional technical report context is empty or unavailable, infer the business recommendations directly from the JSON and project context.
   - Follow any required sections or business-specific instructions in the planner block when they do not conflict with the actual JSON evidence.

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
   - Avoid canned recommendations; tie each action to the actual risks, metrics, and constraints in the JSON.
   - Do not ask the reader follow-up questions.
   - Do not offer extra help, future optional work, or next-step services outside the report itself.
   - Do not use conversational phrases such as `If you want`, `I can`, `let me know`, or `once you confirm`.

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
End the report immediately after the final required section.
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
    parser.add_argument(
        "--planner-input",
        type=str,
        default=None,
        help="Optional planner JSON that steers report strategy, sections, and business context.",
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
    planner_input = (
        load_report_planner_input(args.planner_input)
        if args.planner_input
        else None
    )
    generator = MultiAgentReportGenerator(
        config=config,
        planner_input=planner_input,
    )

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
    "ReportPlannerInput",
    "ReportGeneratorConfig",
    "ReportGenerator",
    "MultiAgentReportGenerator",
    "load_report_planner_input",
]
