"""
AutoDS Planner Agent

Responsibilities:
  1. plan()                       -- Stage 0: parse business description,
                                     generate all downstream Agent configs
  2. replan_after_understanding() -- Stage 1→2: adjust configs based on
                                     DataUnderstandingAgent output
  3. review_modelling()           -- Stage 5→6: natural-language review of
                                     model results for the final report

LLM backend priority:
  Anthropic Claude (ANTHROPIC_API_KEY)  →  OpenAI (OPENAI_API_KEY)  →  rule-based fallback
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    llm_model_anthropic: str = "claude-sonnet-4-6"
    llm_model_openai: str = "gpt-4o-mini"
    temperature: float = 0.0
    output_dir: str = "./autods_pipeline_output/00_planning"
    use_adaptive_replanning: bool = True
    data_sample_rows: int = 5          # rows shown to the LLM


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """
    Pipeline Planner Agent — coordinates all other agents via LLM-generated configs.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self._client = None
        self._provider: Optional[str] = None
        self._load_dotenv()
        self._init_llm()

    # ------------------------------------------------------------------
    # LLM initialisation
    # ------------------------------------------------------------------

    def _load_dotenv(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    def _init_llm(self):
        # Try Anthropic first
        if self._try_anthropic():
            return
        # Fallback to OpenAI
        if self._try_openai():
            return
        print("[PlannerAgent] No LLM available — rule-based fallback mode.")

    def _try_anthropic(self) -> bool:
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key or api_key.startswith("sk-ant-placeholder"):
                return False
            self._client = anthropic.Anthropic(api_key=api_key)
            self._provider = "anthropic"
            print(f"[PlannerAgent] Using Anthropic Claude ({self.config.llm_model_anthropic})")
            return True
        except ImportError:
            return False

    def _try_openai(self) -> bool:
        try:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key or "placeholder" in api_key.lower():
                return False
            self._client = ChatOpenAI(
                model=self.config.llm_model_openai,
                temperature=self.config.temperature,
                api_key=api_key,
            )
            self._provider = "openai"
            print(f"[PlannerAgent] Using OpenAI ({self.config.llm_model_openai})")
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Unified LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        if self._client is None:
            return ""
        try:
            if self._provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.config.llm_model_anthropic,
                    max_tokens=2048,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text

            if self._provider == "openai":
                from langchain_core.messages import HumanMessage, SystemMessage
                resp = self._client.invoke(
                    [SystemMessage(content=system), HumanMessage(content=user)]
                )
                return resp.content

        except Exception as exc:
            print(f"[PlannerAgent] LLM call failed: {exc}")
        return ""

    @staticmethod
    def _parse_json(text: str) -> Dict:
        """Extract the first valid JSON object from arbitrary LLM output."""
        for attempt in (
            lambda t: json.loads(t),
            lambda t: json.loads(re.search(r"```(?:json)?\s*([\s\S]*?)```", t).group(1).strip()),
            lambda t: json.loads(re.search(r"\{[\s\S]*\}", t).group()),
        ):
            try:
                return attempt(text)
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # 1. plan() — pre-run planning
    # ------------------------------------------------------------------

    def plan(
        self,
        business_description: str,
        data_sample: pd.DataFrame,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Stage 0 — generate pipeline configs from business description + data sample.

        Returns a dict with keys:
          target_column, problem_type, primary_metric,
          feature_config   (kwargs for FeatureEngineeringConfig),
          modelling_config (kwargs for ModellingConfig),
          reasoning
        """
        print("\n" + "=" * 60)
        print("STAGE 0: PLANNER — Pre-run Planning")
        print("=" * 60)

        schema = self._describe_schema(data_sample)

        if self._client is not None:
            plan = self._llm_plan(business_description, schema, constraints)
        else:
            plan = self._rule_based_plan(constraints)

        self._save("initial_plan.json", plan)
        self._print_plan_summary(plan)
        return plan

    def _describe_schema(self, df: pd.DataFrame) -> str:
        lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", "Columns:"]
        for col in df.columns:
            n_unique = df[col].nunique()
            samples = df[col].dropna().head(3).tolist()
            lines.append(
                f"  - {col}  dtype={df[col].dtype}  unique={n_unique}  samples={samples}"
            )
        return "\n".join(lines)

    _PLAN_SYSTEM = """\
You are an expert data science project planner.
Given a business objective and dataset schema, output a JSON configuration for an \
automated ML pipeline.

Return ONLY a valid JSON object — no extra text, no markdown fences outside the JSON.

Schema:
{
  "target_column": "<exact column name from schema>",
  "problem_type": "<classification | regression>",
  "primary_metric": "<roc_auc | f1 | accuracy | rmse | r2>",
  "feature_config": {
    "task_description": "<one-sentence description for feature engineering>",
    "use_llm_planner": <true | false>
  },
  "modelling_config": {
    "candidate_model_names": ["<model1>", "<model2>", "<model3>"],
    "cv_folds": <3–10>,
    "primary_metric": "<same as above>"
  },
  "reasoning": "<1-2 sentences explaining key choices>"
}

Available models:
  Classification : LogisticRegression, DecisionTree, RandomForest,
                   GradientBoosting, LightGBM, XGBoost, SVC, KNeighbors
  Regression     : LinearRegression, Ridge, Lasso, DecisionTree,
                   RandomForest, GradientBoosting, LightGBM, XGBoost, SVR

Rules:
- Pick the target column from the schema; if unclear, pick the most likely label column.
- Binary 0/1 target → classification, primary_metric=roc_auc.
- Continuous numeric target → regression, primary_metric=rmse.
- If interpretability is required, prefer LogisticRegression / DecisionTree.
- Select 3–5 candidate models appropriate for the problem size and type.
"""

    def _llm_plan(
        self,
        business_description: str,
        schema: str,
        constraints: Optional[Dict],
    ) -> Dict[str, Any]:
        constraint_text = (
            f"\nConstraints: {json.dumps(constraints, ensure_ascii=False)}"
            if constraints
            else ""
        )
        user = (
            f"Business objective:\n{business_description}\n\n"
            f"Dataset schema:\n{schema}"
            f"{constraint_text}\n\n"
            "Generate the pipeline configuration."
        )
        raw = self._call_llm(self._PLAN_SYSTEM, user)
        plan = self._parse_json(raw)
        if not plan or "target_column" not in plan:
            print("[PlannerAgent] LLM returned invalid plan — using rule-based fallback.")
            return self._rule_based_plan(constraints)
        return plan

    @staticmethod
    def _rule_based_plan(constraints: Optional[Dict]) -> Dict[str, Any]:
        return {
            "target_column": None,
            "problem_type": "classification",
            "primary_metric": "roc_auc",
            "feature_config": {
                "task_description": "AutoDS automated feature engineering",
                "use_llm_planner": False,
            },
            "modelling_config": {
                "candidate_model_names": [
                    "LogisticRegression", "RandomForest", "GradientBoosting"
                ],
                "cv_folds": 5,
                "primary_metric": "roc_auc",
            },
            "reasoning": "Rule-based fallback (no LLM available).",
        }

    @staticmethod
    def _print_plan_summary(plan: Dict[str, Any]) -> None:
        print(f"  target_column  : {plan.get('target_column')}")
        print(f"  problem_type   : {plan.get('problem_type')}")
        print(f"  primary_metric : {plan.get('primary_metric')}")
        models = plan.get("modelling_config", {}).get("candidate_model_names", [])
        print(f"  candidate_models: {models}")
        print(f"  reasoning      : {plan.get('reasoning', '')}")

    # ------------------------------------------------------------------
    # 2. replan_after_understanding() — adaptive replanning
    # ------------------------------------------------------------------

    def replan_after_understanding(
        self,
        understanding_output: Dict[str, Any],
        current_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Stage 1 → adaptive — inspect DataUnderstandingAgent output and adjust
        feature / modelling configs.

        Returns:
          {
            "adjustments": ["<description of each change>", ...],
            "updated_plan": { <partial override of current_plan> }
          }
        """
        if not self.config.use_adaptive_replanning:
            return {"adjustments": [], "updated_plan": {}}

        print("\n[PlannerAgent] Adaptive replanning after DataUnderstanding...")

        if self._client is not None:
            result = self._llm_replan(understanding_output, current_plan)
        else:
            result = self._rule_based_replan(understanding_output, current_plan)

        for adj in result.get("adjustments", []):
            print(f"  [adjustment] {adj}")

        self._save("replan_after_understanding.json", result)
        return result

    _REPLAN_SYSTEM = """\
You are reviewing a data understanding report for an automated ML pipeline.
Based on the data characteristics, decide if any pipeline configurations need adjustment.

Return ONLY a valid JSON object:
{
  "adjustments": ["<short description of each change>"],
  "updated_plan": {
    "primary_metric": "<optionally updated metric>",
    "feature_config": { <only changed keys> },
    "modelling_config": { <only changed keys> }
  }
}

If nothing needs to change, return: {"adjustments": [], "updated_plan": {}}

Common rules:
- Class imbalance ratio > 10:1  → switch primary_metric to "f1"
- Overall missing rate > 30%    → set feature_config.use_llm_planner = false
- Fewer than 200 samples        → set modelling_config.cv_folds to 3,
                                   prefer simpler models
"""

    def _llm_replan(
        self,
        understanding_output: Dict[str, Any],
        current_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = self._summarise_understanding(understanding_output)
        user = (
            f"Current plan:\n{json.dumps(current_plan, indent=2, ensure_ascii=False)}\n\n"
            f"Data understanding summary:\n{summary}\n\n"
            "Identify issues and update the plan accordingly."
        )
        raw = self._call_llm(self._REPLAN_SYSTEM, user)
        result = self._parse_json(raw)
        return result if result else {"adjustments": [], "updated_plan": {}}

    @staticmethod
    def _rule_based_replan(
        understanding_output: Dict[str, Any],
        current_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        adjustments: List[str] = []
        updated: Dict[str, Any] = {}

        # ── Class imbalance ──────────────────────────────────────────
        target_analysis = understanding_output.get("target_analysis", {})
        dist = target_analysis.get("class_distribution", {})
        if dist:
            counts = [v for v in dist.values() if isinstance(v, (int, float))]
            if counts and max(counts) / (sum(counts) + 1e-9) > 0.9:
                adjustments.append(
                    "High class imbalance detected — switching primary_metric to f1."
                )
                updated.setdefault("modelling_config", {})["primary_metric"] = "f1"
                updated["primary_metric"] = "f1"

        # ── Missing rate ─────────────────────────────────────────────
        quality = understanding_output.get("data_quality", {})
        missing_rate = quality.get("missing_rate_overall", 0)
        if isinstance(missing_rate, (int, float)) and missing_rate > 0.30:
            adjustments.append(
                f"High missing rate ({missing_rate:.1%}) — disabling LLM feature planner."
            )
            updated.setdefault("feature_config", {})["use_llm_planner"] = False

        # ── Small dataset ────────────────────────────────────────────
        n_samples = understanding_output.get("total_samples", 0)
        if isinstance(n_samples, int) and 0 < n_samples < 200:
            adjustments.append(
                f"Small dataset ({n_samples} rows) — reducing cv_folds to 3."
            )
            updated.setdefault("modelling_config", {})["cv_folds"] = 3

        return {"adjustments": adjustments, "updated_plan": updated}

    @staticmethod
    def _summarise_understanding(out: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"Total samples : {out.get('total_samples', 'N/A')}")
        lines.append(f"Feature count : {out.get('feature_count', 'N/A')}")

        quality = out.get("data_quality", {})
        if quality:
            lines.append(f"Missing rate  : {quality.get('missing_rate_overall', 'N/A')}")
            lines.append(f"Duplicate rows: {quality.get('duplicate_rows', 'N/A')}")

        target = out.get("target_analysis", {})
        if target:
            dist = target.get("class_distribution") or target.get("target_stats")
            lines.append(f"Target dist.  : {dist}")
            lines.append(f"Class imbalance: {target.get('class_imbalance', 'N/A')}")

        return "\n".join(lines) if lines else json.dumps(out, default=str)[:800]

    # ------------------------------------------------------------------
    # 3. review_modelling() — post-modelling natural-language review
    # ------------------------------------------------------------------

    def review_modelling(
        self,
        modelling_output: Dict[str, Any],
        evaluation_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Post-modelling review.

        Returns:
          {
            "key_findings"   : ["..."],
            "recommendations": ["..."],
            "review_text"    : "<2-4 sentence summary for the business report>"
          }
        """
        print("\n[PlannerAgent] Post-modelling review...")

        if self._client is not None:
            result = self._llm_review(modelling_output, evaluation_output)
        else:
            result = self._rule_based_review(modelling_output, evaluation_output)

        self._save("modelling_review.json", result)
        return result

    _REVIEW_SYSTEM = """\
You are a senior data scientist reviewing model training results for a business report.

Return ONLY a valid JSON object:
{
  "key_findings"   : ["<finding 1>", "<finding 2>", ...],
  "recommendations": ["<recommendation 1>", ...],
  "review_text"    : "<2-4 professional sentences summarising the modelling outcomes>"
}
"""

    def _llm_review(
        self,
        modelling_output: Dict[str, Any],
        evaluation_output: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary = self._summarise_modelling(modelling_output, evaluation_output)
        user = f"Modelling summary:\n{summary}\n\nProvide your review."
        raw = self._call_llm(self._REVIEW_SYSTEM, user)
        result = self._parse_json(raw)
        return result if result else self._rule_based_review(modelling_output, evaluation_output)

    @staticmethod
    def _rule_based_review(
        modelling_output: Dict[str, Any],
        evaluation_output: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        best = modelling_output.get("best_model_name", "Unknown")
        metric = modelling_output.get("primary_metric", "primary metric")
        n_models = modelling_output.get("model_count", "N/A")
        return {
            "key_findings": [
                f"Best model: {best}",
                f"Evaluated {n_models} candidate model(s)",
            ],
            "recommendations": [
                "Monitor model performance on production data.",
                "Retrain periodically as new data arrives.",
            ],
            "review_text": (
                f"The modelling stage evaluated {n_models} candidate model(s). "
                f"{best} achieved the best performance on the {metric} metric. "
                "The model is ready for domain expert validation before deployment."
            ),
        }

    @staticmethod
    def _summarise_modelling(
        modelling_output: Dict[str, Any],
        evaluation_output: Optional[Dict[str, Any]],
    ) -> str:
        lines: List[str] = [
            f"Best model    : {modelling_output.get('best_model_name', 'N/A')}",
            f"Primary metric: {modelling_output.get('primary_metric', 'N/A')}",
            f"Models trained: {modelling_output.get('model_count', 'N/A')}",
        ]
        leaderboard = modelling_output.get("leaderboard")
        if leaderboard is not None:
            try:
                if isinstance(leaderboard, pd.DataFrame):
                    lines.append("Leaderboard (top 3):\n" + leaderboard.head(3).to_string(index=False))
                elif isinstance(leaderboard, list) and leaderboard:
                    lines.append(f"Leaderboard top entry: {leaderboard[0]}")
            except Exception:
                pass
        if evaluation_output:
            lines.append(
                f"Evaluation best model: {evaluation_output.get('best_model_name', 'N/A')}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, filename: str, data: Dict[str, Any]) -> None:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / filename, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
