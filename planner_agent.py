"""
AutoDS Planner Agent

Responsibilities:
  1. plan()                       -- Stage 0: parse business description,
                                     generate all downstream Agent configs
  2. replan_after_understanding() -- Stage 1→2: adjust configs based on
                                     DataUnderstandingAgent output
  3. review_modelling()           -- Stage 5→6: natural-language review of
                                     model results for the final report

LLM backend:
  OpenAI (OPENAI_API_KEY)  →  rule-based fallback
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
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
        from utils import build_chat_llm
        client = build_chat_llm(
            model=self.config.llm_model_openai,
            temperature=self.config.temperature,
        )
        if client is not None:
            self._client = client
            self._provider = "openai"
        else:
            print("[PlannerAgent] No LLM available — rule-based fallback mode.")

    # ------------------------------------------------------------------
    # Unified LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        if self._client is None:
            return ""
        try:
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
        extra_files: Optional[List[Union[str, Path]]] = None,
    ) -> Dict[str, Any]:
        """
        Stage 0 — generate pipeline configs from business description + data sample.

        Parameters
        ----------
        business_description : str
            Plain-English description of the business objective.
        data_sample : pd.DataFrame
            A representative sample of the dataset (used for schema inference).
        constraints : dict, optional
            Hard constraints to pass directly to the LLM (e.g. preferred models).
        extra_files : list of str or Path, optional
            Additional reference files to include in the planning context.
            Supported formats: .json, .xlsx, .xls, .csv, .txt, .md
            These are parsed and summarised as supplementary context for the LLM.

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

        # Parse any supplementary files provided by the caller
        extra_context = self._parse_extra_files(extra_files or [])
        if extra_context:
            print(f"[PlannerAgent] Loaded {len(extra_files)} extra file(s) as planning context.")

        # Normalise the business description before planning.
        # This step interprets informal / non-technical / non-English input and
        # converts it into a structured English task description that the planner
        # LLM can reliably act on.
        normalised = self._normalize_input(business_description, schema)

        if self._client is not None:
            plan = self._llm_plan(normalised, schema, constraints, extra_context)
        else:
            plan = self._rule_based_plan(constraints)

        # Persist both the raw and normalised descriptions for traceability
        plan["raw_business_description"] = business_description
        plan["normalised_description"] = normalised

        self._save("initial_plan.json", plan)
        self._print_plan_summary(plan)
        return plan

    # ------------------------------------------------------------------
    # 1a. Input normalisation — interpret free-form user descriptions
    # ------------------------------------------------------------------

    _NORMALIZE_SYSTEM = """\
You are a data science project analyst.

Your job is to interpret a user's business description, which may be:
- Written in any language (Chinese, English, French, etc.)
- Informal or conversational
- Using non-technical or domain-specific terminology
- Vague or incomplete

Given the user's raw description AND the dataset column schema, produce a concise,
structured English task description that a downstream ML planner can act on.

Return ONLY a plain-English paragraph (no JSON, no bullet points, no markdown).
The paragraph must include:
1. The prediction goal (what is being predicted and why)
2. The likely target column name (inferred from the schema if not explicitly stated)
3. The problem type (classification or regression)
4. Any relevant domain constraints or priorities mentioned by the user
   (e.g. interpretability, speed, fairness)

If the input is already clear and technical, return it as-is with minor corrections only.
Do NOT invent information that is not implied by the user's description or the schema.
"""

    def _normalize_input(self, raw_description: str, schema: str) -> str:
        """
        Interpret and standardise the user's free-form business description.

        Uses a dedicated LLM call to translate informal, multilingual, or
        non-technical input into a structured English task description before
        the main planning step.  Falls back to the original text when no LLM
        is available.

        Parameters
        ----------
        raw_description : str
            The original user-supplied text (any language, any style).
        schema : str
            Dataset column schema string used to help the LLM infer the target.

        Returns
        -------
        str
            Normalised English description, or the original text on fallback.
        """
        if self._client is None:
            # No LLM — pass the raw description through unchanged
            return raw_description

        user = (
            f"User's raw description:\n{raw_description}\n\n"
            f"Dataset schema (for context):\n{schema}\n\n"
            "Produce the standardised task description."
        )

        normalised = self._call_llm(self._NORMALIZE_SYSTEM, user).strip()

        if not normalised:
            # LLM returned nothing — fall back to original
            return raw_description

        print(f"[PlannerAgent] Input normalised:\n  {normalised[:200]}{'...' if len(normalised) > 200 else ''}")
        return normalised

    # ------------------------------------------------------------------
    # 1b. Data preparation — convert raw files to a trainable DataFrame
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        data_file: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        output_csv: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Load a raw data file into a DataFrame, optionally guided by a metadata
        file, and optionally save the result as a clean CSV.

        This method is the entry point for datasets that are not already in a
        pipeline-ready CSV format.  It handles format conversion, column naming,
        and value mapping in one step.

        Supported data formats
        ----------------------
        .csv / .tsv / .txt  Delimited text (separator auto-detected)
        .xlsx / .xls        Excel workbook (first sheet used)
        .json               JSON array of records or dict-of-lists

        Supported metadata formats
        --------------------------
        .json   Data dictionary: {"columns": [...], "value_mappings": {...}}
        .csv    Column list: first column is treated as column names
        .txt    Plain column list: one name per line (UCI .names style)

        Parameters
        ----------
        data_file : str or Path
            Path to the raw data file.
        metadata_file : str or Path, optional
            Path to a metadata / data-dictionary file describing the data.
        output_csv : str or Path, optional
            If given, the resulting DataFrame is written to this path as CSV
            so that the rest of the pipeline can read it with pd.read_csv().

        Returns
        -------
        pd.DataFrame
        """
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"[PlannerAgent] Loading data file: {data_path.name}")

        # ── Step 1: load the raw file ─────────────────────────────────
        df = self._load_raw_file(data_path)
        print(f"[PlannerAgent] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # ── Step 2: apply metadata if provided ───────────────────────
        if metadata_file is not None:
            meta_path = Path(metadata_file)
            if not meta_path.exists():
                print(f"[PlannerAgent] Metadata file not found, skipping: {meta_path}")
            else:
                meta = self._parse_metadata(meta_path)
                df = self._apply_metadata(df, meta)
                print(f"[PlannerAgent] Metadata applied from: {meta_path.name}")

        # ── Step 3: save as CSV if requested ─────────────────────────
        if output_csv is not None:
            out = Path(output_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"[PlannerAgent] Prepared data saved to: {out}")

        return df

    @staticmethod
    def _detect_separator(filepath: Union[str, Path]) -> str:
        """
        Sniff the column separator from the first few lines of a text file.

        Tries comma, semicolon, tab, and pipe in that order.  The separator
        that produces the highest consistent column count across all sampled
        lines wins.  Falls back to comma when detection fails.
        """
        candidates = [",", ";", "\t", "|"]
        try:
            with open(filepath, encoding="utf-8", errors="replace") as fh:
                lines = [fh.readline() for _ in range(10)]
            lines = [ln for ln in lines if ln.strip()]
            best_sep, best_score = ",", 0
            for sep in candidates:
                counts = [len(ln.split(sep)) for ln in lines]
                # Require all sampled lines to have the same column count
                if counts and min(counts) == max(counts) and max(counts) > best_score:
                    best_sep, best_score = sep, max(counts)
            return best_sep
        except Exception:
            return ","

    @staticmethod
    def _load_raw_file(path: Path) -> pd.DataFrame:
        """
        Read a data file into a DataFrame based on its extension.

        For delimited text files the separator is auto-detected via
        _detect_separator().  Excel files use the first sheet.
        JSON files are expected to be an array of records or a dict-of-lists.
        """
        suffix = path.suffix.lower()

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)

        if suffix == ".json":
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            # Accept both list-of-records and dict-of-lists
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                return pd.DataFrame(payload)
            raise ValueError(f"Unsupported JSON structure in {path.name}")

        # Treat everything else as a delimited text file
        sep = PlannerAgent._detect_separator(path)
        return pd.read_csv(path, sep=sep, encoding="utf-8", on_bad_lines="skip")

    @staticmethod
    def _parse_metadata(path: Path) -> Dict[str, Any]:
        """
        Extract column names, value mappings, and descriptions from a metadata
        file.

        Returns a dict with the following optional keys:
          columns        list[str]       — ordered column names
          value_mappings dict            — {col: {old_val: new_val}}
          description    str             — free-text dataset description
        """
        suffix = path.suffix.lower()
        meta: Dict[str, Any] = {}

        if suffix == ".json":
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            # Accept well-known keys; ignore unknowns gracefully
            if "columns" in raw:
                meta["columns"] = [str(c) for c in raw["columns"]]
            if "value_mappings" in raw:
                meta["value_mappings"] = raw["value_mappings"]
            if "description" in raw:
                meta["description"] = raw["description"]

        elif suffix == ".csv":
            # Treat the first column as the list of column names
            try:
                dict_df = pd.read_csv(path, sep=None, engine="python")
                first_col = dict_df.columns[0]
                meta["columns"] = dict_df[first_col].dropna().astype(str).tolist()
            except Exception as exc:
                print(f"[PlannerAgent] Could not parse CSV metadata: {exc}")

        elif suffix in {".txt", ".names"}:
            # One column name per line; lines starting with '#' are comments
            with open(path, encoding="utf-8", errors="replace") as fh:
                names = [
                    ln.strip().split(":")[0].strip()   # handle "name: description" style
                    for ln in fh
                    if ln.strip() and not ln.startswith("#")
                ]
            if names:
                meta["columns"] = names

        return meta

    @staticmethod
    def _apply_metadata(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply extracted metadata to a DataFrame.

        Column renaming
        ---------------
        If metadata supplies a column list whose length matches the DataFrame,
        the columns are renamed in order.  Mismatches are logged and skipped.

        Value mappings
        --------------
        Each entry in value_mappings is applied to its target column via
        Series.map(), leaving unmapped values unchanged.
        """
        columns = meta.get("columns")
        if columns:
            if len(columns) == len(df.columns):
                df = df.copy()
                df.columns = columns
                print(f"[PlannerAgent] Renamed {len(columns)} columns from metadata.")
            else:
                print(
                    f"[PlannerAgent] Column count mismatch — "
                    f"metadata has {len(columns)}, data has {len(df.columns)}. "
                    "Skipping rename."
                )

        mappings = meta.get("value_mappings", {})
        for col, mapping in mappings.items():
            if col in df.columns:
                # Convert mapping keys to the same type as the column values
                df = df.copy()
                df[col] = df[col].map(
                    lambda v, m=mapping: m.get(v, m.get(str(v), v))
                )
                print(f"[PlannerAgent] Applied value mapping to column '{col}'.")

        return df

    # ------------------------------------------------------------------
    # Extra-file parsing
    # ------------------------------------------------------------------

    # Maximum characters read from any single text-based file to avoid
    # overflowing the LLM context window.
    _MAX_FILE_CHARS = 3000

    @staticmethod
    def _parse_extra_files(files: List[Union[str, Path]]) -> str:
        """
        Read and summarise a list of supplementary files.

        Each supported file type is handled as follows:
          .json        — pretty-printed (truncated to _MAX_FILE_CHARS)
          .xlsx / .xls — loaded as a DataFrame; shape + column names + first 3 rows shown
          .csv         — loaded as a DataFrame; shape + column names + first 3 rows shown
          .txt / .md   — raw text (truncated to _MAX_FILE_CHARS)

        Unrecognised extensions are skipped with a warning.

        Returns
        -------
        str
            A formatted string ready to be appended to the LLM prompt, or an
            empty string if no files were provided or all failed to load.
        """
        if not files:
            return ""

        sections: List[str] = []

        for raw_path in files:
            path = Path(raw_path)
            if not path.exists():
                print(f"[PlannerAgent] Extra file not found, skipping: {path}")
                continue

            suffix = path.suffix.lower()
            label = f"[Extra file: {path.name}]"

            try:
                if suffix == ".json":
                    with open(path, encoding="utf-8") as fh:
                        content = json.dumps(json.load(fh), indent=2, ensure_ascii=False)
                    # Truncate to avoid exceeding context limits
                    if len(content) > PlannerAgent._MAX_FILE_CHARS:
                        content = content[: PlannerAgent._MAX_FILE_CHARS] + "\n... [truncated]"
                    sections.append(f"{label}\n{content}")

                elif suffix in {".xlsx", ".xls"}:
                    df = pd.read_excel(path)
                    summary = (
                        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                        f"Columns: {list(df.columns)}\n"
                        f"First 3 rows:\n{df.head(3).to_string(index=False)}"
                    )
                    sections.append(f"{label}\n{summary}")

                elif suffix == ".csv":
                    # Use Python engine with flexible separators for metadata CSVs
                    df = pd.read_csv(path, sep=None, engine="python", nrows=5)
                    summary = (
                        f"Shape: {df.shape[0]}+ rows × {df.shape[1]} columns\n"
                        f"Columns: {list(df.columns)}\n"
                        f"First 3 rows:\n{df.head(3).to_string(index=False)}"
                    )
                    sections.append(f"{label}\n{summary}")

                elif suffix in {".txt", ".md"}:
                    with open(path, encoding="utf-8") as fh:
                        content = fh.read(PlannerAgent._MAX_FILE_CHARS)
                    if len(content) == PlannerAgent._MAX_FILE_CHARS:
                        content += "\n... [truncated]"
                    sections.append(f"{label}\n{content}")

                else:
                    print(f"[PlannerAgent] Unsupported extra-file format '{suffix}', skipping: {path.name}")
                    continue

            except Exception as exc:
                print(f"[PlannerAgent] Failed to read extra file '{path.name}': {exc}")

        if not sections:
            return ""

        return "\n\n".join(sections)

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

Available models (use these exact names):
  Classification : logistic_regression, random_forest, svm_rbf, xgboost, lightgbm
  Regression     : ridge_regression, random_forest_regressor, svr_rbf,
                   xgboost_regressor, lightgbm_regressor

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
        extra_context: str = "",
    ) -> Dict[str, Any]:
        constraint_text = (
            f"\nConstraints: {json.dumps(constraints, ensure_ascii=False)}"
            if constraints
            else ""
        )
        # Append parsed extra-file content after the schema block when present
        extra_text = f"\n\nSupplementary context (from extra files):\n{extra_context}" if extra_context else ""

        user = (
            f"Business objective:\n{business_description}\n\n"
            f"Dataset schema:\n{schema}"
            f"{extra_text}"
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
        c = constraints or {}
        problem_type = c.get("problem_type") or "classification"
        target_column = c.get("target_column") or None

        if problem_type == "regression":
            primary_metric = "rmse"
            candidate_models = ["ridge_regression", "random_forest_regressor", "xgboost_regressor"]
        else:
            primary_metric = "roc_auc"
            candidate_models = ["logistic_regression", "random_forest", "xgboost"]

        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "primary_metric": primary_metric,
            "feature_config": {
                "task_description": "AutoDS automated feature engineering",
                "use_llm_planner": True,
            },
            "modelling_config": {
                "candidate_model_names": candidate_models,
                "cv_folds": 5,
                "primary_metric": primary_metric,
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
