"""
AutoDS Complete Data Science Pipeline - Implementation Guide

This guide shows how to properly integrate all 6 agents in the correct order
with guaranteed data consistency.

Author: AutoDS Team
Created: 2026-03-19
"""

import os
import sys
from pathlib import Path
from typing import Any


# ============================================================================
# PART 1: SETUP AND CONFIGURATION
# ============================================================================

class PipelineConfig:
    """Configuration for the AutoDS Pipeline"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.data_path = None
        # Column separator for CSV files.
        # Leave as None (default) to let the pipeline auto-detect it from the
        # first few lines of the file.  Set explicitly (e.g. ";") to override.
        self.csv_sep = None
        self.target_column = None
        self.problem_type = None   # None = let Planner or auto-detect decide; set explicitly to "classification" or "regression" to override
        self.random_state = 42
        self.output_base_dir = self.project_root / "autods_pipeline_output"

        # Planner settings
        self.business_description = ""   # natural-language task description
        self.use_planner = True          # set False to skip Stage 0

        # Optional data preparation (Requirement 3)
        # Set metadata_file to a .json/.csv/.txt file that describes the dataset
        # structure (column names, value mappings, etc.).  When set, Stage 0 calls
        # PlannerAgent.prepare_data() to convert the raw file into a clean CSV
        # before the pipeline reads it.
        self.metadata_file = None        # path to metadata / data-dictionary file

        # Stage output directories
        self.stage_dirs = {
            0: self.output_base_dir / "00_planning",
            1: self.output_base_dir / "01_understanding",
            2: self.output_base_dir / "02_cleaning",
            3: self.output_base_dir / "03_feature_engineering",
            4: self.output_base_dir / "04_modelling",
            5: self.output_base_dir / "05_evaluation",
            6: self.output_base_dir / "06_reports",
        }


# ============================================================================
# PART 2: COMPLETE PIPELINE IMPLEMENTATION
# ============================================================================

class DataSciencePipeline:
    """
    Complete Data Science Pipeline with guaranteed data consistency
    
    Data Flow:
    Raw Data (CSV)
        ↓
    [1] Understanding → metadata, profiles
        ↓
    [2] Cleaning → cleaned DataFrame ✓ POINT #1
        ↓
    [3] Feature Engineering → X_train, X_test, y_train, y_test ✓ POINT #2
        ↓
    [4] Modelling → trained models ✓ POINT #3
        ↓
    [5] Evaluation → best model ✓ POINT #4
        ↓
    [6] Report Generation → final report
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._validate_config()
        self._setup_directories()
        self._initialize_intermediate_storage()
        self._planner = None          # set after Stage 0
        self._planner_plan: dict = {} # populated by Stage 0
        
    def _validate_config(self):
        """Validate configuration parameters"""
        if not self.config.data_path:
            raise ValueError("data_path must be specified")
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
        # target_column can be left None when use_planner=True — the Planner
        # will infer and set it during Stage 0.
        if not self.config.target_column and not self.config.use_planner:
            raise ValueError("target_column must be specified (or set use_planner=True)")
        
    def _setup_directories(self):
        """Create all necessary output directories"""
        for stage, dir_path in self.config.stage_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Stage {stage} output directory: {dir_path}")
    
    def _initialize_intermediate_storage(self):
        """Initialize storage for intermediate results"""
        self.stage_outputs = {}
        self.data_lineage = []

    # ========================================================================
    # SEPARATOR AUTO-DETECTION
    # ========================================================================

    # Human-readable names for common separators (used in log messages)
    _SEP_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe", " ": "space"}

    def _resolve_csv_separator(self) -> str:
        """
        Return the CSV column separator to use for config.data_path.

        If config.csv_sep is already set (non-None), it is returned as-is so
        that explicit user overrides are always respected.

        Otherwise the separator is sniffed from the first ten lines of the
        file: comma, semicolon, tab, and pipe are tried in order; the one that
        produces the highest *consistent* column count across all sampled lines
        wins.  The result is written back to config.csv_sep so every subsequent
        stage automatically reuses the same value without repeating detection.
        Falls back to comma when the file cannot be read or no clear winner
        is found.
        """
        if self.config.csv_sep is not None:
            # Already resolved — skip detection
            return self.config.csv_sep

        candidates = [",", ";", "\t", "|"]
        detected = ","   # safe default

        try:
            with open(self.config.data_path, encoding="utf-8", errors="replace") as fh:
                lines = [fh.readline() for _ in range(10)]
            lines = [ln for ln in lines if ln.strip()]

            best_sep, best_score = ",", 0
            for sep in candidates:
                counts = [len(ln.split(sep)) for ln in lines]
                # Accept only separators that give a *consistent* column count
                # (min == max) and more than one column.
                if counts and min(counts) == max(counts) and max(counts) > best_score:
                    best_sep, best_score = sep, max(counts)

            detected = best_sep

        except Exception as exc:
            print(f"  [AutoDetect] Separator detection failed ({exc}), defaulting to comma.")

        self.config.csv_sep = detected
        sep_name = self._SEP_NAMES.get(detected, repr(detected))
        print(f"  [AutoDetect] CSV separator detected: {sep_name} ({repr(detected)})")
        return detected

    def _resolve_problem_type(self, df: "pd.DataFrame") -> str:
        """
        Return the problem type to use for this pipeline run.

        If config.problem_type is already set (non-None), it is returned
        as-is so that explicit user overrides are always respected.

        Otherwise the type is inferred from the target column dtype:
        - float or integer with more than 20 unique values → "regression"
        - everything else → "classification"

        The result is written back to config.problem_type so every
        subsequent stage automatically reuses the same value.
        """
        import pandas as pd

        if self.config.problem_type is not None:
            return self.config.problem_type

        target = self.config.target_column
        if target and target in df.columns:
            col = df[target]
            n_unique = col.nunique()
            if pd.api.types.is_float_dtype(col) or (
                pd.api.types.is_integer_dtype(col) and n_unique > 20
            ):
                detected = "regression"
            else:
                detected = "classification"
        else:
            detected = "classification"   # safe default when target unknown

        self.config.problem_type = detected
        print(f"  [AutoDetect] problem_type detected: {detected}")
        return detected

    # ========================================================================
    # STAGE 0: PLANNING
    # ========================================================================

    def run_stage_0_planning(self):
        """
        Stage 0: Planner Agent

        Purpose : parse business description, generate all downstream configs
        Input   : business_description (str) + first rows of the dataset
        Output  : plan dict stored in self._planner_plan
        """
        print("\n" + "=" * 80)
        print("STAGE 0: PLANNING")
        print("=" * 80)

        try:
            import pandas as pd
            from planner_agent import PlannerAgent, PlannerConfig

            planner_config = PlannerConfig(
                output_dir=str(self.config.stage_dirs[0]),
            )
            self._planner = PlannerAgent(planner_config)

            # If a metadata file is provided, run prepare_data() to convert the
            # raw data file into a clean, column-labelled CSV before the pipeline
            # reads it.  The output CSV overwrites config.data_path so that all
            # downstream stages automatically use the prepared file.
            # prepare_data() always writes comma-separated output, so fix the
            # separator immediately instead of running auto-detection on it.
            if self.config.metadata_file:
                prepared_csv = self.config.stage_dirs[0] / "prepared_data.csv"
                self._planner.prepare_data(
                    data_file=self.config.data_path,
                    metadata_file=self.config.metadata_file,
                    output_csv=prepared_csv,
                )
                self.config.data_path = prepared_csv
                self.config.csv_sep = ","   # prepared CSV is always comma-delimited
                print(f"  [Planner] data_path updated to prepared CSV: {prepared_csv}")
            else:
                # Auto-detect (or honour the user's explicit override) before
                # reading the schema sample.
                self._resolve_csv_separator()

            # Load a small sample so the LLM can see the schema
            data_sample = pd.read_csv(self.config.data_path,
                                      sep=self.config.csv_sep,
                                      nrows=planner_config.data_sample_rows)

            # Pass user-supplied overrides as hard constraints so the rule-based
            # fallback (and the LLM) can respect them even without LLM inference.
            user_constraints: dict = {}
            if self.config.target_column:
                user_constraints["target_column"] = self.config.target_column
            if self.config.problem_type:
                user_constraints["problem_type"] = self.config.problem_type

            plan = self._planner.plan(
                business_description=self.config.business_description,
                data_sample=data_sample,
                constraints=user_constraints or None,
            )
            self._planner_plan = plan

            # Apply planner's target_column / problem_type only if not set by user
            if not self.config.target_column and plan.get("target_column"):
                self.config.target_column = plan["target_column"]
                print(f"  [Planner] target_column set to: {self.config.target_column}")

            if not self.config.problem_type and plan.get("problem_type"):
                self.config.problem_type = plan["problem_type"]
                print(f"  [Planner] problem_type set to : {self.config.problem_type}")

            self.stage_outputs[0] = {"plan": plan}
            return plan

        except Exception as exc:
            print(f"❌ Stage 0 failed: {exc}")
            raise

    # ========================================================================
    # STAGE 1: DATA UNDERSTANDING
    # ========================================================================
    
    def run_stage_1_understanding(self):
        """
        Stage 1: Data Understanding
        
        Purpose: Analyze raw data characteristics
        Input: Raw CSV file path
        Output: Metadata + Profiles (saved to disk)
        Data Consistency: ✓ Establishes baseline
        """
        print("\n" + "="*80)
        print("STAGE 1: DATA UNDERSTANDING")
        print("="*80)
        
        try:
            # Import the agent
            from data_understanding_agent import (
                AgentConfig as UnderstandingConfig,
                DataUnderstandingAgent
            )
            import pandas as pd
            
            # Ensure separator is resolved (covers the case where Stage 0 was
            # skipped via use_planner=False, or where csv_sep was not set).
            self._resolve_csv_separator()

            # Load raw data
            print(f"\n📖 Loading raw data from: {self.config.data_path}")
            raw_data = pd.read_csv(self.config.data_path, sep=self.config.csv_sep)
            print(f"✓ Raw data loaded: {raw_data.shape}")
            self.data_lineage.append({
                'stage': 1,
                'output': 'raw_data',
                'shape': raw_data.shape,
                'location': self.config.data_path
            })
            
            # Configure agent
            config = UnderstandingConfig(
                output_dir=str(self.config.stage_dirs[1]),
                target_column=self.config.target_column,
                problem_type=self.config.problem_type,
                dataset_name="AutoDS_Dataset",
                random_state=self.config.random_state,
            )

            # Run understanding agent
            print(f"\n🔍 Running DataUnderstandingAgent...")
            agent = DataUnderstandingAgent(config)
            result = agent.run(raw_data)
            
            # Store results
            self.stage_outputs[1] = {
                'raw_data': raw_data,
                'understanding_result': result,
                'config': config,
            }
            
            print(f"✓ Understanding complete")
            print(f"  - Columns: {len(raw_data.columns)}")
            print(f"  - Rows: {len(raw_data)}")
            print(f"  - Output directory: {self.config.stage_dirs[1]}")

            # ── Adaptive replanning ──────────────────────────────────
            if self._planner is not None and self._planner_plan:
                replan = self._planner.replan_after_understanding(
                    understanding_output=result,
                    current_plan=self._planner_plan,
                )
                updated = replan.get("updated_plan", {})
                if updated:
                    import copy
                    self._planner_plan = self._deep_merge(
                        copy.deepcopy(self._planner_plan), updated
                    )

            return raw_data, result

        except Exception as e:
            print(f"❌ Stage 1 failed: {str(e)}")
            raise
    
    # ========================================================================
    # STAGE 2: DATA CLEANING
    # ========================================================================
    
    def run_stage_2_cleaning(self):
        """
        Stage 2: Data Cleaning
        
        Purpose: Clean and preprocess data
        Input: Raw DataFrame (from Stage 1) ← DATA CONSISTENCY POINT #1
        Output: Cleaned DataFrame
        Guarantee: All rows tracked, data integrity maintained
        """
        print("\n" + "="*80)
        print("STAGE 2: DATA CLEANING")
        print("="*80)
        
        try:
            # Validate previous stage
            if 1 not in self.stage_outputs:
                raise RuntimeError("Stage 1 must be completed first")
            
            # Get input from Stage 1
            raw_data = self.stage_outputs[1]['raw_data']
            print(f"\n🧹 Input data (from Stage 1): {raw_data.shape}")
            
            # Import agent
            from data_cleaning_agent import DataCleaningAgent, DataCleaningConfig

            # Configure — pass target_column so anomaly removal skips the label
            cleaning_config = DataCleaningConfig(
                output_dir=str(self.config.stage_dirs[2]),
                target_column=self.config.target_column,
            )

            # Create and run agent
            agent = DataCleaningAgent(name="DataCleaner", config=cleaning_config)

            print(f"\n⏳ Executing DataCleaningAgent...")
            cleaning_result = agent.run(raw_data)
            cleaned_data = cleaning_result['data']
            cleaning_report = cleaning_result.get('report', {})

            print(f"✓ Cleaning complete")
            print(f"  - Output shape: {cleaned_data.shape}")
            print(f"  - Rows removed: {len(raw_data) - len(cleaned_data)}")
            print(f"  - Data retention: {len(cleaned_data)/len(raw_data)*100:.1f}%")

            # Save for next stage
            cleaned_path = self.config.stage_dirs[2] / "cleaned_data.csv"
            cleaned_data.to_csv(cleaned_path, index=False)
            print(f"  - Saved to: {cleaned_path}")

            # After cleaning, column names are normalised (lowercased).
            # Update config.target_column to match the normalised name so that
            # Stage 3+ can locate the target column in the cleaned DataFrame.
            if self.config.target_column:
                normalised_target = self.config.target_column.lower().strip()
                if normalised_target in cleaned_data.columns:
                    self.config.target_column = normalised_target
                elif self.config.target_column not in cleaned_data.columns:
                    print(f"  ⚠️  target_column '{self.config.target_column}' not found after "
                          f"cleaning; keeping original name for downstream stages.")

            # Auto-detect problem_type if not already set by user or Planner
            self._resolve_problem_type(cleaned_data)
            print(f"  - problem_type: {self.config.problem_type}")

            # Track data lineage
            self.data_lineage.append({
                'stage': 2,
                'input_shape': raw_data.shape,
                'output_shape': cleaned_data.shape,
                'rows_removed': len(raw_data) - len(cleaned_data),
            })

            # Store results
            self.stage_outputs[2] = {
                'cleaned_data': cleaned_data,
                'cleaning_report': cleaning_report,
                'agent': agent,
            }

            return cleaned_data, cleaning_report
            
        except Exception as e:
            print(f"❌ Stage 2 failed: {str(e)}")
            raise
    
    # ========================================================================
    # STAGE 3: FEATURE ENGINEERING
    # ========================================================================
    
    def run_stage_3_feature_engineering(self):
        """
        Stage 3: Feature Engineering
        
        Purpose: Transform features and prepare training/test sets
        Input: Cleaned DataFrame (from Stage 2) ← DATA CONSISTENCY POINT #2
        Output: X_train, X_test, y_train, y_test
        Guarantee: Proper train/test split, feature consistency
        """
        print("\n" + "="*80)
        print("STAGE 3: FEATURE ENGINEERING")
        print("="*80)
        
        try:
            # Validate previous stage
            if 2 not in self.stage_outputs:
                raise RuntimeError("Stage 2 must be completed first")
            
            # Get input from Stage 2
            cleaned_data = self.stage_outputs[2]['cleaned_data']
            print(f"\n⚙️  Input data (from Stage 2): {cleaned_data.shape}")
            
            # Import agent
            from feature_engineering_agent import (
                FeatureEngineeringConfig,
                FeatureEngineeringAgent
            )
            from sklearn.model_selection import train_test_split
            
            # Pull planner overrides (if any)
            fe_cfg = self._planner_plan.get("feature_config", {})

            # Configure agent
            config = FeatureEngineeringConfig(
                target_column=self.config.target_column,
                problem_type=self.config.problem_type,
                task_description=fe_cfg.get(
                    "task_description", "AutoDS Feature Engineering"
                ),
                use_llm_planner=fe_cfg.get("use_llm_planner", True),
                save_artifacts=True,
                output_dir=str(self.config.stage_dirs[3]),
                random_state=self.config.random_state,
            )
            
            # Create and run agent
            agent = FeatureEngineeringAgent(config)
            print(f"\n⏳ Executing FeatureEngineeringAgent...")
            # ← KEY: Using output from Stage 2 as input
            # NOTE: FeatureEngineeringAgent.run() returns a Dictionary, not tuple!
            result = agent.run(cleaned_data)
            X = result['X']
            y = result['y']
            summary = result['summary']
            
            print(f"✓ Feature engineering complete")
            print(f"  - Features shape: {X.shape}")
            print(f"  - Target shape: {y.shape}")
            
            # Perform train/test split
            print(f"\n📊 Performing train/test split (80/20)...")
            
            # ========================================================================
            # 修复：处理小数据集和不平衡类别
            # ========================================================================
            
            # 检查类别分布
            if self.config.problem_type == "classification":
                try:
                    import pandas as pd
                    class_dist = pd.Series(y).value_counts()
                    min_class_count = class_dist.min()
                    
                    # 如果最小类只有 1 个样本，无法 stratify
                    if min_class_count < 2:
                        print(f"  ⚠️  类别不平衡 (最小类只有 {min_class_count} 个样本)")
                        print(f"  → 禁用 stratify 分割")
                        use_stratify = False
                    else:
                        use_stratify = True
                except:
                    use_stratify = True
            else:
                use_stratify = False
            
            # 执行分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=self.config.random_state,
                stratify=y if use_stratify else None
            )
            
            print(f"  - Training set: {X_train.shape}")
            print(f"  - Test set: {X_test.shape}")
            
            # Save datasets
            X_train.to_csv(self.config.stage_dirs[3] / "X_train.csv", index=False)
            X_test.to_csv(self.config.stage_dirs[3] / "X_test.csv", index=False)
            y_train.to_csv(self.config.stage_dirs[3] / "y_train.csv", index=False, header=['target'])
            y_test.to_csv(self.config.stage_dirs[3] / "y_test.csv", index=False, header=['target'])
            
            # Track data lineage
            self.data_lineage.append({
                'stage': 3,
                'input_shape': cleaned_data.shape,
                'X_train_shape': X_train.shape,
                'X_test_shape': X_test.shape,
                'train_test_split': '80/20',
            })
            
            # Store results
            self.stage_outputs[3] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_summary': summary,
                'agent': agent,
            }
            
            return X_train, X_test, y_train, y_test, summary
            
        except Exception as e:
            print(f"❌ Stage 3 failed: {str(e)}")
            raise
    
    # ========================================================================
    # STAGE 4: MODELLING
    # ========================================================================
    
    def run_stage_4_modelling(self):
        """
        Stage 4: Modelling
        
        Purpose: Train and evaluate models
        Input: X_train, X_test, y_train, y_test (from Stage 3) ← DATA CONSISTENCY POINT #3
        Output: Trained models + Leaderboard
        Guarantee: Same data used throughout training and evaluation
        """
        print("\n" + "="*80)
        print("STAGE 4: MODELLING")
        print("="*80)
        
        try:
            # Validate previous stage
            if 3 not in self.stage_outputs:
                raise RuntimeError("Stage 3 must be completed first")
            
            # Get input from Stage 3
            X_train = self.stage_outputs[3]['X_train']
            X_test = self.stage_outputs[3]['X_test']
            y_train = self.stage_outputs[3]['y_train']
            y_test = self.stage_outputs[3]['y_test']

            # Convert string target labels (e.g. '0'/'1' from boolean unification)
            # to integers so that XGBoost and other estimators accept them.
            import pandas as pd
            y_train = pd.to_numeric(y_train, errors='ignore')
            if y_test is not None:
                y_test = pd.to_numeric(y_test, errors='ignore')

            # New ModellingAgent.run() takes the full dataset and splits internally.
            # Recombine Stage 3 splits so the agent can do its own stratified split.
            X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
            y_all = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
            test_size = len(X_test) / (len(X_train) + len(X_test))

            print(f"\n🤖 Input data (from Stage 3):")
            print(f"  - Training set: {X_train.shape}")
            print(f"  - Test set: {X_test.shape}")

            # Import agent
            from modelling_agent import ModellingConfig, ModellingAgent

            # Pull planner overrides (if any)
            mc_cfg = self._planner_plan.get("modelling_config", {})

            # Normalize planner model names to the snake_case names used by ModellingAgent.
            # Planner LLM returns PascalCase (e.g. "RandomForest"); the agent uses
            # snake_case (e.g. "random_forest"). Unrecognised names are passed through
            # unchanged — ModellingAgent will silently skip them and log a warning.
            _NAME_MAP = {
                # classification
                "logisticregression":   "logistic_regression",
                "randomforest":         "random_forest",
                "svc":                  "svm_rbf",
                "svm":                  "svm_rbf",
                "xgboost":              "xgboost",
                "lightgbm":             "lightgbm",
                "decisiontree":         "random_forest",  # closest available
                "kneighbors":           "logistic_regression",
                "gradientboosting":     "xgboost",
                # regression
                "linearregression":     "ridge_regression",
                "ridge":                "ridge_regression",
                "lasso":                "ridge_regression",
                "randomforestregressor":"random_forest_regressor",
                "svr":                  "svr_rbf",
                "xgboostregressor":     "xgboost_regressor",
                "lightgbmregressor":    "lightgbm_regressor",
            }
            planner_names = mc_cfg.get("candidate_model_names") or []
            validated_names = [
                _NAME_MAP.get(n.lower().replace("_", "").replace("-", ""), n)
                for n in planner_names
            ] or None

            # Configure agent
            config = ModellingConfig(
                target_column=self.config.target_column,
                problem_type=self.config.problem_type,
                task_description="AutoDS Modelling",
                primary_metric=mc_cfg.get(
                    "primary_metric",
                    self._planner_plan.get("primary_metric"),
                ),
                cv_folds=mc_cfg.get("cv_folds", 5),
                candidate_model_names=validated_names,
                output_dir=str(self.config.stage_dirs[4]),
                random_state=self.config.random_state,
            )

            # Create and run agent
            agent = ModellingAgent(config)
            print(f"\n⏳ Executing ModellingAgent...")
            result = agent.run(X=X_all, y=y_all, test_size=test_size)

            model_count = len(result.get('leaderboard', pd.DataFrame()))
            print(f"✓ Modelling complete")
            print(f"  - Models trained: {model_count}")
            print(f"  - Best model: {result.get('best_model_name', 'N/A')}")
            print(f"  - Output directory: {self.config.stage_dirs[4]}")

            # Track data lineage
            self.data_lineage.append({
                'stage': 4,
                'training_shape': X_train.shape,
                'test_shape': X_test.shape,
                'models_trained': model_count,
                'best_model': result.get('best_model_name'),
            })
            
            # Store results
            self.stage_outputs[4] = {
                'modelling_result': result,
                'agent': agent,
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Stage 4 failed: {str(e)}")
            raise
    
    # ========================================================================
    # STAGE 5: EVALUATION
    # ========================================================================
    
    def run_stage_5_evaluation(self):
        """
        Stage 5: Evaluation
        
        Purpose: Compare and select best model
        Input: Model artifacts (from Stage 4) ← DATA CONSISTENCY POINT #4
        Output: Best model selection + Evaluation metrics
        Guarantee: Evaluation based on identical model outputs
        """
        print("\n" + "="*80)
        print("STAGE 5: EVALUATION")
        print("="*80)
        
        try:
            # Validate previous stage
            if 4 not in self.stage_outputs:
                raise RuntimeError("Stage 4 must be completed first")
            
            print(f"\n📊 Input data (from Stage 4):")
            print(f"  - Model artifacts directory: {self.config.stage_dirs[4]}")
            
            # Import agent (new Evaluation_agent.py)
            from Evaluation_agent import EvaluationConfig, EvaluationAgent
            
            # Configure agent
            config = EvaluationConfig(
                modelling_output_dir=str(self.config.stage_dirs[4]),
                output_dir=str(self.config.stage_dirs[5]),
                save_artifacts=True,
            )
            
            # Create and run agent
            agent = EvaluationAgent(config)
            print(f"\n⏳ Executing EvaluationAgent...")
            # ← KEY: Using model artifacts from Stage 4
            result = agent.run()
            
            print(f"✓ Evaluation complete")
            print(f"  - Best model: {result.get('best_model_name', 'N/A')}")
            print(f"  - Primary metric: {result.get('primary_metric', 'N/A')}")
            print(f"  - Output directory: {self.config.stage_dirs[5]}")

            # Track data lineage
            self.data_lineage.append({
                'stage': 5,
                'best_model': result.get('best_model_name'),
                'primary_metric': result.get('primary_metric'),
                'candidate_models': result.get('benchmark_overview', {}).get('candidate_model_count'),
            })

            # ── Planner post-modelling review ────────────────────────
            planner_review = {}
            if self._planner is not None:
                modelling_result = self.stage_outputs.get(4, {}).get('modelling_result', {})
                planner_review = self._planner.review_modelling(
                    modelling_output=modelling_result,
                    evaluation_output=result,
                )

            # Store results
            self.stage_outputs[5] = {
                'evaluation_result': result,
                'planner_review': planner_review,
                'agent': agent,
            }

            return result
            
        except Exception as e:
            print(f"❌ Stage 5 failed: {str(e)}")
            raise
    
    # ========================================================================
    # STAGE 6: REPORT GENERATION
    # ========================================================================
    
    def _make_json_serializable(self, obj: Any) -> Any:

        import numpy as np
        import pandas as pd
        
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        elif isinstance(obj, pd.DataFrame):
            # DataFrame -> dict
            return obj.to_dict(orient='records')
        
        elif isinstance(obj, pd.Series):
            # Series -> dict/list
            return obj.to_list() if obj.index.name is None else obj.to_dict()
        
        elif isinstance(obj, dict):
            # 递归处理字典
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        
        elif isinstance(obj, (list, tuple)):
            # 递归处理列表
            return [self._make_json_serializable(item) for item in obj]
        
        elif isinstance(obj, (np.integer, np.floating)):
            # numpy 数值类型转换为 Python 原生类型
            return obj.item()
        
        elif isinstance(obj, np.ndarray):
            # numpy 数组转换为列表
            return obj.tolist()
        
        else:
            # 其他类型转换为字符串
            return str(obj)
    
    def run_stage_6_report_generation(self):
        """
        Stage 6: Report Generation
        
        Purpose: Generate comprehensive final report
        Input: All artifacts (from all Stages)
        Output: HTML + JSON reports
        """
        print("\n" + "="*80)
        print("STAGE 6: REPORT GENERATION")
        print("="*80)
        
        try:
            print(f"\n⏳ Executing ReportGenerator...")

            from multi_agent_report_generator import (
                MultiAgentReportGenerator,
                ReportGeneratorConfig,
            )

            import pandas as pd
            import json

            # Configure
            report_config = ReportGeneratorConfig(
                output_dir=str(self.config.stage_dirs[6]),
            )
            agent = MultiAgentReportGenerator(config=report_config)

            # Build report input dict mapped to the normalized schema expected by
            # MultiAgentReportGenerator (meta/data_understanding/data_cleaning/
            # feature_engineering/modeling/evaluation/business_context).
            print("\n  [*] preparing report data...")
            report_json_path = self.config.stage_dirs[6] / "pipeline_report_input.json"

            # ── Pull raw stage outputs ────────────────────────────────────────
            raw_data      = self.stage_outputs.get(1, {}).get('raw_data')
            understanding = self.stage_outputs.get(1, {}).get('understanding_result', {})
            cleaned_data  = self.stage_outputs.get(2, {}).get('cleaned_data')
            cleaning_rpt  = self.stage_outputs.get(2, {}).get('cleaning_report', {})
            feat_summary  = self.stage_outputs.get(3, {}).get('feature_summary', {})
            mod_result    = self.stage_outputs.get(4, {}).get('modelling_result', {})
            eval_result   = self.stage_outputs.get(5, {}).get('evaluation_result', {})

            # ── Helpers ───────────────────────────────────────────────────────
            leaderboard_df = mod_result.get('leaderboard', pd.DataFrame())
            leaderboard_records = (
                leaderboard_df.to_dict(orient='records')
                if isinstance(leaderboard_df, pd.DataFrame) else []
            )
            best_metrics = mod_result.get('best_model_metrics', {})
            if not isinstance(best_metrics, dict):
                best_metrics = {}

            fi_df = mod_result.get('best_model_feature_importance', pd.DataFrame())
            if isinstance(fi_df, pd.DataFrame) and not fi_df.empty:
                fi_records = fi_df.to_dict(orient='records')
            else:
                fi_records = []

            primary_metric = (
                mod_result.get('primary_metric')
                or eval_result.get('primary_metric')
                or 'roc_auc'
            )
            primary_score = best_metrics.get(f'test_{primary_metric}') or best_metrics.get(primary_metric)

            # models_compared: list of dicts with name + metrics from leaderboard
            models_compared = []
            for i, row in enumerate(leaderboard_records, start=1):
                models_compared.append({
                    "name": row.get('model_name', f'model_{i}'),
                    "rank": i,
                    **{k: v for k, v in row.items() if k != 'model_name'},
                })

            # ── Build normalized schema ───────────────────────────────────────
            report_data = {
                "meta": {
                    "project_name": "AutoDS Pipeline",
                    "dataset_name": str(self.config.data_path),
                    "target_variable": self.config.target_column,
                    "task_type": self.config.problem_type,
                    "timestamp": str(pd.Timestamp.now()),
                    "models_evaluated": len(leaderboard_records),
                },
                "data_understanding": {
                    "n_rows": int(raw_data.shape[0]) if raw_data is not None else None,
                    "n_cols": int(raw_data.shape[1]) if raw_data is not None else None,
                    "n_rows_after_cleaning": int(cleaned_data.shape[0]) if cleaned_data is not None else None,
                    "class_imbalance_ratio": None,
                    "key_insights": [],
                    "missing_values_summary": {},
                },
                "data_cleaning": {
                    "operations_performed": [],
                    "data_quality_score": None,
                    "quality_notes": self._make_json_serializable(cleaning_rpt),
                },
                "feature_engineering": {
                    "features_created": feat_summary.get('used_columns', []),
                    "features_dropped": feat_summary.get('dropped_columns', []),
                    "encoding_applied": {},
                    "feature_importances": fi_records,
                    "key_insights": [],
                    "final_feature_count": feat_summary.get('final_feature_count'),
                },
                "modeling": {
                    "best_model": {
                        "name": mod_result.get('best_model_name', 'N/A'),
                        "params": {},
                        "training_time_seconds": None,
                        "optimization_method": None,
                    },
                    "models_compared": models_compared,
                    "selection_reason": (
                        f"Highest {primary_metric} score on held-out test set."
                    ),
                },
                "evaluation": {
                    "primary_metric": primary_metric,
                    "primary_score": primary_score,
                    "metrics": {k: v for k, v in best_metrics.items() if isinstance(v, (int, float))},
                    "confusion_matrix": {},
                    "cv_scores": [],
                    "cv_mean": None,
                    "cv_std": None,
                    "key_insights": [],
                },
                "business_context": {
                    "use_case": self.config.business_description or "Automated Data Science",
                    "industry": "Technology",
                    "target_audience": "Data Science Team",
                    "business_goal": self.config.business_description or "Automated end-to-end data science pipeline",
                },
                "planner_review": self._make_json_serializable(
                    self.stage_outputs.get(5, {}).get('planner_review', {})
                ),
                "planner_plan": self._make_json_serializable(self._planner_plan),
            }

            # Save input JSON for reference
            with open(report_json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"  [✓] input JSON saved: {report_json_path}")

            # Run report generator — pass dict directly
            print("  [*] generating report...")
            result = agent.run(report_data)

            if result.get('status') == 'success':
                print(f"✓ Report generation complete")
                print(f"  - Mode: {result.get('generation_mode', 'N/A')}")
                saved = result.get('saved_paths') or {}
                for label, path in saved.items():
                    print(f"  - {label}: {path}")
            else:
                err = result.get('error', 'unknown error')
                print(f"❌ Report generation failed: {err}")
                raise Exception(f"Report generation failed: {err}")

            # Store results
            self.stage_outputs[6] = {
                'final_report': result,
                'agent': agent,
            }

            return result
            
        except Exception as e:
            print(f"❌ Stage 6 failed: {str(e)}")
            raise
    
    # ========================================================================
    # COMPLETE PIPELINE EXECUTION
    # ========================================================================
    
    def run_complete_pipeline(self):
        """Execute the complete pipeline from start to finish"""
        print("\n" + "="*80)
        print("🚀 STARTING COMPLETE DATA SCIENCE PIPELINE")
        print("="*80)
        
        try:
            # Stage 0: Planning (optional)
            if self.config.use_planner:
                self.run_stage_0_planning()

            # Run all 6 stages
            self.run_stage_1_understanding()
            self.run_stage_2_cleaning()
            self.run_stage_3_feature_engineering()
            self.run_stage_4_modelling()
            self.run_stage_5_evaluation()
            
            # ✅ Stage 6: Report Generation (enabled)
            self.run_stage_6_report_generation()
            
            print("\n" + "="*80)
            print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print("   (All 6 Stages completed)")
            print("="*80)
            
            return self._get_final_summary()
            
        except Exception as e:
            print(f"\n❌ Pipeline execution failed: {str(e)}")
            raise
    
    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base (override wins on conflicts)."""
        for key, val in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                base[key] = DataSciencePipeline._deep_merge(base[key], val)
            else:
                base[key] = val
        return base

    def _get_final_summary(self):
        """Get final pipeline summary"""
        return {
            'status': 'success',
            'data_lineage': self.data_lineage,
            'output_directories': self.config.stage_dirs,
            'results': {
                'best_model': self.stage_outputs[5]['evaluation_result'].get('best_model_name'),
                'primary_metric': self.stage_outputs[5]['evaluation_result'].get('primary_metric'),
            }
        }
    
    def print_data_lineage(self):
        """Print data flow through pipeline"""
        print("\n" + "="*80)
        print("📊 DATA LINEAGE THROUGH PIPELINE")
        print("="*80)
        
        for item in self.data_lineage:
            stage = item['stage']
            print(f"\nStage {stage}:")
            for key, value in item.items():
                if key != 'stage':
                    print(f"  {key}: {value}")


# ============================================================================
# PART 3: USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of the complete pipeline"""
    
    # Step 1: Configure pipeline
    config = PipelineConfig()
    config.data_path = "your_data.csv"  # ← CHANGE THIS
    config.target_column = "target"      # ← CHANGE THIS
    config.problem_type = "classification"
    
    # Step 2: Create pipeline
    pipeline = DataSciencePipeline(config)
    
    # Step 3: Run complete pipeline
    result = pipeline.run_complete_pipeline()
    
    # Step 4: Print data lineage
    pipeline.print_data_lineage()
    
    # Step 5: Return results
    return result


if __name__ == "__main__":
    result = main()
