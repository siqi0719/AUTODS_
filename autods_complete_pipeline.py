"""
AutoDS Complete Data Science Pipeline

This module orchestrates the complete data science workflow by connecting
all agents in the proper order:

1. DataUnderstandingAgent - Analyze raw data characteristics
2. DataCleaningAgent - Clean and preprocess data
3. FeatureEngineeringAgent - Create and transform features
4. ModellingAgent - Train and evaluate models
5. EvaluationAgent - Compare and select best model
6. ReportGenerator - Generate comprehensive report

Data Flow:
Raw Data
  ↓
Understanding → Understand raw data
  ↓
Cleaned Data
  ↓
Cleaning → Clean data
  ↓
Engineered Data
  ↓
FeatureEngineering → Transform features
  ↓
Trained Models
  ↓
Modelling → Train and evaluate models
  ↓
Best Model Selection
  ↓
Evaluation → Evaluate and select best model
  ↓
Final Report
  ↓
ReportGenerator → Generate comprehensive report
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import all agents
from data_understanding_agent import AgentConfig as UnderstandingConfig
from data_understanding_agent import DataUnderstandingAgent

from data_cleaning_agent import DataCleaningAgent

from feature_engineering_agent import (
    FeatureEngineeringConfig,
    FeatureEngineeringAgent
)

from modelling_agent import (
    ModellingConfig,
    ModellingAgent
)

from evaluation import (
    EvaluationConfig,
    EvaluationAgent
)

from multi_agent_report_generator import (
    ReportGeneratorConfig,
    ReportGenerator
)


class AutoDSPipeline:
    """
    Automated Data Science Pipeline
    
    Orchestrates the complete workflow from raw data to final model selection
    and report generation, ensuring data consistency across all stages.
    """
    
    def __init__(
        self,
        data_path: str,
        target_column: str,
        problem_type: str = "classification",
        output_dir: str = "./autods_pipeline_output",
        random_state: int = 42,
    ):
        """
        Initialize the AutoDS Pipeline
        
        Args:
            data_path: Path to raw data file (CSV or Parquet)
            target_column: Name of the target column
            problem_type: "classification" or "regression"
            output_dir: Base directory for all outputs
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.target_column = target_column
        self.problem_type = problem_type
        self.random_state = random_state
        self.base_output_dir = Path(output_dir)
        
        # Create output directories for each stage
        self.understanding_output = self.base_output_dir / "01_understanding"
        self.cleaning_output = self.base_output_dir / "02_cleaning"
        self.feature_output = self.base_output_dir / "03_feature_engineering"
        self.modelling_output = self.base_output_dir / "04_modelling"
        self.evaluation_output = self.base_output_dir / "05_evaluation"
        self.report_output = self.base_output_dir / "06_reports"
        
        # Create all directories
        for dir_path in [
            self.understanding_output,
            self.cleaning_output,
            self.feature_output,
            self.modelling_output,
            self.evaluation_output,
            self.report_output,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Store intermediate results
        self.raw_data = None
        self.understanding_result = None
        self.cleaned_data = None
        self.cleaning_report = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_engineering_summary = None
        self.modelling_result = None
        self.evaluation_result = None
        self.final_report = None
        
        print("=" * 80)
        print("🚀 AutoDS Complete Data Science Pipeline Initialized")
        print("=" * 80)
        print(f"📁 Data Path: {data_path}")
        print(f"🎯 Target Column: {target_column}")
        print(f"📊 Problem Type: {problem_type}")
        print(f"📂 Output Directory: {output_dir}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline
        
        Returns:
            Dictionary with pipeline execution results
        """
        print("\n" + "=" * 80)
        print("▶️  STARTING AUTODS PIPELINE EXECUTION")
        print("=" * 80)
        
        try:
            # Step 1: Data Understanding
            print("\n[Step 1/6] 📖 Running Data Understanding Agent...")
            self.raw_data, self.understanding_result = self._run_understanding()
            
            # Step 2: Data Cleaning
            print("\n[Step 2/6] 🧹 Running Data Cleaning Agent...")
            self.cleaned_data, self.cleaning_report = self._run_cleaning()
            
            # Step 3: Feature Engineering
            print("\n[Step 3/6] ⚙️  Running Feature Engineering Agent...")
            self.X_train, self.X_test, self.y_train, self.y_test, self.feature_engineering_summary = \
                self._run_feature_engineering()
            
            # Step 4: Modelling
            print("\n[Step 4/6] 🤖 Running Modelling Agent...")
            self.modelling_result = self._run_modelling()
            
            # Step 5: Evaluation
            print("\n[Step 5/6] 📊 Running Evaluation Agent...")
            self.evaluation_result = self._run_evaluation()
            
            # Step 6: Report Generation
            print("\n[Step 6/6] 📝 Generating Final Report...")
            self.final_report = self._run_report_generation()
            
            # Print final summary
            print("\n" + "=" * 80)
            print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            self._print_summary()
            
            return {
                "status": "success",
                "raw_data_shape": str(self.raw_data.shape),
                "cleaned_data_shape": str(self.cleaned_data.shape),
                "training_set_shape": str(self.X_train.shape),
                "test_set_shape": str(self.X_test.shape),
                "best_model": self.evaluation_result.get("best_model_name"),
                "output_directory": str(self.base_output_dir),
            }
        
        except Exception as e:
            print(f"\n❌ Pipeline Execution Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _run_understanding(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Step 1: Run DataUnderstandingAgent
        
        INPUT: Raw data file path
        OUTPUT: Raw DataFrame + Understanding metadata
        
        PURPOSE: Analyze raw data characteristics, identify data quality issues
        """
        print("  ⏳ Loading raw data...")
        df = pd.read_csv(self.data_path)
        print(f"  ✓ Raw data loaded: {df.shape}")
        
        config = UnderstandingConfig(
            data_path=self.data_path,
            output_dir=str(self.understanding_output),
            target_column=self.target_column,
            problem_type=self.problem_type,
            dataset_name="AutoDS_Dataset",
            random_state=self.random_state,
        )
        
        agent = DataUnderstandingAgent(config)
        result = agent.run()
        
        print(f"  ✓ Data Understanding complete")
        print(f"    📊 Columns: {len(df.columns)}")
        print(f"    📈 Rows: {len(df)}")
        print(f"    💾 Output: {self.understanding_output}/")
        
        return df, result
    
    def _run_cleaning(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Step 2: Run DataCleaningAgent
        
        INPUT: Raw DataFrame from Step 1 ← Data Consistency Point
        OUTPUT: Cleaned DataFrame + Cleaning report
        
        PURPOSE: Remove duplicates, handle missing values, remove anomalies,
                 preserve special formats, ensure consistency
        """
        print("  ⏳ Initializing DataCleaningAgent...")
        agent = DataCleaningAgent(
            name="DataCleaner",
            output_dir=str(self.cleaning_output)
        )
        
        print(f"  ⏳ Input data shape: {self.raw_data.shape}")
        cleaned_df = agent.execute(self.raw_data)  # ← Uses output from Step 1
        print(f"  ✓ Output data shape: {cleaned_df.shape}")
        
        # Save cleaned data for next step
        cleaned_path = self.cleaning_output / "cleaned_data.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        
        report = agent.get_cleaning_report()
        
        print(f"  ✓ Data Cleaning complete")
        print(f"    🗑️  Rows removed: {report['cleaning_summary']['rows_removed']}")
        print(f"    📈 Data retention: {report['cleaning_summary']['data_retention_percentage']:.1f}%")
        print(f"    💾 Output: {self.cleaning_output}/")
        
        return cleaned_df, report
    
    def _run_feature_engineering(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Step 3: Run FeatureEngineeringAgent
        
        INPUT: Cleaned DataFrame from Step 2 ← Data Consistency Point
        OUTPUT: X_train, X_test, y_train, y_test + Feature summary
        
        PURPOSE: Handle missing values, scale numeric features, encode categorical,
                 handle rare categories, create meaningful features
        """
        print("  ⏳ Initializing FeatureEngineeringAgent...")
        
        config = FeatureEngineeringConfig(
            target_column=self.target_column,
            problem_type=self.problem_type,
            task_description="AutoDS Feature Engineering",
            save_artifacts=True,
            output_dir=str(self.feature_output),
            random_state=self.random_state,
        )
        
        agent = FeatureEngineeringAgent(config)
        
        print(f"  ⏳ Input data shape: {self.cleaned_data.shape}")
        X, y, summary = agent.run(self.cleaned_data)  # ← Uses output from Step 2
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y if self.problem_type == "classification" else None
        )
        
        # Save processed datasets for next step
        X_train.to_csv(self.feature_output / "X_train.csv", index=False)
        X_test.to_csv(self.feature_output / "X_test.csv", index=False)
        y_train.to_csv(self.feature_output / "y_train.csv", index=False, header=['target'])
        y_test.to_csv(self.feature_output / "y_test.csv", index=False, header=['target'])
        
        print(f"  ✓ Feature Engineering complete")
        print(f"    📊 Training shape: {X_train.shape}")
        print(f"    📊 Test shape: {X_test.shape}")
        print(f"    ✨ Features: {X_train.shape[1]}")
        print(f"    💾 Output: {self.feature_output}/")
        
        return X_train, X_test, y_train, y_test, summary
    
    def _run_modelling(self) -> Dict[str, Any]:
        """
        Step 4: Run ModellingAgent
        
        INPUT: X_train, X_test, y_train, y_test from Step 3 ← Data Consistency Point
        OUTPUT: Trained models + Modelling metadata
        
        PURPOSE: Train multiple models, evaluate on test set, track metrics,
                 identify best model candidates
        """
        print("  ⏳ Initializing ModellingAgent...")
        
        config = ModellingConfig(
            target_column=self.target_column,
            problem_type=self.problem_type,
            task_description="AutoDS Modelling",
            output_dir=str(self.modelling_output),
            random_state=self.random_state,
        )
        
        agent = ModellingAgent(config)
        
        print(f"  ⏳ Training set shape: {self.X_train.shape}")
        print(f"  ⏳ Test set shape: {self.X_test.shape}")
        
        # Use outputs from Step 3
        result = agent.run(
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test
        )
        
        print(f"  ✓ Modelling complete")
        print(f"    🤖 Models trained: {result.get('model_count', 'N/A')}")
        print(f"    🏆 Best model: {result.get('best_model_name', 'N/A')}")
        print(f"    💾 Output: {self.modelling_output}/")
        
        return result
    
    def _run_evaluation(self) -> Dict[str, Any]:
        """
        Step 5: Run EvaluationAgent
        
        INPUT: Modelling artifacts from Step 4 ← Data Consistency Point
        OUTPUT: Model comparison + Best model selection
        
        PURPOSE: Compare all trained models, provide selection evidence,
                 summarize performance metrics
        """
        print("  ⏳ Initializing EvaluationAgent...")
        
        config = EvaluationConfig(
            modelling_output_dir=str(self.modelling_output),
            output_dir=str(self.evaluation_output),
            save_artifacts=True,
        )
        
        agent = EvaluationAgent(config)
        
        # Use outputs from Step 4
        result = agent.run()
        
        print(f"  ✓ Evaluation complete")
        print(f"    🏆 Best model: {result.get('best_model_name', 'N/A')}")
        print(f"    📈 Primary metric: {result.get('primary_metric', 'N/A')}")
        print(f"    💾 Output: {self.evaluation_output}/")
        
        return result
    
    def _run_report_generation(self) -> Dict[str, Any]:
        """
        Step 6: Run ReportGenerator
        
        INPUT: All previous results and artifacts ← Data Consistency Point
        OUTPUT: Comprehensive final report
        
        PURPOSE: Summarize entire pipeline, present findings, make recommendations
        """
        print("  ⏳ Initializing ReportGenerator...")
        
        config = ReportGeneratorConfig(
            understanding_output_dir=str(self.understanding_output),
            cleaning_output_dir=str(self.cleaning_output),
            feature_output_dir=str(self.feature_output),
            modelling_output_dir=str(self.modelling_output),
            evaluation_output_dir=str(self.evaluation_output),
            output_dir=str(self.report_output),
        )
        
        agent = ReportGenerator(config)
        
        # Use all outputs from previous steps
        result = agent.run()
        
        print(f"  ✓ Report generation complete")
        print(f"    📄 HTML Report: {self.report_output}/report.html")
        print(f"    📊 JSON Report: {self.report_output}/report.json")
        print(f"    💾 Output: {self.report_output}/")
        
        return result
    
    def _print_summary(self):
        """Print a comprehensive summary of the pipeline execution"""
        print("\n" + "=" * 80)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        print("\n🔄 Data Progression Through Pipeline:")
        print(f"  1️⃣  Raw Data:          {self.raw_data.shape if self.raw_data is not None else 'N/A'}")
        print(f"  2️⃣  Cleaned Data:      {self.cleaned_data.shape if self.cleaned_data is not None else 'N/A'}")
        print(f"  3️⃣  Training Set:      {self.X_train.shape if self.X_train is not None else 'N/A'}")
        print(f"  4️⃣  Test Set:          {self.X_test.shape if self.X_test is not None else 'N/A'}")
        
        print("\n📁 Output Artifacts Generated:")
        print(f"  📂 Understanding:        {self.understanding_output}")
        print(f"  📂 Cleaning:             {self.cleaning_output}")
        print(f"  📂 Feature Engineering:  {self.feature_output}")
        print(f"  📂 Modelling:            {self.modelling_output}")
        print(f"  📂 Evaluation:           {self.evaluation_output}")
        print(f"  📂 Reports:              {self.report_output}")
        
        print("\n🎯 Pipeline Results:")
        if self.cleaning_report:
            print(f"  📊 Data Retention:       {self.cleaning_report['cleaning_summary']['data_retention_percentage']:.1f}%")
        if self.evaluation_result:
            print(f"  🏆 Best Model:           {self.evaluation_result.get('best_model_name', 'N/A')}")
            print(f"  📈 Primary Metric:       {self.evaluation_result.get('primary_metric', 'N/A')}")
        
        print("\n" + "=" * 80)


def main():
    """Example usage of the AutoDS Pipeline"""
    
    # Initialize pipeline
    pipeline = AutoDSPipeline(
        data_path="your_data.csv",  # ← CHANGE THIS: Path to your CSV file
        target_column="target",      # ← CHANGE THIS: Your target column name
        problem_type="classification",  # or "regression"
        output_dir="./autods_pipeline_output",
        random_state=42,
    )
    
    # Execute complete pipeline
    result = pipeline.run()
    
    # Print final summary
    pipeline._print_summary()
    
    return result


if __name__ == "__main__":
    result = main()
