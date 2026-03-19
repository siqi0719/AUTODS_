"""
AutoDS Integrated Pipeline

This module integrates DataUnderstandingAgent and DataCleaningAgent
to create a complete data preprocessing workflow.

Pipeline Flow:
1. DataUnderstandingAgent: Analyze and understand raw data
2. DataCleaningAgent: Clean data based on understanding
3. AutoDSPipeline: Orchestrate the entire workflow
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from data_understanding_agent import AgentConfig, DataUnderstandingAgent
from data_cleaning_agent import DataCleaningAgent


class AutoDSPipeline:
    """
    Automated Data Science Pipeline
    
    Orchestrates the complete data preprocessing workflow:
    - Understand raw data characteristics
    - Clean and preprocess the data
    - Generate comprehensive reports
    """
    
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize the AutoDS Pipeline
        
        Args:
            data_path: Path to input data file (CSV or Parquet)
            output_dir: Directory to save all outputs
            target_column: Name of target column (optional)
            problem_type: 'classification' or 'regression' (optional)
            dataset_name: Custom dataset name (optional)
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.target_column = target_column
        self.problem_type = problem_type
        self.dataset_name = dataset_name
        self.random_state = random_state
        
        # Create output subdirectories
        self.understanding_output = self.output_dir / "understanding"
        self.cleaning_output = self.output_dir / "cleaning"
        self.understanding_output.mkdir(parents=True, exist_ok=True)
        self.cleaning_output.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.understanding_result = None
        self.cleaned_data = None
        self.cleaning_summary = None
        self.pipeline_report = None
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline
        
        Returns:
            Dictionary with pipeline execution results
        """
        print("=" * 80)
        print("AutoDS Integrated Pipeline Started")
        print("=" * 80)
        
        # Step 1: Data Understanding
        print("\n[Step 1/3] Running Data Understanding Agent...")
        self.understanding_result = self._run_understanding_agent()
        
        # Step 2: Data Cleaning
        print("\n[Step 2/3] Running Data Cleaning Agent...")
        self.cleaned_data, self.cleaning_summary = self._run_cleaning_agent()
        
        # Step 3: Generate integrated report
        print("\n[Step 3/3] Generating integrated report...")
        self.pipeline_report = self._generate_integrated_report()
        
        print("\n" + "=" * 80)
        print("AutoDS Pipeline Completed Successfully")
        print("=" * 80)
        
        return self.pipeline_report
    
    def _run_understanding_agent(self) -> Dict[str, Any]:
        """Run DataUnderstandingAgent"""
        config = AgentConfig(
            data_path=self.data_path,
            output_dir=str(self.understanding_output),
            target_column=self.target_column,
            problem_type=self.problem_type,
            dataset_name=self.dataset_name,
            random_state=self.random_state,
        )
        
        agent = DataUnderstandingAgent(config)
        result = agent.run()
        
        print(f"✓ Understanding complete. Output: {result['output_dir']}")
        print(f"  Generated files: {', '.join(result['generated_files'])}")
        
        return result
    
    def _run_cleaning_agent(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Run DataCleaningAgent"""
        agent = DataCleaningAgent(name="AutoDS_Cleaner")
        
        # Execute cleaning
        cleaned_df = agent.execute(self.data_path)
        cleaning_summary = agent.get_summary()
        
        # Save cleaned data
        cleaned_path = self.cleaning_output / "cleaned_data.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        print(f"✓ Cleaning complete. Cleaned data saved to: {cleaned_path}")
        
        # Save cleaning summary
        summary_path = self.cleaning_output / "cleaning_summary.json"
        self._save_json(summary_path, cleaning_summary)
        print(f"✓ Cleaning summary saved to: {summary_path}")
        
        return cleaned_df, cleaning_summary
    
    def _generate_integrated_report(self) -> Dict[str, Any]:
        """Generate comprehensive integrated report"""
        
        # Load understanding results
        understanding_summary_path = self.understanding_output / "data_understanding_summary.json"
        with open(understanding_summary_path, 'r') as f:
            understanding_summary = json.load(f)
        
        # Create integrated report
        report = {
            "pipeline_name": "AutoDS Integrated Pipeline",
            "status": "completed",
            "data_source": str(self.data_path),
            "dataset_name": self.dataset_name or Path(self.data_path).stem,
            
            # Understanding Phase Results
            "understanding_phase": {
                "status": "completed",
                "output_directory": str(self.understanding_output),
                "summary": understanding_summary,
            },
            
            # Cleaning Phase Results
            "cleaning_phase": {
                "status": "completed",
                "output_directory": str(self.cleaning_output),
                "cleaned_data_path": str(self.cleaning_output / "cleaned_data.csv"),
                "summary": self.cleaning_summary,
            },
            
            # Integrated Insights
            "integrated_insights": self._generate_integrated_insights(),
            
            # Recommendations
            "recommendations": self._generate_recommendations(),
            
            # Next Steps
            "next_steps": self._generate_next_steps(),
            
            # Output Summary
            "output_files": {
                "understanding": [
                    str(self.understanding_output / "data_profile.json"),
                    str(self.understanding_output / "data_quality_report.json"),
                    str(self.understanding_output / "target_analysis.json"),
                    str(self.understanding_output / "data_understanding_summary.json"),
                    str(self.understanding_output / "data_understanding_metadata.json"),
                ],
                "cleaning": [
                    str(self.cleaning_output / "cleaned_data.csv"),
                    str(self.cleaning_output / "cleaning_summary.json"),
                ]
            }
        }
        
        # Save integrated report
        report_path = self.output_dir / "integrated_pipeline_report.json"
        self._save_json(report_path, report)
        print(f"✓ Integrated report saved to: {report_path}")
        
        return report
    
    def _generate_integrated_insights(self) -> Dict[str, Any]:
        """Generate insights from both understanding and cleaning phases"""
        
        insights = {
            "data_quality_improvement": {
                "original_rows": self.cleaning_summary.get('original_shape', (0, 0))[0],
                "cleaned_rows": self.cleaning_summary.get('final_shape', (0, 0))[0],
                "rows_removed": self.cleaning_summary.get('rows_removed', 0),
                "data_retention_rate": f"{self.cleaning_summary.get('data_quality', {}).get('rows_retained', 0):.2f}%"
            },
            
            "column_types": {
                "numeric_columns": self.cleaning_summary.get('numeric_columns', []),
                "categorical_columns": self.cleaning_summary.get('categorical_columns', []),
                "special_format_columns": self.cleaning_summary.get('special_format_columns', []),
            },
            
            "key_findings": [
                f"Identified {len(self.cleaning_summary.get('numeric_columns', []))} numeric columns for statistical analysis",
                f"Identified {len(self.cleaning_summary.get('categorical_columns', []))} categorical columns for encoding",
                f"Removed {self.cleaning_summary.get('rows_removed', 0)} rows with duplicates or anomalies",
            ]
        }
        
        return insights
    
    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations"""
        
        recommendations = [
            "Review the data understanding summary for feature relationships and target distribution",
            "Use cleaned data for downstream machine learning pipeline",
            "Consider the recommended_actions from data_quality_report for further improvements",
            "For classification: check class imbalance and consider stratified splitting",
            "For regression: validate target distribution and consider outlier handling",
        ]
        
        # Add specific recommendations based on cleaning
        if self.cleaning_summary.get('special_format_columns'):
            recommendations.append(
                f"Special format columns {self.cleaning_summary['special_format_columns']} "
                "were preserved. Consider custom encoding if needed."
            )
        
        return recommendations
    
    def _generate_next_steps(self) -> list[str]:
        """Generate next steps for the ML workflow"""
        
        next_steps = [
            "1. Review understanding artifacts in the 'understanding' directory",
            "2. Load cleaned data from cleaned_data.csv for further processing",
            "3. Apply feature engineering based on insights",
            f"4. Prepare data for {'classification' if self.problem_type == 'classification' else 'regression'} modeling",
            "5. Split data into train/validation/test sets",
            "6. Train baseline models and iterate",
        ]
        
        return next_steps
    
    @staticmethod
    def _save_json(path: Path, data: Dict[str, Any]) -> None:
        """Save dictionary as JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary"""
        return {
            "understanding_result": self.understanding_result,
            "cleaning_summary": self.cleaning_summary,
            "pipeline_report": self.pipeline_report,
        }
    
    def print_report(self) -> None:
        """Print human-readable pipeline report"""
        if not self.pipeline_report:
            print("Pipeline has not been executed yet")
            return
        
        print("\n" + "=" * 80)
        print("AUTODS INTEGRATED PIPELINE REPORT")
        print("=" * 80)
        
        print(f"\nDataset: {self.pipeline_report.get('dataset_name')}")
        print(f"Source: {self.pipeline_report.get('data_source')}")
        
        # Data Quality Insights
        insights = self.pipeline_report.get('integrated_insights', {})
        improvement = insights.get('data_quality_improvement', {})
        print(f"\nData Quality Summary:")
        print(f"  Original rows: {improvement.get('original_rows')}")
        print(f"  Cleaned rows: {improvement.get('cleaned_rows')}")
        print(f"  Rows removed: {improvement.get('rows_removed')}")
        print(f"  Data retention: {improvement.get('data_retention_rate')}")
        
        # Column types
        col_types = insights.get('column_types', {})
        print(f"\nColumn Classification:")
        print(f"  Numeric columns: {len(col_types.get('numeric_columns', []))}")
        print(f"  Categorical columns: {len(col_types.get('categorical_columns', []))}")
        print(f"  Special format columns: {len(col_types.get('special_format_columns', []))}")
        
        # Recommendations
        print(f"\nKey Recommendations:")
        for rec in self.pipeline_report.get('recommendations', [])[:3]:
            print(f"  • {rec}")
        
        # Output files
        output_files = self.pipeline_report.get('output_files', {})
        print(f"\nGenerated Artifacts:")
        print(f"  Understanding outputs: {len(output_files.get('understanding', []))} files")
        print(f"  Cleaning outputs: {len(output_files.get('cleaning', []))} files")
        
        print("\n" + "=" * 80)


def main():
    """Example usage of the AutoDS Pipeline"""
    
    # Configuration
    pipeline = AutoDSPipeline(
        data_path="mydata.csv",  # Change to your data path
        output_dir="./autods_output",
        target_column=None,  # Set your target column if available
        problem_type=None,  # "classification" or "regression"
        dataset_name="MyDataset",
    )
    
    # Run pipeline
    result = pipeline.run()
    
    # Print report
    pipeline.print_report()
    
    # Get cleaned data
    cleaned_data = pipeline.cleaned_data
    print(f"\nCleaned data shape: {cleaned_data.shape}")
    print(f"Cleaned data columns: {cleaned_data.columns.tolist()}")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
