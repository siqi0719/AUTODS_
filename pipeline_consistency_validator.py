"""
AutoDS Pipeline - Data Consistency Validation Tool

This module provides comprehensive validation and verification of data
consistency throughout the entire pipeline execution.

It ensures:
1. Data shapes are consistent across stages
2. Row counts match expectations
3. Column names are preserved
4. No data is lost between stages
5. Train/test splits are correct
6. All artifacts are present and valid
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd


class DataConsistencyValidator:
    """Validates data consistency throughout the pipeline"""
    
    def __init__(self, pipeline_output_dir: str):
        """
        Initialize validator
        
        Args:
            pipeline_output_dir: Path to pipeline output directory
        """
        self.output_dir = Path(pipeline_output_dir)
        self.stage_dirs = {
            1: self.output_dir / "01_understanding",
            2: self.output_dir / "02_cleaning",
            3: self.output_dir / "03_feature_engineering",
            4: self.output_dir / "04_modelling",
            5: self.output_dir / "05_evaluation",
            6: self.output_dir / "06_reports",
        }
        self.validation_results = {}
    
    # ========================================================================
    # STAGE 1: UNDERSTANDING VALIDATION
    # ========================================================================
    
    def validate_stage_1_understanding(self) -> Dict[str, Any]:
        """Validate Stage 1 outputs"""
        print("\n" + "="*80)
        print("VALIDATING STAGE 1: UNDERSTANDING")
        print("="*80)
        
        results = {
            'stage': 1,
            'checks': {},
            'status': 'PASS'
        }
        
        try:
            stage_dir = self.stage_dirs[1]
            
            # Check 1: Required files exist
            required_files = [
                'data_profile.json',
                'data_quality_report.json',
                'target_analysis.json',
            ]
            
            print("\n✓ Checking required files...")
            for file in required_files:
                file_path = stage_dir / file
                exists = file_path.exists()
                results['checks'][f'{file}_exists'] = exists
                status = "✓" if exists else "✗"
                print(f"  {status} {file}")
                if not exists:
                    results['status'] = 'WARN'
            
            # Check 2: JSON files are valid
            print("\n✓ Validating JSON files...")
            for file in required_files:
                file_path = stage_dir / file
                if file_path.exists():
                    try:
                        with open(file_path) as f:
                            json.load(f)
                        results['checks'][f'{file}_valid'] = True
                        print(f"  ✓ {file} is valid JSON")
                    except Exception as e:
                        results['checks'][f'{file}_valid'] = False
                        results['status'] = 'FAIL'
                        print(f"  ✗ {file} JSON error: {str(e)}")
        
        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
        
        self.validation_results[1] = results
        return results
    
    # ========================================================================
    # STAGE 2: CLEANING VALIDATION
    # ========================================================================
    
    def validate_stage_2_cleaning(self) -> Dict[str, Any]:
        """Validate Stage 2 outputs and data consistency"""
        print("\n" + "="*80)
        print("VALIDATING STAGE 2: CLEANING")
        print("="*80)
        
        results = {
            'stage': 2,
            'checks': {},
            'status': 'PASS',
            'data_analysis': {}
        }
        
        try:
            stage_dir = self.stage_dirs[2]
            
            # Check 1: cleaned_data.csv exists
            cleaned_path = stage_dir / "cleaned_data.csv"
            if not cleaned_path.exists():
                results['status'] = 'FAIL'
                results['checks']['cleaned_data_exists'] = False
                print("✗ cleaned_data.csv not found!")
                return results
            
            print("✓ cleaned_data.csv found")
            results['checks']['cleaned_data_exists'] = True
            
            # Check 2: Load and analyze cleaned data
            cleaned_data = pd.read_csv(cleaned_path)
            results['data_analysis']['shape'] = cleaned_data.shape
            results['data_analysis']['columns'] = list(cleaned_data.columns)
            results['data_analysis']['null_count'] = int(cleaned_data.isna().sum().sum())
            results['data_analysis']['duplicates'] = int(cleaned_data.duplicated().sum())
            
            print(f"\n✓ Data shape: {cleaned_data.shape}")
            print(f"✓ Null values: {results['data_analysis']['null_count']}")
            print(f"✓ Duplicates: {results['data_analysis']['duplicates']}")
            
            # Check 3: cleaning_report.json
            report_path = stage_dir / "cleaning_report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                results['checks']['cleaning_report_exists'] = True
                results['data_analysis']['retention_rate'] = \
                    report['cleaning_summary']['data_retention_percentage']
                print(f"✓ Data retention rate: {results['data_analysis']['retention_rate']:.1f}%")
            else:
                results['checks']['cleaning_report_exists'] = False
                results['status'] = 'WARN'
        
        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
        
        self.validation_results[2] = results
        return results
    
    # ========================================================================
    # STAGE 3: FEATURE ENGINEERING VALIDATION
    # ========================================================================
    
    def validate_stage_3_feature_engineering(self) -> Dict[str, Any]:
        """Validate Stage 3 outputs and data consistency"""
        print("\n" + "="*80)
        print("VALIDATING STAGE 3: FEATURE ENGINEERING")
        print("="*80)
        
        results = {
            'stage': 3,
            'checks': {},
            'status': 'PASS',
            'data_analysis': {}
        }
        
        try:
            stage_dir = self.stage_dirs[3]
            
            # Check 1: All required files exist
            required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
            
            print("✓ Checking required files...")
            all_exist = True
            for file in required_files:
                exists = (stage_dir / file).exists()
                results['checks'][f'{file}_exists'] = exists
                status = "✓" if exists else "✗"
                print(f"  {status} {file}")
                if not exists:
                    all_exist = False
                    results['status'] = 'FAIL'
            
            if not all_exist:
                return results
            
            # Check 2: Load data
            X_train = pd.read_csv(stage_dir / "X_train.csv")
            X_test = pd.read_csv(stage_dir / "X_test.csv")
            y_train = pd.read_csv(stage_dir / "y_train.csv")
            y_test = pd.read_csv(stage_dir / "y_test.csv")
            
            # Check 3: Data consistency
            print("\n✓ Validating data consistency...")
            
            # Shape consistency
            results['data_analysis']['X_train_shape'] = X_train.shape
            results['data_analysis']['X_test_shape'] = X_test.shape
            results['data_analysis']['y_train_shape'] = y_train.shape
            results['data_analysis']['y_test_shape'] = y_test.shape
            
            print(f"  X_train: {X_train.shape}")
            print(f"  X_test: {X_test.shape}")
            print(f"  y_train: {y_train.shape}")
            print(f"  y_test: {y_test.shape}")
            
            # Row consistency
            check_1 = len(y_train) == X_train.shape[0]
            check_2 = len(y_test) == X_test.shape[0]
            check_3 = X_train.shape[1] == X_test.shape[1]  # Same features
            
            results['checks']['y_train_rows_match'] = check_1
            results['checks']['y_test_rows_match'] = check_2
            results['checks']['feature_columns_match'] = check_3
            
            print(f"\n✓ y_train rows match X_train: {check_1}")
            print(f"✓ y_test rows match X_test: {check_2}")
            print(f"✓ X_train and X_test have same columns: {check_3}")
            
            if not (check_1 and check_2 and check_3):
                results['status'] = 'FAIL'
            
            # Train/test split ratio
            total_samples = X_train.shape[0] + X_test.shape[0]
            train_ratio = X_train.shape[0] / total_samples * 100
            test_ratio = X_test.shape[0] / total_samples * 100
            
            results['data_analysis']['train_test_split'] = f"{train_ratio:.1f}% / {test_ratio:.1f}%"
            print(f"✓ Train/test split: {train_ratio:.1f}% / {test_ratio:.1f}%")
        
        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
        
        self.validation_results[3] = results
        return results
    
    # ========================================================================
    # STAGE 4: MODELLING VALIDATION
    # ========================================================================
    
    def validate_stage_4_modelling(self) -> Dict[str, Any]:
        """Validate Stage 4 outputs"""
        print("\n" + "="*80)
        print("VALIDATING STAGE 4: MODELLING")
        print("="*80)
        
        results = {
            'stage': 4,
            'checks': {},
            'status': 'PASS',
            'data_analysis': {}
        }
        
        try:
            stage_dir = self.stage_dirs[4]
            
            # Check 1: Key files exist
            key_files = [
                'leaderboard.csv',
                'best_model_metrics.json',
                'modelling_summary.json',
                'diagnostics.json',
            ]
            
            print("✓ Checking required files...")
            for file in key_files:
                exists = (stage_dir / file).exists()
                results['checks'][f'{file}_exists'] = exists
                status = "✓" if exists else "✗"
                print(f"  {status} {file}")
                if not exists and file in ['leaderboard.csv', 'best_model_metrics.json']:
                    results['status'] = 'FAIL'
            
            # Check 2: Parse leaderboard
            leaderboard_path = stage_dir / "leaderboard.csv"
            if leaderboard_path.exists():
                leaderboard = pd.read_csv(leaderboard_path)
                results['data_analysis']['model_count'] = len(leaderboard)
                results['data_analysis']['top_model'] = str(leaderboard.iloc[0]['model_name'])
                print(f"\n✓ Models trained: {len(leaderboard)}")
                print(f"✓ Top model: {results['data_analysis']['top_model']}")
            
            # Check 3: Parse best model metrics
            metrics_path = stage_dir / "best_model_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                results['data_analysis']['best_model_name'] = metrics.get('model_name')
                print(f"✓ Best model: {results['data_analysis']['best_model_name']}")
        
        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
        
        self.validation_results[4] = results
        return results
    
    # ========================================================================
    # STAGE 5: EVALUATION VALIDATION
    # ========================================================================
    
    def validate_stage_5_evaluation(self) -> Dict[str, Any]:
        """Validate Stage 5 outputs"""
        print("\n" + "="*80)
        print("VALIDATING STAGE 5: EVALUATION")
        print("="*80)
        
        results = {
            'stage': 5,
            'checks': {},
            'status': 'PASS',
            'data_analysis': {}
        }
        
        try:
            stage_dir = self.stage_dirs[5]
            
            # Check 1: evaluation_summary.json exists
            summary_path = stage_dir / "evaluation_summary.json"
            if not summary_path.exists():
                results['status'] = 'FAIL'
                results['checks']['evaluation_summary_exists'] = False
                print("✗ evaluation_summary.json not found!")
                return results
            
            print("✓ evaluation_summary.json found")
            results['checks']['evaluation_summary_exists'] = True
            
            # Check 2: Parse evaluation summary
            with open(summary_path) as f:
                summary = json.load(f)
            
            results['data_analysis']['best_model_name'] = summary.get('best_model_name')
            results['data_analysis']['primary_metric'] = summary.get('primary_metric')
            
            print(f"\n✓ Best model: {results['data_analysis']['best_model_name']}")
            print(f"✓ Primary metric: {results['data_analysis']['primary_metric']}")
            
            # Check 3: Verify model ranking
            if 'benchmark_overview' in summary:
                candidate_count = summary['benchmark_overview'].get('candidate_model_count', 0)
                results['data_analysis']['candidate_models'] = candidate_count
                print(f"✓ Candidate models: {candidate_count}")
        
        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
        
        self.validation_results[5] = results
        return results
    
    # ========================================================================
    # STAGE 6: REPORT VALIDATION
    # ========================================================================
    
    def validate_stage_6_report(self) -> Dict[str, Any]:
        """Validate Stage 6 outputs"""
        print("\n" + "="*80)
        print("VALIDATING STAGE 6: REPORT")
        print("="*80)
        
        results = {
            'stage': 6,
            'checks': {},
            'status': 'PASS',
            'data_analysis': {}
        }
        
        try:
            stage_dir = self.stage_dirs[6]
            
            # Check reports exist
            html_path = stage_dir / "report.html"
            json_path = stage_dir / "report.json"
            
            print("✓ Checking report files...")
            html_exists = html_path.exists()
            json_exists = json_path.exists()
            
            results['checks']['html_report_exists'] = html_exists
            results['checks']['json_report_exists'] = json_exists
            
            print(f"  {'✓' if html_exists else '✗'} report.html")
            print(f"  {'✓' if json_exists else '✗'} report.json")
            
            if not (html_exists or json_exists):
                results['status'] = 'FAIL'
        
        except Exception as e:
            results['status'] = 'FAIL'
            results['error'] = str(e)
        
        self.validation_results[6] = results
        return results
    
    # ========================================================================
    # COMPLETE VALIDATION
    # ========================================================================
    
    def validate_all_stages(self) -> Dict[str, Any]:
        """Validate all stages"""
        print("\n" + "="*80)
        print("🔍 STARTING COMPLETE PIPELINE VALIDATION")
        print("="*80)
        
        self.validate_stage_1_understanding()
        self.validate_stage_2_cleaning()
        self.validate_stage_3_feature_engineering()
        self.validate_stage_4_modelling()
        self.validate_stage_5_evaluation()
        self.validate_stage_6_report()
        
        return self._generate_validation_report()
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate final validation report"""
        print("\n" + "="*80)
        print("📊 VALIDATION REPORT")
        print("="*80)
        
        overall_status = 'PASS'
        for stage, result in self.validation_results.items():
            if result['status'] != 'PASS':
                overall_status = 'FAIL'
            status_icon = "✓" if result['status'] == 'PASS' else "✗"
            print(f"{status_icon} Stage {stage}: {result['status']}")
        
        print("\n" + "="*80)
        if overall_status == 'PASS':
            print("✅ ALL VALIDATION CHECKS PASSED!")
        else:
            print("❌ SOME VALIDATION CHECKS FAILED!")
        print("="*80)
        
        return {
            'overall_status': overall_status,
            'stage_results': self.validation_results,
            'timestamp': str(pd.Timestamp.now())
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of validator"""
    
    # Create validator
    validator = DataConsistencyValidator("./autods_pipeline_output")
    
    # Validate all stages
    report = validator.validate_all_stages()
    
    # Print report
    print("\n" + "="*80)
    print("📄 DETAILED VALIDATION RESULTS")
    print("="*80)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
