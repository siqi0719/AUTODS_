import pandas as pd
import numpy as np
from typing import Union, Any, Dict, Optional
import re
import json
from pathlib import Path
from datetime import datetime


class DataCleaningAgent:
    """
    Data Cleaning Agent

    Handles data cleaning tasks:
    - Remove duplicates
    - Handle missing values (only for numeric columns)
    - Handle anomalies (only for numeric columns)
    - Data type conversion (only for convertible columns)
    - Data consistency (preserve special formats like phone numbers)
    """

    def __init__(self, name: str = "DataCleaner", output_dir: Optional[str] = None):
        self.name = name
        self.execution_log = []
        self.numeric_cols = []  # Track numeric columns
        self.categorical_cols = []  # Track categorical columns
        self.special_cols = []  # Track special format columns (e.g., phone numbers)
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cleaning_report = {}  # Store JSON report

    def execute(self, input_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Execute data cleaning

        Args:
            input_data: Path to CSV file or pandas DataFrame

        Returns:
            Cleaned pandas DataFrame
        """
        # Load data
        if isinstance(input_data, str):
            df = pd.read_csv(input_data)
        else:
            df = input_data.copy()

        print(f"[{self.name}] Input shape: {df.shape}")
        print(f"[{self.name}] Column info:")
        self._print_column_info(df)
        original_shape = df.shape

        # Step 0: Identify column types
        self._identify_column_types(df)

        # Step 1: Remove duplicates
        df = self._remove_duplicates(df)

        # Step 2: Handle missing values (numeric columns only)
        df = self._handle_missing_values(df)

        # Step 3: Handle anomalies (numeric columns only)
        df = self._handle_anomalies(df)

        # Step 4: Preserve special formats (e.g., phone numbers)
        df = self._preserve_special_formats(df)

        # Step 5: Ensure consistency (clean only necessary parts)
        df = self._ensure_consistency(df)

        print(f"\n[{self.name}] Output shape: {df.shape}")
        self._print_column_info(df)

        self.execution_log.append({
            'original_shape': original_shape,
            'final_shape': df.shape,
            'rows_removed': original_shape[0] - df.shape[0],
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'special_cols': self.special_cols
        })

        # Generate JSON report automatically
        self._generate_json_report(original_shape, df)

        return df

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """
        Identify column types: numeric, categorical, special format
        """
        print(f"\n[{self.name}] Identifying column types...")

        for col in df.columns:
            # First check if it's a special format (phone number, etc.)
            if self._is_special_format(df[col]):
                self.special_cols.append(col)
                print(f"  - {col}: SPECIAL FORMAT (preserve as-is)")

            # Check if it's already numeric type
            elif df[col].dtype in ['float64', 'int64']:
                self.numeric_cols.append(col)
                print(f"  - {col}: NUMERIC (will be cleaned)")

            # Try to convert to numeric
            elif self._can_convert_to_numeric(df[col]):
                self.numeric_cols.append(col)
                print(f"  - {col}: CAN BE NUMERIC (will convert then clean)")

            # Otherwise treat as categorical
            else:
                self.categorical_cols.append(col)
                print(f"  - {col}: CATEGORICAL (preserve as-is)")

    def _is_special_format(self, series: pd.Series) -> bool:
        """
        Check if column is a special format (phone number, email, etc.)
        """
        # Check non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        # Phone number patterns: 11 digits, may contain - or space
        phone_pattern = r'^1[3-9]\d{9}$|^\+?1?\d{10,11}$|^\d{3}-\d{3}-\d{4}$|^\d{3} \d{3} \d{4}$'

        # Check if most of the values match phone number format
        phone_count = sum(1 for val in non_null.astype(str)
                          if isinstance(val, str) and re.match(phone_pattern, str(val).strip()))

        if len(non_null) > 0 and phone_count / len(non_null) > 0.8:
            return True

        return False

    def _can_convert_to_numeric(self, series: pd.Series) -> bool:
        """
        Check if column can be converted to numeric type
        """
        if series.dtype in ['float64', 'int64']:
            return True

        # Try to convert, if most can be successfully converted, consider it numeric
        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        try:
            converted = pd.to_numeric(non_null, errors='coerce')
            conversion_rate = (converted.notna().sum() / len(non_null))
            # If we can convert more than 80%, consider this a numeric column
            return conversion_rate > 0.8
        except:
            return False

    def _print_column_info(self, df: pd.DataFrame) -> None:
        """Print dataframe column information"""
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            null_count = df[col].isna().sum()
            dtype = df[col].dtype
            print(f"  {col}: dtype={dtype}, non_null={non_null_count}, null={null_count}")

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        print(f"\n[{self.name}] Removing duplicates...")
        duplicates_before = len(df)
        df = df.drop_duplicates()
        duplicates_removed = duplicates_before - len(df)
        if duplicates_removed > 0:
            print(f"[{self.name}] ✓ Removed {duplicates_removed} duplicate rows")
        else:
            print(f"[{self.name}] ✓ No duplicates found")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values - only process numeric columns
        """
        print(f"\n[{self.name}] Handling missing values (numeric columns only)...")

        # Only process columns identified as numeric
        for col in self.numeric_cols:
            if col not in df.columns:
                continue

            if df[col].isna().sum() > 0:
                # If not numeric yet, convert first
                if df[col].dtype not in ['float64', 'int64']:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass

                # Fill missing values with median
                if df[col].dtype in ['float64', 'int64']:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"  ✓ {col}: Filled {df[col].isna().sum()} missing values with median ({median_val:.2f})")

        return df

    def _handle_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle anomalies using IQR method - only process numeric columns
        """
        print(f"\n[{self.name}] Handling anomalies (numeric columns only)...")

        rows_before = len(df)

        for col in self.numeric_cols:
            if col not in df.columns:
                continue

            # Ensure column is numeric type
            if df[col].dtype not in ['float64', 'int64']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    continue

            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:  # Skip if IQR is zero
                continue

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            anomaly_count = len(anomalies)

            # Remove anomalies
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            if anomaly_count > 0:
                print(f"  ✓ {col}: Removed {anomaly_count} anomalies (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")

        rows_after = len(df)
        total_removed = rows_before - rows_after
        if total_removed > 0:
            print(f"\n[{self.name}] Total anomaly rows removed: {total_removed}")

        return df

    def _preserve_special_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preserve special format columns (e.g., phone numbers)
        """
        print(f"\n[{self.name}] Preserving special formats...")

        for col in self.special_cols:
            if col not in df.columns:
                continue

            print(f"  ✓ {col}: Preserving as special format (e.g., phone numbers)")
            # Only trim, don't do other transformations
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()

        return df

    def _ensure_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure data consistency - only standardize categorical columns
        """
        print(f"\n[{self.name}] Ensuring consistency...")

        # Only process categorical columns to make them lowercase and trim
        for col in self.categorical_cols:
            if col not in df.columns:
                continue

            if df[col].dtype == 'object':
                # Remove leading/trailing whitespace and convert to lowercase
                df[col] = df[col].astype(str).str.strip().str.lower()
                print(f"  ✓ {col}: Trimmed whitespace and converted to lowercase")

        return df

    def _generate_json_report(self, original_shape: tuple, cleaned_df: pd.DataFrame) -> None:
        """
        Generate comprehensive JSON report of cleaning process
        """
        self.cleaning_report = {
            "metadata": {
                "agent_name": self.name,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            },
            "input_data": {
                "shape": {
                    "rows": int(original_shape[0]),
                    "columns": int(original_shape[1])
                },
                "total_cells": int(original_shape[0] * original_shape[1])
            },
            "output_data": {
                "shape": {
                    "rows": int(cleaned_df.shape[0]),
                    "columns": int(cleaned_df.shape[1])
                },
                "total_cells": int(cleaned_df.shape[0] * cleaned_df.shape[1])
            },
            "cleaning_summary": {
                "rows_removed": int(original_shape[0] - cleaned_df.shape[0]),
                "rows_retained": int(cleaned_df.shape[0]),
                "rows_removed_percentage": round((original_shape[0] - cleaned_df.shape[0]) / original_shape[0] * 100, 2) if original_shape[0] > 0 else 0,
                "data_retention_percentage": round(cleaned_df.shape[0] / original_shape[0] * 100, 2) if original_shape[0] > 0 else 0
            },
            "column_classification": {
                "numeric_columns": self.numeric_cols,
                "categorical_columns": self.categorical_cols,
                "special_format_columns": self.special_cols,
                "column_type_summary": {
                    "numeric_count": len(self.numeric_cols),
                    "categorical_count": len(self.categorical_cols),
                    "special_format_count": len(self.special_cols)
                }
            },
            "data_quality_metrics": {
                "original": self._calculate_quality_metrics(cleaned_df, "original"),
                "cleaned": self._calculate_quality_metrics(cleaned_df, "cleaned")
            },
            "cleaning_operations": {
                "duplicates_removed": 0,
                "missing_values_handled": len(self.numeric_cols) > 0,
                "anomalies_removed": "IQR method applied to numeric columns",
                "special_formats_preserved": self.special_cols
            }
        }
        
        # Save to JSON file
        self._save_json_report()
    
    def _calculate_quality_metrics(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        return {
            "completeness": round(df.notna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2) if (df.shape[0] * df.shape[1]) > 0 else 0,
            "duplicates": int(df.duplicated().sum()),
            "null_values": int(df.isna().sum().sum()),
            "null_percentage": round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2) if (df.shape[0] * df.shape[1]) > 0 else 0
        }
    
    def _save_json_report(self) -> None:
        """Save cleaning report to JSON file"""
        json_path = self.output_dir / "cleaning_report.json"
        
        # Convert to serializable format
        report_serializable = json.loads(json.dumps(self.cleaning_report, default=str))
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n[{self.name}] ✓ JSON report saved to: {json_path}")
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get cleaning report as dictionary"""
        return self.cleaning_report

    def get_log(self):
        """Get execution log"""
        return self.execution_log

    def get_summary(self) -> dict:
        """
        Get cleaning summary report
        """
        if not self.execution_log:
            return {}

        latest_log = self.execution_log[-1]
        return {
            'original_shape': latest_log['original_shape'],
            'final_shape': latest_log['final_shape'],
            'rows_removed': latest_log['rows_removed'],
            'numeric_columns': latest_log['numeric_cols'],
            'categorical_columns': latest_log['categorical_cols'],
            'special_format_columns': latest_log['special_cols'],
            'data_quality': {
                'rows_retained': latest_log['final_shape'][0] / latest_log['original_shape'][0] * 100,
                'rows_removed_pct': latest_log['rows_removed'] / latest_log['original_shape'][0] * 100
            }
        }