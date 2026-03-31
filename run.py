#!/usr/bin/env python3
"""
AUTODS 0.0.1
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
# ============================================================================
# step 0：cleaning old module
# ============================================================================
'''
print("\n[*] old module removing...")

modules_to_remove = []
for module_name in list(sys.modules.keys()):
    if 'langchain' in module_name.lower():
        modules_to_remove.append(module_name)
        print(f"  - remove: {module_name}")
    elif 'multi_agent_report_generator' in module_name and 'standalone' not in module_name:
        modules_to_remove.append(module_name)
        print(f"  - remove: {module_name}")

for module_name in modules_to_remove:
    del sys.modules[module_name]

print("[✓] old module cleaned\n")



# ============================================================================
# step 1: generating sample dataset
# ============================================================================
# Describe your task in plain English
print("📊 step 1: generate sample dataset...")

np.random.seed(42)
n_samples = 100

data = {
    'customer_id': np.arange(1, n_samples + 1),
    'age': np.random.randint(20, 70, n_samples),
    'income': np.random.randint(30000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'purchase_history': np.random.randint(0, 100, n_samples),
    'target': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)


df.loc[np.random.choice(len(df), 5, replace=False), 'age'] = np.nan
df.loc[np.random.choice(len(df), 3, replace=False), 'income'] = 999999

print(f"✓  {len(df)} samples generated")


df.to_csv("_example_data_temp.csv", index=False)
print("✓ data saved\n")
'''


# ============================================================================
# step 2：loading Pipeline
# ============================================================================

print("⏳ step 2: loading Pipeline...")

try:
    from autods_implementation_guide import PipelineConfig, DataSciencePipeline
    print("✓ Pipeline loaded\n")
except ImportError as e:
    print(f"❌ failed: {e}")
    print("\nPlease ensure that this script is run in the directory containing all agent files.\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# step 3：config
# ============================================================================

print("🔧 step 3：config...")

config = PipelineConfig()
config.data_path = "adult.csv"
config.csv_sep = ","             # Bank Marketing CSV uses semicolon separator
config.random_state = 42

# ── Planner settings ─────────────────────────────────────────────────────────
# Describe the business objective in natural language.
# The Planner Agent will infer target_column, problem_type, candidate models, etc.
# If you already know target_column / problem_type, set them here and the Planner
# will keep your values and only fill in the rest.
config.business_description = (
    "This is the UCI Adult (Census Income) dataset extracted from the 1994 US Census database. "
    "The goal is to predict whether a person's annual income exceeds $50,000 (binary classification). "
    "The target column is 'income' (values: '>50K' or '<=50K'). "
    "Input features include demographic and employment information: "
    "age, workclass, fnlwgt (census sampling weight), education, education-num, "
    "marital-status, occupation, relationship, race, sex, "
    "capital-gain, capital-loss, hours-per-week, and native-country. "
    "The dataset has approximately 48,842 instances with a mix of continuous and categorical features. "
    "Some records contain unknown values denoted by '?' — these should be treated as a separate "
    "category for categorical features rather than dropped. "
    "The class distribution is heavily imbalanced: approximately 24% earn >50K and 76% earn <=50K, "
    "so ROC-AUC and F1-score are preferred over accuracy as primary metrics. "
    "The positive class is '>50K'. "
    "Interpretability is important as this is a socioeconomic prediction task — "
    "Logistic Regression, Decision Tree, and Random Forest are preferred. "
    "Note that 'fnlwgt' is a census sampling weight, not a predictive demographic feature, "
    "and may be excluded or treated with caution."
)
config.use_planner = True   # set False to skip Stage 0 entirely

# Optional: override Planner's inference if you already know these values.
config.target_column = "15"             # keep None to let Planner infer
config.problem_type = "classification" # keep None to let Planner infer
# ─────────────────────────────────────────────────────────────────────────────

print(f"✓ finished: {config.data_path} | target: {config.target_column}\n")

# ============================================================================
# step 4：running Pipeline
# ============================================================================

print("="*80)
print("🚀 running Pipeline")
print("="*80 + "\n")

pipeline = DataSciencePipeline(config)

try:

    result = pipeline.run_complete_pipeline()
    
    print("\n" + "="*80)
    print("✅ finished pipeline！")
    print("="*80)
    print(f"\n📁 output: {config.output_base_dir}\n")
    
    print("Pipeline flowing:")
    pipeline.print_data_lineage()
    
    print("\n" + "="*80)
    print("🎉 6 steps finished")
    print("="*80 + "\n")
    
    print("📊 file generated...：")
    print(f"  - report: {config.output_base_dir / '06_reports' / 'report.md'}")
    print(f"  - JSON: {config.output_base_dir / '06_reports' / 'report.json'}\n")
    
except Exception as e:
    print(f"\n❌ Pipeline running failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
