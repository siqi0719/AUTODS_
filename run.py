#!/usr/bin/env python3
"""
AUTODS 0.0.1
"""
import httpx
http_client = httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))
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
config.data_path = "rental_training.csv"
# config.csv_sep is None by default — the pipeline auto-detects the separator.
# Override only when auto-detection fails, e.g.:
#   config.csv_sep = ";"   # for Bank Marketing and other semicolon-delimited files
config.random_state = 42

# ── Planner settings ─────────────────────────────────────────────────────────
# Describe the business objective in natural language.
# The Planner Agent will infer target_column, problem_type, candidate models, etc.
# If you already know target_column / problem_type, set them here and the Planner
# will keep your values and only fill in the rest.
config.business_description = (
"I'm a real estate agent, and I need you to analyze this data for me and tell me what to do next."
)
config.use_planner = True   # set False to skip Stage 0 entirely

# Optional: override Planner's inference if you already know these values.
config.target_column = "rent"             # keep None to let Planner infer
config.problem_type = "regression" # keep None to let Planner infer
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
