#!/usr/bin/env python3
"""
AutoDS 简单示例 - 无需外部数据文件

直接在代码中生成示例数据，方便测试和演示
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_simple_example_data(output_file="example_data.csv", n_samples=100):
    """
    创建简单的示例数据集
    
    Args:
        output_file: 输出文件路径
        n_samples: 样本数量
    
    Returns:
        DataFrame
    """
    
    print("\n" + "="*80)
    print("📊 创建简单示例数据集")
    print("="*80 + "\n")
    
    print(f"生成 {n_samples} 个样本...")
    
    # 生成示例数据
    np.random.seed(42)
    
    data = {
        'customer_id': np.arange(1, n_samples + 1),
        'age': np.random.randint(20, 70, n_samples),
        'income': np.random.randint(30000, 150000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'purchase_history': np.random.randint(0, 100, n_samples),
        'target': np.random.randint(0, 2, n_samples)  # 二分类目标
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些缺失值（模拟真实数据）
    missing_indices = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'age'] = np.nan
    
    # 添加一些异常值
    anomaly_indices = np.random.choice(len(df), size=int(0.03 * len(df)), replace=False)
    df.loc[anomaly_indices, 'income'] = np.random.choice([999999, -10000], size=len(anomaly_indices))
    
    # 保存数据
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ 数据集已创建")
    print(f"  - 文件: {output_file}")
    print(f"  - 行数: {len(df)}")
    print(f"  - 列数: {len(df.columns)}")
    print(f"  - 缺失值: {df.isnull().sum().sum()}")
    print(f"\n数据样本:")
    print(df.head(10))
    print(f"\n统计信息:")
    print(df.describe())
    
    return df


def main():
    """主函数"""
    
    # 创建数据
    df = create_simple_example_data(
        output_file="example_data.csv",
        n_samples=100
    )
    
    print("\n" + "="*80)
    print("✅ 示例数据创建成功！")
    print("="*80)
    print("\n现在可以运行 Pipeline:")
    print("  python run.py")
    print("\n或修改 run.py 中的 config 指向此文件:")
    print("  config.data_path = 'example_data.csv'")
    print("  config.target_column = 'target'")
    print()


if __name__ == "__main__":
    main()
