#!/usr/bin/env python3
"""
立即修复 Stage 6 - 复制粘贴即用

直接修改 multi_agent_report_generator.py 中的 LangChain 导入
"""

import os
import sys


def main():
    print("\n" + "=" * 80)
    print("🔧 修复 Stage 6 - LangChain 导入问题")
    print("=" * 80 + "\n")

    file_path = "multi_agent_report_generator.py"

    # 检查文件
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到 {file_path}")
        print("\n💡 请在包含此文件的目录中运行此脚本")
        return False

    print(f"📂 发现文件：{file_path}")

    # 读取
    print("📖 读取文件...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 读取失败：{e}")
        return False

    # 检查是否已经修复
    if "langchain_core" in content and "try:" in content:
        print("✅ 文件已包含兼容性处理，无需修复")
        return True

    # 替换
    print("⚙️  应用修复...")

    # 方案 1：替换导入块
    old_block1 = """from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate"""

    new_block1 = """try:
    from langchain_core.chains import LLMChain, SequentialChain
    from langchain_core.prompts import PromptTemplate
except:
    from langchain.chains import LLMChain, SequentialChain
    from langchain.prompts import PromptTemplate"""

    if old_block1 in content:
        print("  ✓ 匹配到导入块 1")
        content = content.replace(old_block1, new_block1)
    else:
        print("  ⚠️  导入块 1 未找到，尝试备选方案...")
        # 备选方案
        if "from langchain.chains import" in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "from langchain.chains import" in line:
                    print(f"  ✓ 在第 {i + 1} 行找到")
                    # 简单替换
                    lines[i] = "try:"
                    lines.insert(i + 1, "    from langchain_core.chains import LLMChain, SequentialChain")
                    lines.insert(i + 2, "except:")
                    lines.insert(i + 3, "    from langchain.chains import LLMChain, SequentialChain")

                    # 处理下一行的 PromptTemplate
                    if i + 4 < len(lines) and "from langchain.prompts import" in lines[i + 4]:
                        lines[i + 4] = "try:"
                        lines.insert(i + 5, "    from langchain_core.prompts import PromptTemplate")
                        lines.insert(i + 6, "except:")
                        lines.insert(i + 7, "    from langchain.prompts import PromptTemplate")

                    content = '\n'.join(lines)
                    break

    # 创建备份
    backup_file = f"{file_path}.backup"
    print(f"💾 创建备份：{backup_file}")
    try:
        with open(backup_file, 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as orig:
                f.write(orig.read())
    except Exception as e:
        print(f"⚠️  备份失败（继续）：{e}")

    # 写入
    print("📝 写入修复后的文件...")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"❌ 写入失败：{e}")
        return False

    print("\n" + "=" * 80)
    print("✅ 修复完成！")
    print("=" * 80)
    print("\n下一步：运行 Pipeline")
    print("  python run.py")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)