#!/usr/bin/env python3
"""
BDD-RAG-System 主入口

用法:
    python main.py
    python main.py --use-case-file docs/用例示例.md
"""

import argparse
import json
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
KB1_DIR = PROJECT_ROOT / "knowledge-base/kb1_aw_sdk/indexed/chroma_db"
KB2_DIR = PROJECT_ROOT / "knowledge-base/kb2_bdd_scenarios/indexed/chroma_db"
OUTPUT_DIR = PROJECT_ROOT / "generated_scripts"
USE_CASE_FILE = PROJECT_ROOT / "docs/用例示例.md"


def main():
    parser = argparse.ArgumentParser(description="BDD-RAG Pipeline")
    parser.add_argument(
        "--use-case-file",
        type=str,
        default=str(USE_CASE_FILE),
        help="用例文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="生成脚本输出目录"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=" Anthropic API Key（优先级: 命令行 > config.yaml > 环境变量）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（默认 config/config.yaml）"
    )
    args = parser.parse_args()

    # 导入模块
    from core.pipeline import load_use_cases, run_pipeline, Pipeline
    from core.llm import create_llm
    from config.config_loader import load_config, get_api_key, get_model, get_llm_config

    # 加载配置
    config = load_config(args.config)
    api_key = args.api_key or get_api_key(config)
    model = get_model(config)

    print("=" * 70)
    print("BDD-RAG Pipeline")
    print("=" * 70)

    # 解析用例
    print(f"\n[1] 加载用例: {args.use_case_file}")
    use_cases = load_use_cases(args.use_case_file)
    print(f"    共 {len(use_cases)} 个用例")

    for uc in use_cases:
        print(f"    - {uc.id}: {uc.name}")

    # 创建 Pipeline
    print(f"\n[2] 初始化 Pipeline")
    print(f"    KB1: {KB1_DIR}")
    print(f"    KB2: {KB2_DIR}")

    llm_cfg = get_llm_config(config)
    llm = None
    if llm_cfg["api_key"]:
        llm = create_llm(
            provider=llm_cfg["provider"],
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg["api_key"],
            model=llm_cfg["model"],
        )
        print(f"    LLM: {llm_cfg['provider']}/{llm_cfg['model']} ({llm_cfg['base_url']}) (已配置)")
    else:
        print(f"    LLM: 未配置 API_KEY，将跳过脚本生成")

    pipeline = Pipeline(
        kb1_dir=str(KB1_DIR),
        kb2_dir=str(KB2_DIR),
        llm=llm,
        output_dir=args.output_dir
    )

    # 执行 Pipeline
    print(f"\n[3] 执行 Pipeline...")
    results = pipeline.run_batch(use_cases)

    # 汇总报告
    print(f"\n{'='*70}")
    print("Pipeline 执行完成 - 结果汇总")
    print(f"{'='*70}")

    summary = []
    for r in results:
        status = "✅" if r.test_result and r.test_result.get("passed") else "❌"
        test_info = ""
        if r.test_result:
            tc = r.test_result
            test_info = f"{tc['passed_count']}/{tc['total']}"
        print(f"  {r.use_case_id}: {'命中' if r.hit else 'MISS'} | {status} {test_info}")
        summary.append(r.to_dict())

    # 保存汇总
    summary_path = Path(args.output_dir) / "pipeline_summary.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
