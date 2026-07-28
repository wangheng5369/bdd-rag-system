#!/usr/bin/env python3
"""
LLM 连通性测试
验证 API Key 配置正确，能够正常调用 Claude
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config, get_llm_config
from core.llm import create_llm


def test_llm_connection():
    """测试 LLM 连通性"""
    config = load_config()
    llm_cfg = get_llm_config(config)
    provider = llm_cfg["provider"]
    base_url = llm_cfg["base_url"]
    api_key = llm_cfg["api_key"]
    model = llm_cfg["model"]

    print(f"提供商: {provider}")
    print(f"Base URL: {base_url}")
    print(f"模型: {model}")
    print(f"API Key 已配置: {'是' if api_key else '否'}")

    if not api_key:
        print("❌ API Key 未配置，请检查 config.yaml 或环境变量 OPENAI_API_KEY")
        return False

    llm = create_llm(provider=provider, base_url=base_url, api_key=api_key, model=model)

    # 简单测试调用
    print("\n测试调用 Claude...")
    response = llm.generate("Hello, 简单回复'ok'即可")
    print(f"响应: {response[:200]}")

    if response.startswith("["):
        print("❌ LLM 调用失败")
        return False

    print("✅ LLM 连通正常")
    return True


def test_script_prompt():
    """测试完整的 Step 生成 prompt"""
    from core.llm import LLMGenerator

    llm_cfg = get_llm_config(load_config())
    if not llm_cfg["api_key"]:
        print("❌ API Key 未配置，跳过 prompt 测试")
        return

    llm = create_llm(
        provider=llm_cfg["provider"],
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        model=llm_cfg["model"],
    )

    prompt = llm.build_script_prompt(
        scenario="删除节点",
        steps=[{"gherkin": "When 删除节点，名称为node-01"}],
        sdk_examples=[{
            "content": "def delete_node(cluster_id: str, node_id: str) -> bool:\n    pass"
        }],
        existing_bdds=[{
            "content": "When 删除节点，名称为NAME"
        }]
    )

    print("\n--- Prompt Preview ---")
    print(prompt[:500])
    print("--- End ---\n")

    print("调用 Claude 生成脚本...")
    response = llm.generate(prompt)
    print(f"响应:\n{response[:500]}")


if __name__ == "__main__":
    print("=" * 60)
    print("LLM 连通性测试")
    print("=" * 60)

    ok = test_llm_connection()

    if ok:
        print("\n" + "=" * 60)
        print("Prompt + 生成测试")
        print("=" * 60)
        test_script_prompt()
