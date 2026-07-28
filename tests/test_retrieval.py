#!/usr/bin/env python3
"""
Retrieval 模块测试

基于 docs/用例示例.md 的用例测试 retrieval 逻辑
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval.retrieval import Retrieval


def load_use_case(file_path: str) -> str:
    """加载用例文档"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# 用例查询
USE_CASES = [
    {
        "id": "Tc_Func_Node_001",
        "name": "创建100G数据盘的节点",
        "query": "创建100G数据盘的节点 创建集群 创建节点 2u4g规格 设置数据盘100G",
        "expected": "知识库中应有匹配"
    },
    {
        "id": "Tc_Func_Node_002",
        "name": "删除100G数据盘的节点",
        "query": "删除集群下200G数据盘的节点 删除节点",
        "expected": "知识库中可能无匹配"
    },
    {
        "id": "Tc_Func_Node_003",
        "name": "纳管100G数据盘的节点",
        "query": "在ecs服务创建100G的虚机 纳管虚机到cce集群 添加节点 纳管",
        "expected": "知识库中可能无匹配"
    },
]


def test_retrieval():
    """测试 retrieval 模块"""

    # 配置路径
    project_root = Path(__file__).parent.parent
    kb1_dir = project_root / "knowledge-base" / "kb1_aw_sdk" / "indexed" / "chroma_db"
    kb2_dir = project_root / "knowledge-base" / "kb2_bdd_scenarios" / "indexed" / "chroma_db"
    use_case_file = project_root / "docs" / "用例示例.md"
    output_file = project_root / "tests" / "retrieval_results.json"

    print("=" * 70)
    print("Retrieval 模块测试 - 基于用例示例")
    print("=" * 70)

    # 加载用例文档
    print(f"\n[1] 加载用例文档: {use_case_file}")
    use_case_content = load_use_case(str(use_case_file))

    # 2. 初始化 Retrieval - 使用余弦相似度阈值 0.4
    print(f"\n[2] 初始化 Retrieval (阈值: 0.4)")
    retrieval = Retrieval(
        persist_dirs={
            "kb1": str(kb1_dir),
            "kb2": str(kb2_dir),
        },
        similarity_threshold=0.4
    )

    # 3. 执行检索
    print(f"\n[3] 执行检索...")

    all_results = {}

    for i, uc in enumerate(USE_CASES, 1):
        query = uc["query"]
        print(f"\n{'='*70}")
        print(f"用例 {i}: {uc['id']} - {uc['name']}")
        print(f"查询: {query}")
        print(f"预期: {uc['expected']}")
        print(f"{'='*70}")

        # KB2 (BDD) 检索
        print(f"\n[KB2 - BDD知识库] 检索...")
        kb2_results = retrieval.retrieve(query, kb_name="kb2", top_k=5)

        if kb2_results:
            print(f"  召回 {len(kb2_results)} 条结果 (阈值: {retrieval.similarity_threshold}):")
            for j, r in enumerate(kb2_results, 1):
                print(f"    {j}. score={r['score']:.4f}")
                title = r['metadata'].get('title_path', 'N/A')
                print(f"       {title[:70]}...")
        else:
            print(f"  [召回 Miss] 无满足阈值({retrieval.similarity_threshold})的结果")

        # KB1 (SDK) 检索
        print(f"\n[KB1 - SDK知识库] 检索...")
        kb1_results = retrieval.retrieve(query, kb_name="kb1", top_k=5)

        if kb1_results:
            print(f"  召回 {len(kb1_results)} 条结果 (阈值: {retrieval.similarity_threshold}):")
            for j, r in enumerate(kb1_results, 1):
                print(f"    {j}. score={r['score']:.4f}")
                title = r['metadata'].get('title_path', 'N/A')
                print(f"       {title[:70]}...")
        else:
            print(f"  [召回 Miss] 无满足阈值({retrieval.similarity_threshold})的结果")

        # 测试降级召回
        print(f"\n[降级召回] 测试 retrieve_with_fallback (KB2 → KB1)...")
        fallback_results, did_fallback = retrieval.retrieve_with_fallback(
            query=query,
            primary_kb="kb2",
            fallback_kb="kb1",
            top_k=5
        )
        if did_fallback:
            print(f"  [降级] 主知识库召回miss，使用了KB1降级")
        else:
            print(f"  [未降级] 主知识库KB2有结果")
        print(f"  最终召回结果: {len(fallback_results)} 条")

        # 判断是否召回miss
        total_recall = len(kb2_results) + len(kb1_results)
        if total_recall == 0:
            recall_status = "召回MISS - 需要生成新step"
        else:
            recall_status = f"召回成功 - {total_recall}条结果"

        print(f"\n  【判定】: {recall_status}")

        # 保存结果
        all_results[uc["id"]] = {
            "name": uc["name"],
            "query": query,
            "expected": uc["expected"],
            "recall_status": recall_status,
            "kb2_results_count": len(kb2_results),
            "kb1_results_count": len(kb1_results),
            "fallback_used": did_fallback,
            "kb2_top3": [
                {
                    "score": round(r["score"], 4),
                    "title": r['metadata'].get('title_path', '')[:100]
                }
                for r in kb2_results[:3]
            ],
            "kb1_top3": [
                {
                    "score": round(r["score"], 4),
                    "title": r['metadata'].get('title_path', '')[:100]
                }
                for r in kb1_results[:3]
            ]
        }

    # 4. 保存结果
    print(f"\n[4] 保存结果到: {output_file}")
    output_data = {
        "use_case_file": str(use_case_file),
        "threshold": Retrieval.DEFAULT_SIMILARITY_THRESHOLD,
        "results": all_results
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"    已保存!")

    # 5. 总结
    print(f"\n" + "=" * 70)
    print("测试完成 - 结果摘要")
    print("=" * 70)

    for uc_id, result in all_results.items():
        print(f"\n  {uc_id}: {result['name']}")
        print(f"    召回状态: {result['recall_status']}")
        print(f"    KB2: {result['kb2_results_count']}条, KB1: {result['kb1_results_count']}条")
        if result['fallback_used']:
            print(f"    降级: 是")


if __name__ == "__main__":
    test_retrieval()
