#!/usr/bin/env python3
"""
Reranker 模块测试
测试 Reranker 与 Retrieval 的串联
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.retrieval.retrieval import Retrieval
from core.rerank.rerank import Reranker


def test_rerank():
    """测试 Reranker 模块"""

    # 配置路径
    project_root = Path(__file__).parent.parent
    kb1_dir = project_root / "knowledge-base" / "kb1_aw_sdk" / "indexed" / "chroma_db"
    kb2_dir = project_root / "knowledge-base" / "kb2_bdd_scenarios" / "indexed" / "chroma_db"
    output_file = project_root / "tests" / "rerank_results.json"

    print("=" * 70)
    print("Reranker 模块测试 - Retrieval + Rerank 串联")
    print("=" * 70)

    # 1. 初始化 Retrieval
    print(f"\n[1] 初始化 Retrieval (阈值: 0.6)")
    retrieval = Retrieval(
        persist_dirs={
            "kb1": str(kb1_dir),
            "kb2": str(kb2_dir),
        },
        similarity_threshold=0.6
    )

    # 2. 初始化 Reranker
    print(f"\n[2] 初始化 Reranker (阈值: 0.6)")
    reranker = Reranker(
        model_name="BAAI/bge-reranker-base",
        threshold=0.6,
        device="cpu"
    )

    # 3. 测试用例
    query = "创建100G数据盘的节点"
    print(f"\n[3] 查询: {query}")

    # 4. 先召回
    print(f"\n[4] Retrieval 召回...")
    kb2_results = retrieval.retrieve(query, kb_name="kb2", top_k=10)
    print(f"    KB2 召回 {len(kb2_results)} 条结果")

    kb1_results = retrieval.retrieve(query, kb_name="kb1", top_k=10)
    print(f"    KB1 召回 {len(kb1_results)} 条结果")

    # 合并召回结果
    all_candidates = kb2_results + kb1_results
    print(f"    合并后共 {len(all_candidates)} 条候选")

    # 5. 再重排
    print(f"\n[5] Reranker 重排...")
    reranked_results = reranker.rerank(query, all_candidates, top_k=5)
    print(f"    重排后返回 {len(reranked_results)} 条结果 (阈值: {reranker.threshold})")

    # 6. 打印结果
    print(f"\n[6] 重排结果 (Top 5):")
    for i, r in enumerate(reranked_results, 1):
        print(f"    {i}. score={r['score']:.4f}")
        title = r['metadata'].get('title_path', 'N/A')
        print(f"       {title[:70]}...")

    # 7. 保存结果
    print(f"\n[7] 保存结果到: {output_file}")
    output_data = {
        "query": query,
        "retrieval_kb2_count": len(kb2_results),
        "retrieval_kb1_count": len(kb1_results),
        "reranked_count": len(reranked_results),
        "top5": [
            {
                "score": round(r["score"], 4),
                "title": r['metadata'].get('title_path', '')[:100],
                "content_preview": r["content"][:200].replace("\n", " ")
            }
            for r in reranked_results[:5]
        ]
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"    已保存!")

    print(f"\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_rerank()
