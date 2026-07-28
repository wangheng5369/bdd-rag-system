#!/usr/bin/env python3
"""
BDD知识库Embedding脚本
使用 markdown-it-py 进行 AST 解析，按 h2/h3 层级分片

分片策略: Scenario Chunk
- h2 标题: 模块边界（一级分片）
- h3 标题: 场景边界（二级分片，每个 h3 包含 Gherkin + Python 实现）

分片结果: 5个模块 × 2 = 10 个 chunks
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from markdown_it import MarkdownIt

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 配置路径
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "knowledge-base" / "kb2_bdd_scenarios" / "raw"
CHUNKED_DIR = PROJECT_ROOT / "knowledge-base" / "kb2_bdd_scenarios" / "chunked"
INDEXED_DIR = PROJECT_ROOT / "knowledge-base" / "kb2_bdd_scenarios" / "indexed"
PERSIST_DIR = INDEXED_DIR / "chroma_db"

# 确保目录存在
for d in [CHUNKED_DIR, INDEXED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# BGE-Large-ZH Embedding 配置
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_MODEL_KWARGS = {"device": "cpu"}  # 可改为 "cuda" 如果有GPU
QUERY_INSTRUCTION = "为检索任务编码："

# 向量数据库配置
COLLECTION_NAME = "cce_bdd_kb2"


@dataclass
class ChunkInfo:
    """分片信息"""
    title_path: str      # 标题路径，如 "一、 集群模块 / 1. 步骤定义与 Gherkin 范例"
    level: int           # 标题层级 (2=h2, 3=h3)
    content: str         # 分片内容
    module_name: str      # 模块名
    scenario_name: str    # 场景名（Gherkin或Python）


def parse_frontmatter(content: str) -> Tuple[Optional[Dict], str]:
    """
    解析 frontmatter (--- 包裹的 YAML 头部)
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            import yaml
            try:
                metadata = yaml.safe_load(parts[1])
                return metadata, parts[2].strip()
            except yaml.YAMLError:
                pass
    return None, content


def split_by_ast(markdown_text: str) -> List[ChunkInfo]:
    """
    基于 AST 按标题层级分片

    分片规则:
    - h2 标题: 模块边界 (一级分片)
    - h3 标题: 场景边界 (二级分片，每个 h3 包含 Gherkin + Python 实现)

    Returns:
        分片列表
    """
    # 解析 frontmatter
    _, content = parse_frontmatter(markdown_text)

    # 解析 AST
    md = MarkdownIt()
    tokens = md.parse(content)

    chunks = []

    # 第一遍：收集 h2/h3 标题信息
    headings = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag[1])
            if level not in (2, 3):
                continue
            if token.map:
                start_line = token.map[0]
                # 找到标题内容
                heading_content = ""
                for j in range(i + 1, len(tokens)):
                    if tokens[j].type == "inline":
                        heading_content = tokens[j].content.strip()
                        break
                    if tokens[j].type == "heading_close":
                        break
                headings.append({
                    "level": level,
                    "line": start_line,
                    "content": heading_content
                })

    # 按行分片：每个 h3 标题 + 其后续内容直到下一个同级或上级标题
    lines = content.split("\n")

    for idx, heading in enumerate(headings):
        level = heading["level"]
        title = heading["content"]
        start_line = heading["line"]

        # 确定该 chunk 的结束位置（下一个同级或上级标题的行号）
        end_line = len(lines)
        for next_h in headings[idx + 1:]:
            if next_h["level"] <= level:
                end_line = next_h["line"]
                break

        # 收集该 chunk 的内容（从标题下一行到 end_line 之前）
        chunk_lines = []
        for i in range(start_line + 1, end_line):
            if i < len(lines):
                chunk_lines.append(lines[i])

        chunk_content = "\n".join(chunk_lines).strip()
        if not chunk_content:
            continue

        # 确定标题路径
        if level == 2:
            title_path = title
            module_name = title
            scenario_name = ""
        else:  # level == 3
            # 找所属的 h2
            parent_h2 = ""
            for h in headings[:idx]:
                if h["level"] == 2:
                    parent_h2 = h["content"]
            title_path = f"{parent_h2} / {title}"
            module_name = parent_h2
            scenario_name = title

        chunks.append(ChunkInfo(
            title_path=title_path,
            level=level,
            content=chunk_content,
            module_name=module_name,
            scenario_name=scenario_name
        ))

    return chunks


def chunks_to_documents(chunks: List[ChunkInfo]) -> List[Document]:
    """
    将分片信息转换为 LangChain Document
    """
    documents = []
    for i, chunk in enumerate(chunks):
        # 构建包含标题路径的完整内容
        full_content = f"{chunk.title_path}\n\n{chunk.content}"

        metadata = {
            "chunk_id": i,
            "title_path": chunk.title_path,
            "level": chunk.level,
            "module_name": chunk.module_name,
            "scenario_name": chunk.scenario_name,
        }

        documents.append(Document(page_content=full_content, metadata=metadata))

    return documents


def load_raw_documents(raw_dir: Path) -> List[str]:
    """
    加载原始文档
    """
    documents = []
    for file_path in raw_dir.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append(content)
    return documents


def main():
    print("=" * 60)
    print("BDD 知识库 Embedding 开始")
    print("=" * 60)

    # 1. 加载原始文档
    print(f"\n[1/6] 加载原始文档 from {RAW_DIR}...")
    raw_docs = load_raw_documents(RAW_DIR)
    print(f"    加载了 {len(raw_docs)} 个文档")

    # 2. AST 分片
    print(f"\n[2/6] 使用 AST 分片策略处理文档...")
    all_chunks = []
    for doc_content in raw_docs:
        chunks = split_by_ast(doc_content)
        all_chunks.extend(chunks)

    print(f"    共生成 {len(all_chunks)} 个 chunk")

    # 按层级统计
    level_counts = {}
    for chunk in all_chunks:
        level_counts[chunk.level] = level_counts.get(chunk.level, 0) + 1
    print(f"    层级分布: {level_counts}")

    # 3. 转换为 LangChain Document
    print(f"\n[3/6] 转换为 LangChain Document...")
    documents = chunks_to_documents(all_chunks)
    print(f"    转换了 {len(documents)} 个 Document")

    # 保存 chunked 文档
    chunked_file = CHUNKED_DIR / "chunks.json"
    chunk_data = [
        {
            "chunk_id": d.metadata["chunk_id"],
            "title_path": d.metadata["title_path"],
            "level": d.metadata["level"],
            "module_name": d.metadata["module_name"],
            "scenario_name": d.metadata["scenario_name"],
            "content_preview": d.page_content[:100].replace("\n", " ")
        }
        for d in documents
    ]
    with open(chunked_file, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)
    print(f"    已保存到 {chunked_file}")

    # 打印分片预览
    print(f"\n    Chunk 预览 (全部):")
    for i, chunk in enumerate(all_chunks, 1):
        preview = chunk.content[:80].replace("\n", " ")
        print(f"      {i}. [{chunk.level}] {chunk.title_path}")
        print(f"         {preview}...")

    # 4. 初始化 Embedding 模型
    print(f"\n[4/6] 初始化 Embedding 模型: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={**EMBEDDING_MODEL_KWARGS, "local_files_only": True},
        query_instruction=QUERY_INSTRUCTION,
    )
    print(f"    模型加载完成")

    # 5. 向量化并存储到 ChromaDB
    print(f"\n[5/6] 向量化文档并存储到 ChromaDB...")
    print(f"    向量数据库路径: {PERSIST_DIR}")
    print(f"    Collection 名称: {COLLECTION_NAME}")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )

    # 6. 验证
    print(f"\n[6/6] 验证向量数据库...")
    count = vectorstore._collection.count()
    print(f"    向量数据库中共有 {count} 条向量")

    # 简单测试检索
    test_queries = [
        "如何创建CCE集群",
        "创建node的步骤",
        "BDD Gherkin 范例"
    ]

    for query in test_queries:
        print(f"\n    测试检索: '{query}'")
        results = vectorstore.similarity_search(query, k=2)
        print(f"    Top-2 检索结果:")
        for i, r in enumerate(results, 1):
            meta = r.metadata
            preview = r.page_content[:80].replace("\n", " ")
            level = meta.get('level', 'N/A')
            title_path = meta.get('title_path', 'N/A')
            print(f"      {i}. [{level}] {title_path}")
            print(f"         {preview}...")

    print("\n" + "=" * 60)
    print("BDD 知识库 Embedding 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
