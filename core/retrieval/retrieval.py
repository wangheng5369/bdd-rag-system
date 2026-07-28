"""
检索模块 - 向量数据库召回
基于 BAAI/bge-large-zh-v1.5 embedding 实现 top-k 召回
使用余弦相似度（Cosine Similarity）
封装为 LangChain BaseRetriever 接口
"""

from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import re
import numpy as np


# 默认配置
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_QUERY_INSTRUCTION = "为检索任务编码："

# Reranker 默认阈值
DEFAULT_RERANKER_THRESHOLD = 0.6

# ============================================================
# 阶段2: 动作关键词同义词映射表 (Action Mapping)
# ============================================================
ACTION_MAPPING = {
    "CREATE": ["创建", "新增", "初始化", "add", "create", "添加"],
    "DELETE": ["删除", "销毁", "移除", "delete", "remove", "卸载"],
    "UPDATE": ["修改", "设置", "变更", "set", "update", "更新", "变更", "重置"],
    "QUERY": ["查询", "获取", "获取", "query", "get", "list", "describe"],
    "ENABLE": ["启用", "激活", "启动", "enable", "start"],
    "DISABLE": ["禁用", "停用", "停止", "disable", "stop"],
}

# 动作组互斥映射：同一组内的动作可以共存，不同组的动作可能冲突
# 这里定义跨组冲突规则（当 Query 属于 A 组，BDD 属于 B 组时，直接熔断）
ACTION_CONFLICT_GROUPS = [
    {"CREATE", "DELETE"},   # 创建 vs 删除 互斥
    {"ENABLE", "DISABLE"},   # 启用 vs 禁用 互斥
    {"CREATE", "UPDATE"},   # 创建 vs 重置/修改 互斥（重置是特殊修改）
    {"DELETE", "UPDATE"},   # 删除 vs 重置/修改 互斥
]


def get_action_group(action: str) -> str | None:
    """
    根据动作关键词获取所属的动作组

    Args:
        action: 动作关键词，如 "创建", "delete"

    Returns:
        动作组名 (如 "CREATE")，未找到返回 None
    """
    for group, keywords in ACTION_MAPPING.items():
        if action in keywords:
            return group
    return None


def detect_action_group(text: str) -> set:
    """
    从文本中检测所有动作组

    Args:
        text: 输入文本

    Returns:
        检测到的动作组集合，如 {"CREATE", "DELETE"}
    """
    found_groups = set()
    for group, keywords in ACTION_MAPPING.items():
        for keyword in keywords:
            if keyword in text:
                found_groups.add(group)
                break
    return found_groups


def is_hard_rule_miss(query: str, candidate_content: str) -> bool:
    """
    阶段2核心: 规则熔断 (Hard Rule Miss)

    通过动作关键词同义词映射表判断 Query 与候选 BDD 是否属于互斥的动作组。
    若 Query 属于 A 组，召回的 BDD 属于 B 组，且 {A, B} 在同一互斥对中，直接熔断。

    Args:
        query: 查询文本
        candidate_content: 候选 BDD 内容

    Returns:
        True 表示硬规则冲突，需要熔断（过滤掉）
        False 表示无硬规则冲突
    """
    query_groups = detect_action_group(query)
    doc_groups = detect_action_group(candidate_content)

    if not query_groups or not doc_groups:
        # 任一方没有检测到动作组，不触发硬规则熔断
        return False

    # 检查是否存在跨组冲突
    # 互斥对 {A, B}：query 有 A 且 doc 有 B，或 query 有 B 且 doc 有 A
    for conflict_pair in ACTION_CONFLICT_GROUPS:
        pair_list = list(conflict_pair)
        if len(pair_list) != 2:
            continue
        g1, g2 = pair_list
        # query 有 g1 且 doc 有 g2，或 query 有 g2 且 doc 有 g1
        if (g1 in query_groups and g2 in doc_groups) or \
           (g2 in query_groups and g1 in doc_groups):
            return True

    return False


class Retrieval:
    """向量检索模块 - 使用余弦相似度"""

    # 默认相似度阈值
    DEFAULT_SIMILARITY_THRESHOLD = 0.6

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        persist_dirs: Optional[Dict[str, str]] = None,
        similarity_threshold: float = 0.6,
        reranker_threshold: float = DEFAULT_RERANKER_THRESHOLD
    ):
        """
        初始化检索模块

        Args:
            embedding_model: Embedding 模型名称
            persist_dirs: 各知识库的 ChromaDB 持久化目录
                          格式: {"kb1": "/path/to/vdb1", "kb2": "/path/to/vdb2", "kb3": "/path/to/vdb3"}
            similarity_threshold: 相似度阈值，低于此值的结果会被过滤掉
            reranker_threshold: Reranker 重排分数阈值，低于此值的结果会被过滤
        """
        self.embedding_model = embedding_model
        self.persist_dirs = persist_dirs or {}
        self.similarity_threshold = similarity_threshold
        self.reranker_threshold = reranker_threshold

        # 初始化 embedding 函数
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu", "local_files_only": True},
            query_instruction=DEFAULT_QUERY_INSTRUCTION,
        )

        # 初始化 Reranker (延迟导入避免循环依赖)
        from core.rerank.rerank import Reranker as RerankerClass
        self.reranker = RerankerClass(threshold=reranker_threshold)

        # 加载各知识库的向量数据库
        self.vectorstores: Dict[str, Chroma] = {}
        self._load_vectorstores()

    def _load_vectorstores(self):
        """加载各知识库的向量数据库（使用余弦相似度）"""
        for kb_name, persist_dir in self.persist_dirs.items():
            if Path(persist_dir).exists():
                self.vectorstores[kb_name] = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=self._get_collection_name(kb_name),
                    collection_metadata={"hnsw:space": "cosine"}  # 指定使用余弦相似度
                )
                print(f"已加载知识库 {kb_name}: {persist_dir} (余弦相似度)")

    def _get_collection_name(self, kb_name: str) -> str:
        """获取知识库对应的 collection 名称"""
        collection_names = {
            "kb1": "cce_sdk_kb1",
            "kb2": "cce_bdd_kb2",
            "kb3": "cce_bug_kb3",
        }
        return collection_names.get(kb_name, kb_name)

    def retrieve(
        self,
        query: str,
        kb_name: str = "kb2",
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        从指定知识库中召回 top-k 条相似语句（使用余弦相似度）

        Args:
            query: 查询文本
            kb_name: 知识库名称 (kb1/kb2/kb3)
            top_k: 返回条数
            threshold: 相似度阈值，低于此值的结果会被过滤（默认 0.6）

        Returns:
            List[Dict]: 召回结果列表，包含 score, content, metadata
                      如果返回空列表，表示召回 miss
        """
        if kb_name not in self.vectorstores:
            print(f"警告: 知识库 {kb_name} 未加载")
            return []

        vectorstore = self.vectorstores[kb_name]
        effective_threshold = threshold if threshold is not None else self.similarity_threshold

        # 执行余弦相似度搜索
        # 注意：ChromaDB 的 similarity_search_with_score 返回的分数就是余弦相似度（当 collection 使用 cosine 时）
        results = vectorstore.similarity_search_with_score(query, k=top_k)

        # 过滤：只保留相似度 >= threshold 的结果
        recalled = []
        for doc, score in results:
            # 余弦相似度分数（0-1之间，越大越相似）
            cosine_score = float(score)

            if cosine_score >= effective_threshold:
                recalled.append({
                    "score": cosine_score,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        return recalled

    def compute_confidence(self, query: str, candidate: str) -> float:
        """
        计算召回置信度 (余弦相似度)

        Args:
            query: 查询文本
            candidate: 候选文本

        Returns:
            float: 置信度分数 [0, 1]，越接近1越相似
        """
        # 计算 query 和 candidate 的 embedding
        query_emb = self.embeddings.embed_query(query)
        candidate_emb = self.embeddings.embed_query(candidate)

        # 计算余弦相似度
        q = np.array(query_emb)
        c = np.array(candidate_emb)

        # 余弦相似度 = dot(q, c) / (||q|| * ||c||)
        similarity = np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c))

        return float(similarity)

    def filter_by_action_conflict(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        阶段2: 规则熔断筛选 - 过滤掉动作组互斥的结果

        使用动作关键词同义词映射表 + 硬规则熔断机制:
        - 若 Query 属于 CREATE 组，而召回的 BDD 属于 DELETE 组，直接熔断

        Args:
            query: 查询文本
            candidates: 候选列表

        Returns:
            List[Dict]: 过滤后的候选列表
        """
        if not candidates:
            return []

        # 从query中检测动作组
        query_groups = detect_action_group(query)
        if not query_groups:
            # 如果query中没有检测到动作组，不进行过滤
            return candidates

        filtered = []
        hard_rule_miss_count = 0

        for candidate in candidates:
            content = candidate.get('content', '')

            # 阶段2核心: 硬规则熔断检查
            if is_hard_rule_miss(query, content):
                hard_rule_miss_count += 1
                continue

            filtered.append(candidate)

        if hard_rule_miss_count > 0:
            print(f"[阶段2-规则熔断] 熔断了 {hard_rule_miss_count} 条冲突结果")

        return filtered

    def retrieve_with_fallback(
        self,
        query: str,
        primary_kb: str = "kb2",
        fallback_kb: str = "kb1",
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        召回，带自动降级 + 正则筛选

        流程：
        1. 阶段1 (Retrieval): 从主知识库召回，按阈值过滤
        2. 阶段2 (正则筛选): 过滤动作冲突的结果
        3. 如果主知识库miss，降级到备选知识库

        Args:
            query: 查询文本
            primary_kb: 主知识库名称
            fallback_kb: 降级知识库名称
            top_k: 返回条数

        Returns:
            Tuple[List[Dict], bool]: (召回结果列表, 是否降级)
                - 如果降级标志为 True，表示主知识库召回 miss，使用了 fallback
                - 如果返回空列表，表示两个知识库都没有召回结果
        """
        # 阶段1: 从主知识库召回
        results = self.retrieve(query, primary_kb, top_k)

        # 阶段2: 正则筛选 - 过滤动作冲突的结果
        filtered_results = self.filter_by_action_conflict(query, results)
        print(f"[阶段2-正则筛选] 过滤后: {len(filtered_results)}/{len(results)} 条结果")

        # 阶段3: Rerank 重排
        if filtered_results:
            reranked = self.reranker.rerank(query, filtered_results, top_k=top_k)
            rerank_count = len(reranked)
            print(f"[阶段3-Rerank] 重排后: {rerank_count}/{len(filtered_results)} 条结果 (阈值: {self.reranker.threshold})")
            if rerank_count == 0:
                print(f"[阶段3-Rerank] 无结果，降级到 {fallback_kb}")
                # 降级到备选知识库
                results = self.retrieve(query, fallback_kb, top_k)
                filtered_results = self.filter_by_action_conflict(query, results)
                if filtered_results:
                    reranked = self.reranker.rerank(query, filtered_results, top_k=top_k)
                    if reranked:
                        return reranked, True
                return [], True
            return reranked, False

        # 判断召回 miss
        if len(filtered_results) == 0:
            print(f"[召回 Miss] 主知识库 {primary_kb} 筛选后无结果，降级到 {fallback_kb}")
            # 降级到备选知识库
            results = self.retrieve(query, fallback_kb, top_k)
            filtered_results = self.filter_by_action_conflict(query, results)
            print(f"[阶段2-正则筛选] 降级库筛选后: {len(filtered_results)}/{len(results)} 条结果")

            if len(filtered_results) == 0:
                print(f"[召回 Miss] 降级知识库 {fallback_kb} 筛选后也无结果")
                return [], True  # 降级标志仍为 True，表示尝试过降级

            # 降级库结果进行 Rerank
            if filtered_results:
                reranked = self.reranker.rerank(query, filtered_results, top_k=top_k)
                if reranked:
                    return reranked, True
                return [], True

            return [], True

        return filtered_results, False  # 正常召回

    def multi_kb_retrieve(
        self,
        query: str,
        kb_names: List[str] = None,
        top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        从多个知识库中召回

        Args:
            query: 查询文本
            kb_names: 知识库名称列表，如 ["kb1", "kb2"]
            top_k: 每个知识库的返回条数

        Returns:
            Dict[str, List[Dict]]: 各知识库的召回结果
        """
        if kb_names is None:
            kb_names = list(self.vectorstores.keys())

        results = {}
        for kb_name in kb_names:
            results[kb_name] = self.retrieve(query, kb_name, top_k)

        return results


class BGERetriever(BaseRetriever):
    """
    LangChain 兼容的 BGE Retriever
    封装 Retrieval 类为 LangChain BaseRetriever 接口
    """

    def __init__(
        self,
        persist_dirs: Optional[Dict[str, str]] = None,
        kb_name: str = "kb2",
        top_k: int = 5,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ):
        """
        初始化 LangChain Retriever

        Args:
            persist_dirs: 各知识库的 ChromaDB 持久化目录
            kb_name: 默认检索的知识库
            top_k: 默认返回条数
            embedding_model: Embedding 模型
        """
        super().__init__()

        self.kb_name = kb_name
        self.top_k = top_k

        # 初始化底层 Retrieval
        self.retrieval = Retrieval(
            embedding_model=embedding_model,
            persist_dirs=persist_dirs or {}
        )

    def _get_relevant_documents(
        self,
        query: str,
        **kwargs
    ) -> List[Document]:
        """
        LangChain BaseRetriever 要求的接口
        """
        top_k = kwargs.get("k", self.top_k)
        kb_name = kwargs.get("kb_name", self.kb_name)

        results = self.retrieval.retrieve(query, kb_name, top_k)

        # 转换为 LangChain Document
        documents = []
        for r in results:
            documents.append(Document(
                page_content=r["content"],
                metadata=r.get("metadata", {})
            ))

        return documents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步版本"""
        return self._get_relevant_documents(query)


def create_retriever(
    kb1_dir: str,
    kb2_dir: str,
    kb3_dir: Optional[str] = None,
    default_kb: str = "kb2",
    top_k: int = 5
) -> BGERetriever:
    """
    工厂函数：创建 Retriever

    Args:
        kb1_dir: KB1 (SDK) 向量数据库路径
        kb2_dir: KB2 (BDD) 向量数据库路径
        kb3_dir: KB3 (Bug) 向量数据库路径
        default_kb: 默认检索的知识库
        top_k: 默认返回条数

    Returns:
        BGERetriever 实例
    """
    persist_dirs = {
        "kb1": kb1_dir,
        "kb2": kb2_dir,
    }
    if kb3_dir:
        persist_dirs["kb3"] = kb3_dir

    return BGERetriever(
        persist_dirs=persist_dirs,
        kb_name=default_kb,
        top_k=top_k
    )
