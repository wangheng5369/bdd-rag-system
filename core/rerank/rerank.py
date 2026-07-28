"""
重排模块 - bge-reranker
使用 Cross-Attention 机制对召回结果进行重排
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

# bge-reranker 模型名称
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_THRESHOLD = 0.6


@dataclass
class RerankResult:
    """重排结果"""
    score: float           # cross-attention score
    content: str           # 文档内容
    metadata: dict         # 元数据


class Reranker:
    """重排模块 - 使用 bge-reranker 进行 Cross-Attention 重排"""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu"
    ):
        """
        初始化 Reranker

        Args:
            model_name: bge-reranker 模型名称
            threshold: 重排分数阈值，低于此值的结果会被过滤
            device: 推理设备，"cpu" 或 "cuda"
        """
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 bge-reranker 模型"""
        try:
            from sentence_transformers import CrossEncoder
            import os
            # 设置环境变量强制使用本地文件
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self.model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device,
            )
            print(f"Reranker 模型已加载: {self.model_name}")
        except Exception as e:
            print(f"警告: Reranker 模型加载失败 ({e.__class__.__name__}: {e})，将使用简单排序")
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        对候选结果进行重排

        Args:
            query: 查询文本
            candidates: 候选列表，每个元素包含 content 和 metadata
            top_k: 只返回 top_k 条结果，None 表示返回所有

        Returns:
            List[Dict]: 重排后的结果列表，按 score 降序排列
        """
        if not candidates:
            return []

        if self.model is None:
            # 如果模型未加载，返回原始顺序
            return candidates[:top_k] if top_k else candidates

        # 提取文本对
        sentence_pairs = [
            (query, candidate.get("content", ""))
            for candidate in candidates
        ]

        # 计算 cross-attention 分数
        scores = self.model.predict(sentence_pairs)

        # 组合结果
        scored_results = []
        for i, candidate in enumerate(candidates):
            score = float(scores[i]) if hasattr(scores, '__iter__') else float(scores)
            scored_results.append({
                "score": score,
                "content": candidate.get("content", ""),
                "metadata": candidate.get("metadata", {})
            })

        # 按分数降序排列
        scored_results.sort(key=lambda x: x["score"], reverse=True)

        # 过滤低于阈值的结果
        filtered_results = [
            r for r in scored_results
            if r["score"] >= self.threshold
        ]

        # 限制返回数量
        if top_k is not None:
            filtered_results = filtered_results[:top_k]

        return filtered_results

    def is_score_pass(self, score: float) -> bool:
        """
        判断重排分数是否通过阈值

        Args:
            score: Cross-Attention 打分

        Returns:
            bool: True 表示通过阈值 (>= threshold)
        """
        return score >= self.threshold

    def compute_score(self, query: str, document: str) -> float:
        """
        计算单个 query-document 对的分数

        Args:
            query: 查询文本
            document: 文档文本

        Returns:
            float: cross-attention score
        """
        if self.model is None:
            return 0.0

        score = self.model.predict([(query, document)])
        return float(score[0] if hasattr(score, '__iter__') else score)


class BGERerankerWrapper:
    """
    BGE Reranker 封装类
    用于 LangChain ContextualCompressionRetriever
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu"
    ):
        self.reranker = Reranker(model_name, threshold, device)

    def compress_documents(
        self,
        documents: List,
        query: str
    ) -> List:
        """
        对文档列表进行重排压缩

        Args:
            documents: LangChain Document 列表
            query: 查询文本

        Returns:
            List: 重排后的文档列表
        """
        # 转换为候选格式
        candidates = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]

        # 重排
        reranked = self.reranker.rerank(query, candidates)

        # 转换回 Document 格式
        from langchain_core.documents import Document
        return [
            Document(page_content=r["content"], metadata=r["metadata"])
            for r in reranked
        ]


def create_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "cpu"
) -> Reranker:
    """
    工厂函数：创建 Reranker

    Args:
        model_name: 模型名称
        threshold: 阈值
        device: 设备

    Returns:
        Reranker 实例
    """
    return Reranker(model_name=model_name, threshold=threshold, device=device)
