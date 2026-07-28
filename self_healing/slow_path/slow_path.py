"""
Slow Path - 慢速路径处理
处理复杂逻辑、未知分支的疑难问题
通过 RAG + LLM 进行深度分析
"""

import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from core.retrieval.retrieval import Retrieval
from core.rerank.rerank import Reranker


class SlowPathHealer:
    """慢速路径自愈处理器 - 复杂问题RAG分析"""

    # 检索结果相关性阈值，低于此值视为检索 MISS
    RELEVANCE_THRESHOLD = 0.4

    def __init__(
        self,
        llm=None,
        kb3_dir: str = None,
        embedding_model: str = "BAAI/bge-large-zh-v1.5"
    ):
        """
        初始化 SlowPathHealer

        Args:
            llm: LLMGenerator 实例，用于深度分析
            kb3_dir: KB3 Bug 知识库的 ChromaDB 路径
            embedding_model: Embedding 模型名称
        """
        self.llm = llm
        self.top_k = 3

        # 初始化 Retrieval 用于 KB3 检索
        if kb3_dir:
            self.retrieval = Retrieval(
                persist_dirs={"kb3": kb3_dir},
                embedding_model=embedding_model,
                similarity_threshold=0.3  # KB3 用较低阈值
            )
        else:
            self.retrieval = None

        # 初始化 Reranker
        self.reranker = Reranker(threshold=0.5)

    def diagnose_and_heal(
        self,
        log_info: dict,
        kb3_results: Optional[List[Dict]] = None,
        fast_path_category: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        通过RAG分析复杂问题

        Args:
            log_info: 日志结构化信息
            kb3_results: 可选，预召回的 KB3 结果（用于复用 Pipeline 的召回结果）
            fast_path_category: 可选，Fast Path 分类结果（VERSION_BUG / ENV_FAULT / UNKNOWN）
            case_id: 可选，用例编号，用于报告文件名

        Returns:
            Tuple[bool, str, Optional[str]]:
            - 是否修复成功
            - 处理结果描述
            - 排查单路径（如果需要人工介入）
        """
        self.fast_path_category = fast_path_category
        self.case_id = case_id or log_info.get('case_id', '')

        # 1. RAG 检索 - 在 bug 知识库中查找匹配记录
        if kb3_results is not None:
            bug_matches = kb3_results
        else:
            bug_matches = self._rag_retrieve(log_info)

        # 2. 判断检索是否 MISS
        retrieval_hit = True
        if not bug_matches:
            retrieval_hit = False
        else:
            # 3. Rerank 获取 Top-3，并检查 top-1 相似度
            top_bugs = self._rerank_bugs(log_info, bug_matches)
            if not top_bugs:
                retrieval_hit = False
            else:
                top_score = top_bugs[0].get('score', 1.0)
                if top_score < self.RELEVANCE_THRESHOLD:
                    retrieval_hit = False

        if not retrieval_hit:
            # 检索 MISS，视为新版本问题
            return self._handle_retrieval_miss(log_info)

        # 4. LLM分析
        analysis_result = self._llm_analyze(log_info, top_bugs)

        # 5. 生成排查单
        ticket_path = self._create_ticket(log_info, top_bugs, analysis_result, retrieval_hit=True)

        return False, f"已生成排查单 {ticket_path}，等待人工确认", ticket_path

    def _handle_retrieval_miss(self, log_info: dict) -> Tuple[bool, str, Optional[str]]:
        """
        处理检索 MISS 的情况，生成"疑似新版本问题"报告

        Args:
            log_info: 日志结构化信息

        Returns:
            Tuple[bool, str, Optional[str]]:
        """
        # 生成特殊分析结果
        analysis_result = f"""**检索结果**：找不到已知 bug，疑似新版本问题

**Fast Path 分类**：{self.fast_path_category or 'UNKNOWN'}

**分析结论**：
当前错误已通过 Fast Path 分类为 **{self.fast_path_category or 'UNKNOWN'}**，但在 Bug 知识库中未找到匹配的历史记录。
这可能意味着：
1. 这是一个新发现的问题，尚未录入知识库
2. 问题的特征与现有记录差异较大，需要人工进一步分析

**建议**：
1. 人工确认是否为新版本兼容性问题
2. 如确认是新问题，建议后续录入知识库以便后续复用
"""
        # 生成排查单，retrieval_hit=False
        ticket_path = self._create_ticket(log_info, [], analysis_result, retrieval_hit=False)

        return False, f"检索 MISS，已生成排查单 {ticket_path}", ticket_path

    def _rag_retrieve(self, log_info: dict) -> List[Dict[str, Any]]:
        """
        RAG检索 - 在bug知识库中查找匹配记录（多路召回）

        Args:
            log_info: 日志信息

        Returns:
            List[Dict]: 匹配的bug记录
        """
        if not self.retrieval:
            return []

        queries = []
        all_results = []
        seen = set()

        # query1: 错误信息核心词
        msg = log_info.get('message', '')
        if msg:
            queries.append(msg[:200])

        # query2: API 名 + 错误类型
        step_name = log_info.get('step_name', '')
        error_type = log_info.get('type', '')
        if step_name:
            queries.append(f"{step_name} {error_type}")

        # query3: 从 step_name 提取模块名
        # 例如 step_build_node_req -> node
        import re
        module_match = re.search(r'(node|cluster|pool|volume|network)', step_name.lower())
        if module_match:
            queries.append(f"{module_match.group(1)} {error_type}" if error_type else module_match.group(1))

        # 合并召回结果
        for query in queries:
            if not query:
                continue
            try:
                results = self.retrieval.retrieve(
                    query=query,
                    kb_name="kb3",
                    top_k=5,
                    threshold=0.2  # 用较低阈值保证召回
                )
                for r in results:
                    key = r.get('content', '')[:100]
                    if key not in seen:
                        seen.add(key)
                        all_results.append(r)
            except Exception as e:
                print(f"[SlowPath] RAG 检索失败: {e}")
                continue

        return all_results

    def _rerank_bugs(
        self,
        log_info: dict,
        bugs: List[Dict]
    ) -> List[Dict]:
        """
        对bug记录进行rerank

        Args:
            log_info: 日志信息
            bugs: bug列表

        Returns:
            List[Dict]: Top-3 bug记录（去重后）
        """
        if not bugs:
            return []

        query = log_info.get('message', '')[:200]

        # 去重 - 优先基于 metadata.bug_id，其次基于内容前 100 字符
        seen_bug_ids = set()
        seen_content_keys = set()
        unique_bugs = []
        for bug in bugs:
            content = bug.get('content', '')
            # 优先从 metadata 获取 bug_id（来自 embedding 时存储）
            bug_id = bug.get('metadata', {}).get('bug_id', '')

            if bug_id and bug_id not in seen_bug_ids:
                seen_bug_ids.add(bug_id)
                unique_bugs.append(bug)
            else:
                key = content[:100]
                if key not in seen_content_keys:
                    seen_content_keys.add(key)
                    unique_bugs.append(bug)

        try:
            reranked = self.reranker.rerank(query, unique_bugs)
            return reranked[:self.top_k]
        except Exception as e:
            print(f"[SlowPath] Rerank 失败: {e}")
            return unique_bugs[:self.top_k]

    def _llm_analyze(
        self,
        log_info: dict,
        top_bugs: List[Dict],
        retrieval_hit: bool = True
    ) -> str:
        """
        使用LLM分析问题

        Args:
            log_info: 日志信息
            top_bugs: Top-3 bug记录
            retrieval_hit: 检索是否命中

        Returns:
            str: LLM分析结果（简洁的诊断结论）
        """
        if not self.llm:
            return "未配置 LLM，无法进行深度分析"

        # 构建 Bug 上下文
        bug_context = ""
        for i, bug in enumerate(top_bugs, 1):
            bug_context += f"""
=== 历史 Bug {i} ===
{bug.get('content', '')}
"""

        prompt = f"""你是一位 CCE 云平台故障诊断专家。请分析以下当前错误与历史 Bug 的关联。

## 当前错误信息
- 错误类型: {log_info.get('type', 'unknown')}
- 错误信息: {log_info.get('message', 'unknown')}
- 相关 Step: {log_info.get('step_name', 'unknown')}
- Scenario: {log_info.get('scenario', 'unknown')}
- 所属模块: {log_info.get('module', 'unknown')}
- Fast Path 分类: {self.fast_path_category or 'UNKNOWN'}

## 历史 Bug 记录
{bug_context}

## 输出要求

1. **只输出"诊断结论"部分，不要输出完整报告格式**（不要包含"基础信息"、"错误摘要"等章节）
2. 诊断结论中必须包含：
   - **问题判定**：{"检索命中 - 疑似已知 bug" if retrieval_hit else "检索 MISS - 找不到已知 bug，疑似新版本问题"}
   - **关联的 Bug ID**（如有）：CCE-BUG-xxx
   - **根因分析**：简洁描述问题根因
   - **建议修复方向**：不要建议修改脚本，只给出平台/SDK 层面的修复建议
3. **不要输出时间信息**（由系统自动填写）
4. 当前错误已经过 Fast Path 分类为 **{self.fast_path_category or 'UNKNOWN'}**，脚本本身是有效的，不要建议修改脚本。

请输出简洁的诊断结论：
"""

        try:
            result = self.llm.generate(prompt)
            if not result:
                return "LLM 分析失败"

            # 过滤掉 think 标签内容
            import re
            # 移除  标签及其内容
            result = re.sub(r'\[think\][\s\S]*?\[/think\]', '', result)
            # 移除  标签及其内容
            result = re.sub(r'<think>[\s\S]*?</think>', '', result)
            # 清理多余的空行
            result = re.sub(r'\n{3,}', '\n\n', result)

            return result.strip()
        except Exception as e:
            return f"LLM 分析异常: {e}"

    def _create_ticket(
        self,
        log_info: dict,
        top_bugs: List[Dict],
        analysis: str,
        retrieval_hit: bool = True
    ) -> str:
        """
        创建排查单

        Args:
            log_info: 日志信息
            top_bugs: Top-3 bug记录
            analysis: LLM分析结果
            retrieval_hit: 检索是否命中

        Returns:
            str: 排查单路径
        """
        case_id = self.case_id or log_info.get('case_id', '')
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        ticket_id = f"{case_id}_{timestamp_str}" if case_id else f"TICKET_{timestamp_str}"

        scenario = log_info.get('scenario', '未知')
        module = log_info.get('module', 'Unknown')
        step_name = log_info.get('step_name', '')
        error_msg = log_info.get('message', '')

        # 构建检索结果描述
        if retrieval_hit:
            bug_ids = [bug.get('metadata', {}).get('bug_id', f'Bug {i+1}') for i, bug in enumerate(top_bugs)]
            bug_list = ", ".join(bug_ids) if bug_ids else "无"
            retrieval_note = f"疑似已知 bug（{bug_list}）"
        else:
            retrieval_note = "找不到已知 bug，疑似新版本问题"

        # 读取模板
        template_path = Path(__file__).parent.parent.parent / "docs" / "llm-analysis-template.md"
        try:
            template = template_path.read_text(encoding="utf-8")
        except Exception:
            template = ""

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        if template:
            # 使用与 Fast Path 相同的模板格式
            report = template.replace("{scenario}", scenario)
            report = report.replace("{module}", module)
            report = report.replace("{step_name}", step_name)
            report = report.replace("{error_type}", log_info.get('type', 'error'))
            report = report.replace("{error_category}", f"{self.fast_path_category or 'UNKNOWN'} / {retrieval_note}")
            report = report.replace("{error_message}", error_msg[:500])
            # conclusion_title 和 conclusion_content 由 LLM 分析结果填充
            report = report.replace("{conclusion_title}", "诊断结论")
            report = report.replace("{conclusion_content}", analysis)
            report = report.replace("{extra_content}", "")
            report = report.replace("{status}", "待处理-平台修复")
            report = report.replace("{analysis_path}", f"Slow Path - {self.fast_path_category or 'UNKNOWN'} / 检索{'命中' if retrieval_hit else 'MISS'}")
            report = report.replace("{timestamp}", timestamp)
        else:
            # 无模板时的 fallback
            report = f"""# CCE 自动化测试失败分析报告

## 一、基础信息

| 字段 | 值 |
|------|-----|
| **失败用例** | {scenario} |
| **所属模块** | {module} |
| **失败步骤** | {step_name} |
| **错误类型** | {log_info.get('type', 'error')} |

## 二、错误摘要

**类型判定**：`{self.fast_path_category or 'UNKNOWN'} / {retrieval_note}`

**错误信息**：
```
{error_msg[:500]}
```

## 三、诊断结论

{analysis}

---

**状态**：待处理-平台修复
**分析路径**：Slow Path - {self.fast_path_category or 'UNKNOWN'} / 检索{'命中' if retrieval_hit else 'MISS'}
**生成时间**：{timestamp}
"""

        # 保存排查单 - 文件名包含 case_id，根据类型保存到对应子目录
        ticket_dir = Path("tickets")

        # 根据 Fast Path 分类确定子目录
        if self.fast_path_category == 'VERSION_BUG':
            sub_dir = ticket_dir / "version_bug"
        elif self.fast_path_category == 'ENV_FAULT':
            sub_dir = ticket_dir / "env_fault"
        elif self.fast_path_category == 'SCRIPT_ERROR':
            sub_dir = ticket_dir / "script_error"
        elif self.fast_path_category == 'RESOURCE_INSUFFICIENT':
            sub_dir = ticket_dir / "resource_insufficient"
        else:
            sub_dir = ticket_dir

        sub_dir.mkdir(exist_ok=True)
        ticket_path = sub_dir / f"{ticket_id}.md"

        try:
            with open(ticket_path, "w", encoding="utf-8") as f:
                f.write(report)
        except Exception as e:
            print(f"[SlowPath] 保存排查单失败: {e}")
            ticket_path = sub_dir / f"{ticket_id}_fallback.txt"
            with open(ticket_path, "w", encoding="utf-8") as f:
                f.write(report)

        return str(ticket_path)

    def update_knowledge_base(
        self,
        ticket_id: str,
        resolution: str,
        kb3_dir: str
    ):
        """
        问题解决后，更新bug知识库

        Args:
            ticket_id: 排查单ID
            resolution: 解决方案
            kb3_dir: 向量数据库路径
        """
        # TODO: 实现知识库更新逻辑
        # 1. 读取 tickets/{ticket_id}.md
        # 2. 追加新的 Bug 记录到 KB3 原始文档
        # 3. 重新 Embedding 并存入 ChromaDB
        pass
