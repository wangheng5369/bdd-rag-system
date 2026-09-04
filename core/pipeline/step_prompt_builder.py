"""
Step 级匹配 + Prompt 组装器
对每个用例的每个步骤单独做 KB2 匹配，未命中则从 KB1 取 SDK
输出 prompt 文件（用于 LLM 调用前）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

from core.retrieval.retrieval import Retrieval
from core.step_writer import StepWriter
from core.config import RECALL_THRESHOLD, RERANKER_THRESHOLD, KB1_RECALL_THRESHOLD


@dataclass
class StepMatchResult:
    """单个步骤的匹配结果"""
    original_step: str           # 原始步骤文本
    matched_bdd: Optional[str]   # 命中的已有 BDD 语句（字面量）
    is_new: bool                 # 是否需要新写
    sdk_info: Optional[Dict]     # KB1 中匹配的 SDK 信息
    rerank_score: float          # 重排分数
    # Reference 字段（追踪知识库来源）
    reference_type: str = ""     # existing_bdd / new_sdk / param_config
    reference_source: str = ""    # kb1 / kb2 / kb3
    reference_id: str = ""       # chunk_xxx / api_xxx


@dataclass
class UseCasePromptResult:
    """单个用例的 prompt 组装结果"""
    use_case_id: str
    use_case_name: str
    step_results: List[StepMatchResult]
    prompt: str                  # 组装好的完整 prompt


class StepPromptBuilder:
    """
    对每个用例的每个步骤单独匹配，组装 prompt
    """

    # BDD Step 匹配阈值（向量检索）
    SIMILARITY_THRESHOLD = RECALL_THRESHOLD
    # Reranker 阈值（低于此值认为需要新写）
    RERANK_THRESHOLD = RERANKER_THRESHOLD
    # KB1 SDK 匹配时使用更低的初始阈值（让更多候选通过，再由硬规则过滤）
    KB1_SIMILARITY_THRESHOLD = KB1_RECALL_THRESHOLD

    def __init__(self, kb1_dir: str, kb2_dir: str):
        self.retrieval = Retrieval(
            persist_dirs={"kb1": kb1_dir, "kb2": kb2_dir},
            similarity_threshold=self.SIMILARITY_THRESHOLD,
            reranker_threshold=self.RERANK_THRESHOLD
        )
        self.step_writer = StepWriter()

    def build_for_use_case(self, use_case) -> UseCasePromptResult:
        """
        对单个用例的所有步骤进行匹配和 prompt 组装

        Args:
            use_case: UseCase 实例

        Returns:
            UseCasePromptResult: 包含每步匹配结果和组装好的 prompt
        """
        step_results: List[StepMatchResult] = []

        print(f"\n{'='*60}")
        print(f"用例: {use_case.id} - {use_case.name}")
        print(f"{'='*60}")

        for i, step_text in enumerate(use_case.steps, 1):
            print(f"\n  步骤 {i}: {step_text}")

            # Step 级 KB2 检索
            kb2_results, did_fallback = self.retrieval.retrieve_with_fallback(
                query=step_text,
                primary_kb="kb2",
                fallback_kb="kb1",
                top_k=5
            )

            if kb2_results and kb2_results[0]["score"] >= self.RERANK_THRESHOLD:
                # 命中已有 BDD
                matched_bdd = kb2_results[0]["content"]
                score = kb2_results[0]["score"]
                print(f"    ✅ 命中 KB2 (score={score:.4f})")
                print(f"       {matched_bdd[:80]}...")

                # 判断命中的 BDD 是否是 When 操作步骤
                # 如果只是 Then/Given（验证/预置），仍需要从 KB1 找对应的 When 操作 API
                gherkin_lines = self._extract_gherkin(matched_bdd)
                has_when_step = any(gl.startswith("When ") for gl in gherkin_lines)

                sdk_info = None
                if not has_when_step:
                    # 命中的只是 Then/Given，需要额外查 KB1 获取 When 操作 API
                    kb1_raw = self.retrieval.retrieve(
                        query=step_text,
                        kb_name="kb1",
                        top_k=5,
                        threshold=self.KB1_SIMILARITY_THRESHOLD
                    )
                    kb1_filtered = self.retrieval.filter_by_action_conflict(step_text, kb1_raw)
                    kb1_reranked = self._rerank_by_resource_match(step_text, kb1_filtered)
                    if kb1_reranked:
                        sdk_info = kb1_reranked[0]
                        print(f"    📦 KB1 补充 When API (score={sdk_info['score']:.4f})")
                        print(f"       {sdk_info['content'][:60]}...")

                step_results.append(StepMatchResult(
                    original_step=step_text,
                    matched_bdd=matched_bdd,
                    is_new=False,
                    sdk_info=sdk_info,
                    rerank_score=score,
                    reference_type="existing_bdd",
                    reference_source="kb2",
                    reference_id=self._extract_chunk_id(matched_bdd)
                ))
            else:
                # 未命中，需要新写
                print(f"    ❌ KB2 未命中，需要新写 Step")

                # 查 KB1 获取 SDK（使用更低阈值，让更多候选通过）
                kb1_results = self.retrieval.retrieve(
                    query=step_text,
                    kb_name="kb1",
                    top_k=5,
                    threshold=self.KB1_SIMILARITY_THRESHOLD
                )
                # 对 KB1 结果做动作冲突过滤
                kb1_results = self.retrieval.filter_by_action_conflict(step_text, kb1_results)
                # 对 KB1 结果按资源关键词重排序（优先匹配 query 中的资源名）
                kb1_results = self._rerank_by_resource_match(step_text, kb1_results)

                # 取得分最高的 SDK
                best_sdk = kb1_results[0] if kb1_results else None
                if best_sdk:
                    print(f"    📦 KB1 SDK (score={best_sdk['score']:.4f})")
                    print(f"       {best_sdk['content'][:80]}...")

                # 构建 Reference 信息
                if best_sdk:
                    api_name = self._extract_api_name(best_sdk.get("content", ""))
                    ref_type = "param_config" if "设置" in step_text else "new_sdk"
                    step_results.append(StepMatchResult(
                        original_step=step_text,
                        matched_bdd=None,
                        is_new=True,
                        sdk_info=best_sdk,
                        rerank_score=best_sdk["score"],
                        reference_type=ref_type,
                        reference_source="kb1",
                        reference_id=f"api_{api_name}" if api_name else "unknown"
                    ))
                else:
                    step_results.append(StepMatchResult(
                        original_step=step_text,
                        matched_bdd=None,
                        is_new=True,
                        sdk_info=None,
                        rerank_score=0.0,
                        reference_type="missing",
                        reference_source="",
                        reference_id=""
                    ))

        # 组装 prompt
        prompt = self._assemble_prompt(use_case, step_results)

        return UseCasePromptResult(
            use_case_id=use_case.id,
            use_case_name=use_case.name,
            step_results=step_results,
            prompt=prompt
        )

    def _assemble_prompt(self, use_case, step_results: List[StepMatchResult]) -> str:
        """
        组装完整的 User Prompt

        注意：Role 和约束已移至 System Prompt，这里只包含任务相关的内容
        """
        # 分离已有和新写步骤
        existing_steps = [r for r in step_results if not r.is_new]
        new_steps = [r for r in step_results if r.is_new]

        # KB2 命中但只有 Then/Given（无 When）的步骤，也需要补充 SDK 到 Context 2
        existing_need_sdk = [r for r in existing_steps if r.sdk_info is not None]

        # 构建 Context 1：已有的 BDD 步骤（不包含需要补充 SDK 的）
        existing_for_ctx1 = [r for r in existing_steps if r.sdk_info is None]
        context1 = self._build_context1(existing_for_ctx1)

        # 构建 Context 2：SDK 信息（包含新写步骤 + 命中后需补充 SDK 的步骤）
        context2 = self._build_context2(new_steps + existing_need_sdk)

        # 构建 Context 3：原始用例
        context3 = self._build_context3(use_case)

        # 统计信息
        new_step_count = len(new_steps)
        existing_step_count = len(existing_for_ctx1)
        sdk_api_count = len(set(r.sdk_info.get("content", "")[:50] for r in new_steps if r.sdk_info))

        prompt = f"""# Task
将【待转换的原始文本用例】翻译并组装成标准的 Gherkin 格式 BDD 场景，并为需要新写的 BDD 步骤绑定对应的 SDK 调用。

## 本次任务统计

- 已有 BDD 步骤可复用：{existing_step_count} 个
- 需要新写的步骤：{new_step_count} 个
- 可用的 SDK API：{sdk_api_count} 个

---

# Context 1: 已有匹配到的 BDD 步骤定义库

> 规则：必须原封不动复用以下已有语句，不得修改措辞

{context1}

---

# Context 2: 可用的 SDK 函数知识库

> 规则：仅使用以下 API 创建新 BDD 语句和 Python 实现

{context2}

---

# Context 3: 待转换的原始文本用例

{context3}

---

# 输出要求

请按以下顺序输出三个板块：

## 板块一：.feature 文件

输出完整的 Feature 文件内容，使用标准 Gherkin 格式。

## 板块二：.py 文件

仅输出新写步骤的 Python pytest-bdd 实现代码。

## 板块三：Reference 映射表

每个 step 必须标注来源，格式如下：

```reference
## Reference
| Step | Type | Source | ID |
|------|------|--------|-----|
| Given 已存在可用集群 | existing_bdd | kb2 | chunk_001 |
| When 构造创建ECS虚机请求 | new_sdk | kb1 | api_create_ecs |
| ... | ... | ... | ... |
```
"""
        return prompt.strip()

    def _build_context1(self, existing_steps: List[StepMatchResult]) -> str:
        """
        构建 Context 1：已有 BDD 步骤（表格格式，含 ID）

        格式：
        | ID | BDD 语句 |
        |----|----------|
        | kb2/chunk_xxx | Given ... |
        """
        if not existing_steps:
            return "（无，已有的 BDD 步骤为空，所有步骤均需新写）"

        # 构建表格
        lines = ["| ID | BDD 语句 |", "|----|----------|"]
        seen = set()
        for r in existing_steps:
            # 提取 Gherkin 行
            gherkin_lines = self._extract_gherkin(r.matched_bdd)
            for gl in gherkin_lines:
                gl = gl.strip()
                if gl and gl not in seen:
                    seen.add(gl)
                    # reference_id 格式: kb2/chunk_xxx
                    ref_id = f"{r.reference_source}/{r.reference_id}" if r.reference_id else "unknown"
                    lines.append(f"| {ref_id} | {gl} |")

        return "\n".join(lines) if len(lines) > 2 else "（无）"

    def _build_context2(self, new_steps: List[StepMatchResult]) -> str:
        """
        构建 Context 2：SDK 信息（表格格式，含 ID）

        格式：
        | ID | API 名称 | 签名 |
        |----|----------|------|
        | kb1/api_xxx | create_ecs | def create_ecs(...) |
        """
        if not new_steps:
            return "（无，所有步骤均已匹配到已有 BDD）"

        # 按 API 去重
        api_map = {}  # api_name -> (ref_id, content)
        for r in new_steps:
            if r.sdk_info:
                content = r.sdk_info.get("content", "")
                api_name = self._extract_api_name(content)
                ref_id = f"{r.reference_source}/{r.reference_id}" if r.reference_id else f"kb1/unknown"

                if api_name and api_name not in api_map:
                    api_map[api_name] = (ref_id, content)

        if not api_map:
            return "（无，所有步骤均已匹配到已有 BDD）"

        # 构建表格头部
        lines = ["| ID | API 名称 | 签名/代码 |", "|----|----------|------|"]

        for api_name, (ref_id, content) in api_map.items():
            # 提取函数签名
            signature = self._extract_sdk_signature(content)
            lines.append(f"| {ref_id} | {api_name} | `{signature}` |")

        result = "\n".join(lines)

        # 添加详细代码块
        result += "\n\n### API 详细代码\n"
        for api_name, (ref_id, content) in api_map.items():
            result += f"\n#### {api_name} ({ref_id})\n"
            result += self._extract_sdk_block(content)

        return result

    def _build_context3(self, use_case) -> str:
        """构建 Context 3：原始用例"""
        steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(use_case.steps)])
        expected_text = use_case.expected.replace("####", "").strip()

        return f"""【用例编号】：{use_case.id}
【用例名称】：{use_case.name}
【用例步骤】：
{steps_text}
【预期结果】：
{expected_text}"""

    def _extract_gherkin(self, content: str) -> List[str]:
        """从 BDD 文档内容中提取 Gherkin 行"""
        import re
        lines = []
        # 匹配 ```gherkin 或 ``` 包围的代码块中的内容
        for block in re.findall(r"```gherkin\s*\n(.*?)```", content, re.DOTALL):
            for line in block.split("\n"):
                line = line.strip()
                if line and (line.startswith(("Given ", "When ", "Then ", "And "))):
                    lines.append(line)
        # 回退：如果没有 gherkin 代码块，尝试直接匹配 Given/When/Then/And 开头的行
        if not lines:
            for line in content.split("\n"):
                line = line.strip()
                if line and (line.startswith(("Given ", "When ", "Then ", "And "))):
                    lines.append(line)
        return lines

    def _extract_api_name(self, content: str) -> str:
        """从 SDK 文档中提取 API 名称"""
        import re
        # 匹配 "接口名称：" 或 "def func_name("
        m = re.search(r"接口名称[：:]\s*(.+)", content)
        if m:
            return m.group(1).strip()
        m = re.search(r"def\s+(\w+)\s*\(", content)
        if m:
            return m.group(1).strip()
        return ""

    def _extract_sdk_signature(self, content: str) -> str:
        """从 SDK 文档中提取函数签名"""
        import re
        # 匹配 def func_name(...) 的第一行
        m = re.search(r"(def\s+\w+\s*\([^)]*\)[^:]*?)[:\n]", content)
        if m:
            return m.group(1).strip()
        # 回退：提取第一行 def
        m = re.search(r"(def\s+\w+\s*\([^)]*\))", content)
        if m:
            return m.group(1).strip()
        return "(签名提取失败)"

    def _extract_chunk_id(self, content: str) -> str:
        """从 BDD 文档中提取 chunk ID"""
        import re
        # 尝试从 metadata 中提取 chunk_id
        m = re.search(r'chunk[_-]?id[：:\s]*([^\s\]#]+)', content, re.IGNORECASE)
        if m:
            return f"chunk_{m.group(1).strip()}"
        # 尝试从标题路径提取
        m = re.search(r'##\s+(.+?)(?:\s*/|$)', content)
        if m:
            title = m.group(1).strip()
            # 转换为安全的 ID 格式
            safe_id = re.sub(r'[^\w]+', '_', title)[:20]
            return f"chunk_{safe_id}"
        return "chunk_unknown"

    def _build_reference_table(self, step_results: List[StepMatchResult]) -> str:
        """
        从 step_results 生成 Reference 映射表

        Returns:
            Markdown 格式的 Reference 表
        """
        lines = ["| Step | Type | Source | ID |", "|------|------|--------|-----|"]
        for r in step_results:
            # 提取 step 的 Gherkin 描述
            if r.matched_bdd:
                gherkin_lines = self._extract_gherkin(r.matched_bdd)
                step_text = gherkin_lines[0] if gherkin_lines else r.original_step
            else:
                step_text = r.original_step

            # 截断过长的 step 文本
            if len(step_text) > 50:
                step_text = step_text[:47] + "..."

            lines.append(f"| {step_text} | {r.reference_type} | {r.reference_source} | {r.reference_id} |")

        return "\n".join(lines)

    def _extract_sdk_block(self, content: str) -> str:
        """提取 SDK 函数签名和示例块"""
        import re
        # 提取 python 代码块
        match = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
        if match:
            return f"```python\n{match.group(1).strip()}\n```"
        # 回退：提取函数签名行
        match = re.search(r"(def \w+\s*\([^)]*\).*?)(?:\n\n|\Z)", content, re.DOTALL)
        if match:
            return f"```python\n{match.group(1).strip()}\n```"
        return content[:200]

    def _rerank_by_resource_match(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        对 KB1 结果按资源关键词匹配重排序

        从 query 中提取资源名（如"节点"→node，"集群"→cluster），
        优先选择函数名/接口名中包含对应资源词的 API。

        Args:
            query: 查询文本
            results: KB1 检索结果

        Returns:
            重排序后的结果
        """
        import re

        # 从 query 中提取资源词
        resource_map = {
            "node": ["node", "节点"],
            "cluster": ["cluster", "集群"],
            "nodepool": ["nodepool", "节点池"],
            "pvc": ["pvc", "存储"],
            "deployment": ["deployment", "工作负载"],
        }

        # 找出 query 中有哪些资源词
        matched_resources = []
        for key, keywords in resource_map.items():
            for kw in keywords:
                if kw in query:
                    matched_resources.append(key)
                    break

        if not matched_resources or not results:
            return results

        def resource_score(item: Dict) -> float:
            """计算资源匹配分——仅在函数名/接口名中按单词分割精确匹配"""
            import re
            content = item.get("content", "")

            # 提取函数/API名（优先接口名称，其次 def）
            api_name = ""
            m = re.search(r'接口名称[：:]\s*(\S+)', content)
            if m:
                api_name = m.group(1)
            else:
                m = re.search(r'def\s+(\w+)', content)
                if m:
                    api_name = m.group(1)

            # 按下划线分割
            parts = api_name.split('_')

            # 计算资源匹配分
            resource_match_count = sum(1 for res in matched_resources if res in parts)

            # 计算动作匹配分：从 api_name 的动作前缀判断
            # create_xxx, delete_xxx, reset_xxx, add_xxx, get_xxx, update_xxx
            action_prefix = parts[0] if parts else ""
            action_map = {
                "create": "CREATE", "delete": "DELETE", "reset": "UPDATE",
                "add": "CREATE", "update": "UPDATE", "get": "QUERY", "set": "UPDATE"
            }
            api_action = action_map.get(action_prefix, "")

            # 从 query 检测动作
            from core.retrieval.retrieval import detect_action_group
            query_actions = detect_action_group(query)
            action_match = api_action in query_actions if api_action else False

            # 综合分：资源匹配 + 动作匹配加权
            score = 0.0
            if resource_match_count > 0 and action_match:
                score = 20.0  # 资源+动作都匹配，加最高分
            elif resource_match_count > 0:
                score = 5.0   # 仅资源匹配，低加分（可能选错了动作）

            return score

        # 重排序：原分数 + 资源匹配分
        reranked = []
        for item in results:
            item = dict(item)  # 复制
            item["_resource_score"] = resource_score(item)
            item["score"] = item["score"] + item["_resource_score"]
            reranked.append(item)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked


def run_and_output(use_case_file: str, kb1_dir: str, kb2_dir: str, output_dir: str):
    """运行完整流程并输出 prompt 文件"""
    from core.pipeline import load_use_cases

    use_cases = load_use_cases(use_case_file)
    builder = StepPromptBuilder(kb1_dir, kb2_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_prompts = {}

    for uc in use_cases:
        result = builder.build_for_use_case(uc)
        prompt_path = Path(output_dir) / f"{uc.id}_prompt.md"

        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(result.prompt)

        all_prompts[uc.id] = {
            "prompt_path": str(prompt_path),
            "steps_summary": [
                {
                    "original": r.original_step,
                    "is_new": r.is_new,
                    "score": r.rerank_score
                }
                for r in result.step_results
            ]
        }

        print(f"\n✅ Prompt 已输出: {prompt_path}")

    # 汇总
    summary_path = Path(output_dir) / "_prompt_summary.json"
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=2)
    print(f"\n📋 汇总: {summary_path}")

    return all_prompts


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    use_case_file = project_root / "docs/用例示例.md"
    kb1_dir = project_root / "knowledge-base/kb1_aw_sdk/indexed/chroma_db"
    kb2_dir = project_root / "knowledge-base/kb2_bdd_scenarios/indexed/chroma_db"
    output_dir = project_root / "generated_scripts/prompts"

    run_and_output(
        use_case_file=str(use_case_file),
        kb1_dir=str(kb1_dir),
        kb2_dir=str(kb2_dir),
        output_dir=str(output_dir)
    )
