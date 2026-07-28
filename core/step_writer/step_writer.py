"""
StepWriter 模块
当 KB2 召回 MISS 时，从 KB1 检索 SDK 信息，调用 LLM 生成 BDD Step 链
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from core.llm import LLMGenerator


@dataclass
class Step:
    """单个 BDD Step"""
    step_type: str      # "When" / "And" / "Then" / "Given"
    gherkin: str        # 完整的 Gherkin 行
    api_call: str       # 最终 API 调用名
    impl_hint: str      # Python 实现提示


@dataclass
class SDKInfo:
    """解析后的 SDK 信息"""
    api_name: str              # API 名称（中文）
    func_name: str             # 函数名（英文）
    module: str                # 所属模块
    signature: str             # 函数签名
    params: List[Dict]          # 参数列表 [{name, type, required, default, description}]
    returns: str                # 返回值描述
    example: str               # 调用示例
    description: str           # 简要说明


class StepWriter:
    """
    Step 生成器
    接收 intent + KB1 检索结果，调用 LLM 生成 When/And/Then Step 链
    """

    # 意图 → 标准 API 前缀映射
    INTENT_API_PREFIX = {
        "纳管": "register",
        "注册": "register",
        "创建": "create",
        "删除": "delete",
        "查询": "get",
        "获取": "get",
        "更新": "update",
        "修改": "update",
        "启用": "enable",
        "停用": "disable",
    }

    def __init__(self, llm: Optional[LLMGenerator] = None):
        """
        初始化 StepWriter

        Args:
            llm: LLMGenerator 实例，默认自动创建
        """
        self.llm = llm or LLMGenerator()

    def write_steps(
        self,
        intent: str,
        kb1_results: List[Dict],
        example_values: Optional[Dict] = None
    ) -> List[Step]:
        """
        根据意图和 KB1 SDK 信息生成 BDD Step 链

        Args:
            intent: 用户意图，如 "纳管node"、"创建集群"
            kb1_results: KB1 检索结果列表，每项包含 content 和 metadata
            example_values: 可选的示例值，如 {"name": "node-01", "disk_size": 100}

        Returns:
            List[Step]: 生成的 Step 列表
        """
        example_values = example_values or {}

        # 1. 从 KB1 结果中解析 SDK 信息
        sdk_info = self._parse_sdk(kb1_results, intent)
        if not sdk_info:
            print(f"[StepWriter] 未从 KB1 中解析到 SDK 信息，intent={intent}")
            return []

        # 2. 组装 prompt
        prompt = self._build_prompt(intent, sdk_info, example_values)

        # 3. 调用 LLM 生成 Step
        raw_output = self.llm.generate(prompt)
        if raw_output.startswith("[LLMGenerator"):
            print(f"[StepWriter] LLM 未配置或调用失败: {raw_output}")
            return []

        # 4. 解析 LLM 输出为 Step 列表
        steps = self._parse_llm_output(raw_output, sdk_info.func_name)
        return steps

    def _parse_sdk(self, kb1_results: List[Dict], intent: str) -> Optional[SDKInfo]:
        """
        从 KB1 检索结果中解析 SDK 信息

        Args:
            kb1_results: KB1 检索结果
            intent: 用户意图

        Returns:
            SDKInfo 或 None
        """
        if not kb1_results:
            return None

        # 优先使用得分最高的结果
        best = kb1_results[0]
        content = best.get("content", "")
        metadata = best.get("metadata", {})

        # 解析 API 名称（从标题路径）
        title_path = metadata.get("title_path", "")
        api_name = self._extract_api_name_from_title(title_path, content, intent)

        # 解析函数签名
        signature = self._extract_signature(content)

        # 解析参数表
        params = self._extract_params(content)

        # 解析示例
        example = self._extract_example(content)

        # 解析描述
        description = self._extract_description(content)

        # 从 content 中提取函数名
        func_name = self._extract_func_name(content, intent)

        if not signature and not func_name:
            return None

        return SDKInfo(
            api_name=api_name,
            func_name=func_name,
            module=self._extract_module(title_path),
            signature=signature,
            params=params,
            returns=self._extract_return(content),
            example=example,
            description=description
        )

    def _extract_api_name_from_title(
        self,
        title_path: str,
        content: str,
        intent: str
    ) -> str:
        """从标题路径提取 API 名称"""
        if title_path:
            # 形如 "Node Module / 2. 注册节点（When）"
            parts = title_path.split("/")
            if len(parts) >= 2:
                return parts[-1].strip()
            return title_path.strip()

        # 回退：从 content 第一行提取
        lines = content.split("\n")
        for line in lines:
            if "接口名称" in line or "API名称" in line:
                return line.split("：")[-1].strip()

        return intent

    def _extract_func_name(self, content: str, intent: str) -> str:
        """从 content 中提取函数名（英文）"""
        # 匹配 def xxx(...) 模式
        match = re.search(r"def\s+(\w+)\s*\(", content)
        if match:
            return match.group(1)

        # 意图前缀匹配
        for prefix, eng in self.INTENT_API_PREFIX.items():
            if prefix in intent:
                return f"{eng}_node"  # 默认后缀，可改进

        return "unknown_func"

    def _extract_signature(self, content: str) -> str:
        """提取函数签名"""
        # 匹配 ```python ... def func(...) ... ``` 块
        match = re.search(
            r"```python\s*\n(.*?def\s+\w+\s*\([^)]*\).*?)\n```",
            content,
            re.DOTALL
        )
        if match:
            sig = match.group(1).strip()
            # 清理：移除注释行
            lines = [l for l in sig.split("\n") if not l.strip().startswith("#")]
            return "\n".join(lines)

        # 回退：直接搜索 def 行
        match = re.search(r"(def\s+\w+\s*\([^)]*\)[^`]*?)(?:\n```|$)", content, re.DOTALL)
        if match:
            return match.group(1).strip()

        return ""

    def _extract_params(self, content: str) -> List[Dict]:
        """提取参数表"""
        params = []

        # 匹配 Markdown 表格行
        # | 参数名 | 类型 | 是否必填 | 默认值 | 说明 |
        table_match = re.search(
            r"参数名.*?\n\|[-:\s|]+\|\n((?:.*?\n)*?)(?=\n\S|\n```|$)",
            content,
            re.DOTALL
        )

        if table_match:
            table_lines = table_match.group(1).strip().split("\n")
            for line in table_lines:
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) >= 2 and cells[0] and cells[0] != "参数名":
                    param = {
                        "name": cells[0].strip("`").strip(),
                        "type": cells[1] if len(cells) > 1 else "str",
                        "required": "是" in cells[2] if len(cells) > 2 else False,
                        "default": cells[3] if len(cells) > 3 else None,
                        "description": cells[4] if len(cells) > 4 else "",
                    }
                    params.append(param)

        return params

    def _extract_example(self, content: str) -> str:
        """提取调用示例"""
        match = re.search(
            r"```python\s*\n(.*?)\n```",
            content,
            re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return ""

    def _extract_return(self, content: str) -> str:
        """提取返回值描述"""
        match = re.search(r"返回类型[：:]\s*[`\"]*(\w+)[`\"]*", content)
        if match:
            return match.group(1)
        match = re.search(r"返回[：:]\s*(.+?)(?:\n|$)", content)
        if match:
            return match.group(1).strip()
        return "dict"

    def _extract_description(self, content: str) -> str:
        """提取简要说明"""
        match = re.search(r"简要说明[：:]\s*(.+?)(?:\n|$)", content)
        if match:
            return match.group(1).strip()

        # 回退：取第一段非标题文本
        lines = content.split("\n")
        for line in lines:
            if line.strip() and not line.startswith("#") and not line.startswith("```"):
                return line.strip()[:200]
        return ""

    def _extract_module(self, title_path: str) -> str:
        """提取模块名"""
        if title_path:
            parts = title_path.split("/")
            if len(parts) >= 2:
                return parts[0].strip()
        return "unknown"

    def _build_prompt(
        self,
        intent: str,
        sdk: SDKInfo,
        example_values: Dict
    ) -> str:
        """组装 Step 生成的 prompt"""
        # 构建参数表
        params_lines = []
        for p in sdk.params:
            required_mark = "**是**" if p["required"] else "否"
            default = p.get("default", "无")
            params_lines.append(
                f"| `{p['name']}` | {p['type']} | {required_mark} | {default} | {p['description']} |"
            )
        params_table = "\n".join(params_lines) if params_lines else "（无可用参数）"

        # 示例值
        example_parts = []
        for k, v in example_values.items():
            example_parts.append(f"{k}={v!r}")
        example_vals_str = ", ".join(example_parts) if example_parts else "无"

        prompt = f"""你是一个 BDD Step 生成器。根据以下信息生成 When/And/Then 步骤链。

## 背景
{sdk.description}

## 意图
用户意图：{intent}
API 名称：{sdk.api_name}
所属模块：{sdk.module}

## API 信息
- 函数名：{sdk.func_name}
- 函数签名：
```python
{sdk.signature}
```
- 参数表：
| 参数名 | 类型 | 是否必填 | 默认值 | 说明与限制 |
|--------|------|----------|--------|------------|
{params_table}
- 返回值：{sdk.returns}
- 调用示例：
```python
{sdk.example}
```

## 示例值（来自用户 Query）
{example_vals_str}

## Step 生成规则
1. 第一步：When 构造{intent}请求，名称为"xxx"（从用户 Query 提取具体名称）
2. 中间步骤：And 设置xxx参数（仅设置用户 Query 中提到的参数，不要设置所有参数）
3. 最后一步：When 基于构造的请求，{intent}（执行 API 调用）
4. 步骤数量根据用户 Query 决定，Query 提到什么参数就设置什么参数
5. 参数名称使用中文自然语言描述，如 "disk_size" → "磁盘大小"，"labels" → "资源标签"
6. 仅输出 Step 列表，每行一条 Gherkin 语句，不要输出其他内容

## 输出格式
逐行输出，每行一条 Gherkin 语句，例如：
When 构造{intent}请求，名称为"node-worker-01"
And 设置node的磁盘大小为100G
And 设置node的资源标签为{{"a":"b"}}
When 基于构造的请求，{intent}
"""
        return prompt.strip()

    def _parse_llm_output(self, raw_output: str, func_name: str) -> List[Step]:
        """解析 LLM 输出为 Step 列表"""
        steps = []
        lines = raw_output.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 提取 step type (When/And/Then/Given)
            step_type = None
            if line.startswith("When "):
                step_type = "When"
            elif line.startswith("And "):
                step_type = "And"
            elif line.startswith("Then "):
                step_type = "Then"
            elif line.startswith("Given "):
                step_type = "Given"
            else:
                # 忽略无法识别的行
                continue

            # 构建 impl_hint
            impl_hint = self._build_impl_hint(line, func_name)

            steps.append(Step(
                step_type=step_type,
                gherkin=line,
                api_call=func_name,
                impl_hint=impl_hint
            ))

        return steps

    def _build_impl_hint(self, gherkin: str, func_name: str) -> str:
        """根据 Gherkin 行构建实现提示"""
        # 这是一个简单的提示，实际可更复杂
        return f"调用 {func_name}() 实现此步骤"


def create_step_writer(llm: Optional[LLMGenerator] = None) -> StepWriter:
    """
    工厂函数：创建 StepWriter

    Args:
        llm: LLMGenerator 实例

    Returns:
        StepWriter 实例
    """
    return StepWriter(llm=llm)
