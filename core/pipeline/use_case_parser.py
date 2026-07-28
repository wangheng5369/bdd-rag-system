"""
用例解析器 - 从 docs/用例示例.md 解析出结构化用例列表
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class UseCase:
    """结构化用例"""
    id: str            # e.g. "Tc_Func_Node_001"
    name: str          # e.g. "创建100G数据盘的节点"
    query: str         # 用于检索的 query
    steps: List[str]  # 用例步骤文本列表
    expected: str      # 预期结果


def parse_use_cases(file_path: str) -> List[UseCase]:
    """
    解析用例文档，返回结构化用例列表

    Args:
        file_path: 用例文档路径

    Returns:
        List[UseCase]: 用例列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 用例以 "## " 开头，后面是用例名称
    # 用例ID在 "### 用例编号：" 后面
    # 用例名称在 "### 用例名称：" 后面
    # 用例步骤在 "### 用例步骤：" 下面
    # 预期结果在 "### 预期结果：" 下面

    cases = []
    # 分割每个用例（以 ### 用例编号 开头的大标题作为分界）
    case_blocks = re.split(r"\n(?=### 用例编号[：:])", "\n" + content)

    for block in case_blocks:
        block = block.strip()
        if not block:
            continue

        # 提取用例编号
        id_match = re.search(r"### 用例编号[：:]\s*(\S+)", block)
        if not id_match:
            continue
        case_id = id_match.group(1).strip()

        # 提取用例名称
        name_match = re.search(r"### 用例名称[：:]\s*(.+?)(?=\n)", block)
        case_name = name_match.group(1).strip() if name_match else ""

        # 提取步骤（### 用例步骤： 到 ### 预期结果： 之间的内容）
        steps_match = re.search(
            r"### 用例步骤[：:]\s*\n((?:.*?\n)*?)(?=### 预期结果|$)",
            block,
            re.DOTALL
        )
        steps_text = steps_match.group(1).strip() if steps_match else ""
        steps = _parse_steps(steps_text)

        # 提取预期结果
        expected_match = re.search(
            r"### 预期结果[：:]\s*\n((?:.*?\n)*)",
            block,
            re.DOTALL
        )
        expected = expected_match.group(1).strip() if expected_match else ""

        # 构建检索 query（用例名称 + 步骤拼接）
        query = _build_query(case_name, steps)

        cases.append(UseCase(
            id=case_id,
            name=case_name,
            query=query,
            steps=steps,
            expected=expected
        ))

    return cases


def _parse_steps(steps_text: str) -> List[str]:
    """解析步骤文本，提取每一步（支持 #### 前缀）"""
    lines = steps_text.split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        # 匹配 "#### 1、xxx" 或 "1、xxx" 格式
        m = re.match(r"^#{1,6}\s*\d+[、.]\s*(.+)$", line)
        if m:
            steps.append(m.group(1).strip())
    return steps


def _build_query(name: str, steps: List[str]) -> str:
    """从用例名称和步骤构建检索 query"""
    parts = [name]
    for step in steps:
        # 去掉编号前缀
        step = re.sub(r"^\d+[、.]\s*", "", step)
        parts.append(step)
    return " ".join(parts)


def load_use_cases(file_path: str = "docs/用例示例.md") -> List[UseCase]:
    """
    加载用例文件的便捷函数

    Args:
        file_path: 用例文档路径

    Returns:
        List[UseCase]: 用例列表
    """
    path = Path(file_path)
    if not path.is_absolute():
        # 相对于项目根目录
        project_root = Path(__file__).parent.parent.parent
        path = project_root / file_path
    return parse_use_cases(str(path))
