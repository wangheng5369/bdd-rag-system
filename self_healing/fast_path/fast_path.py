"""
Fast Path - 快速路径处理
处理确定性高、简单的错误：脚本问题、环境故障、资源不足
"""

import time
import re
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional


class ErrorCategory(Enum):
    SCRIPT_ERROR = "SCRIPT_ERROR"          # 脚本问题（语法、KeyError、类型错误等）
    ENV_FAULT = "ENV_FAULT"                # 环境故障（401/403/404/Timeout等）
    RESOURCE_INSUFFICIENT = "RESOURCE"    # 资源不足（配额、库存、磁盘空间等）
    VERSION_BUG = "VERSION_BUG"           # 版本bug（需RAG+LLM分析）
    UNKNOWN = "UNKNOWN"                    # 未知


class FastPathHealer:
    """快速路径自愈处理器"""

    # 错误类型关键词
    SCRIPT_ERROR_KEYWORDS = [
        "KeyError", "TypeError", "ValueError", "AttributeError",
        "IndentationError", "SyntaxError", "NameError",
        "ImportError", "ModuleNotFoundError",
        "deprecated", "removed", "Field is immutable",
    ]

    ENV_FAULT_KEYWORDS = [
        "401", "403", "404", "500", "502", "503", "504",
        "Unauthorized", "Forbidden", "Not Found", "Timeout",
        "Connection refused", "Connection reset",
        "network", "unreachable",
    ]

    RESOURCE_KEYWORDS = [
        "quota", "Quota", "exceeded", "out of stock",
        "No Space", "Out of memory", "OOM",
        "capacity", "insufficient",
    ]

    def __init__(self, llm=None):
        self.llm = llm

    def diagnose_and_heal(self, log_info: dict) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """
        Fast Path 诊断

        Args:
            log_info: 日志结构化信息

        Returns:
            Tuple[bool, str, Optional[str], Optional[str]]:
            - 是否可修复
            - 处理结果描述
            - 报告路径（仅对简单错误生成报告，复杂错误返回None由SlowPath处理）
            - 错误分类（category.value）
        """
        error_msg = log_info.get('message', '')
        category = self._classify_error(error_msg)

        print(f"  [FastPath] 错误分类: {category.value}")

        # 对于脚本问题和资源不足，可直接生成报告
        if category in (ErrorCategory.SCRIPT_ERROR, ErrorCategory.RESOURCE_INSUFFICIENT):
            ticket_path = self._generate_direct_report(log_info, category)
            return True, f"{category.value} 已生成分析报告", ticket_path, category.value

        # 版本bug、环境故障等复杂问题，移交 SlowPath
        return False, f"{category.value} - 复杂问题，移交 Slow Path", None, category.value

    def _classify_error(self, error_msg: str) -> ErrorCategory:
        """根据错误信息分类"""
        msg_lower = error_msg.lower()

        # cgroup/containerd 相关 -> 版本bug（优先判断，因为可能同时含500）
        if "cgroup" in msg_lower or "containerd" in msg_lower:
            return ErrorCategory.VERSION_BUG

        # 脚本问题（语法错误、API弃用、不可变字段）
        for kw in self.SCRIPT_ERROR_KEYWORDS:
            if kw.lower() in msg_lower:
                return ErrorCategory.SCRIPT_ERROR

        # 资源不足
        for kw in self.RESOURCE_KEYWORDS:
            if kw.lower() in msg_lower:
                return ErrorCategory.RESOURCE_INSUFFICIENT

        # 环境故障
        for kw in self.ENV_FAULT_KEYWORDS:
            if kw.lower() in msg_lower:
                return ErrorCategory.ENV_FAULT

        return ErrorCategory.UNKNOWN

    def _generate_direct_report(self, log_info: dict, category: ErrorCategory) -> str:
        """为简单错误直接生成分析报告"""
        error_msg = log_info.get('message', '')
        scenario = log_info.get('scenario', '未知')
        module = log_info.get('module', 'Unknown')
        step_name = log_info.get('step_name', '')
        case_id = log_info.get('case_id', '')

        # 生成结论
        conclusion_title, conclusion_content, status, analysis_path_label = self._build_conclusion(
            error_msg, category, scenario, module
        )

        # 读取模板
        template_path = Path(__file__).parent.parent.parent / "docs" / "llm-analysis-template.md"
        try:
            template = template_path.read_text(encoding="utf-8")
        except Exception:
            template = ""

        # 填充模板
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        if template:
            report = template.replace("{scenario}", scenario)
            report = report.replace("{module}", module)
            report = report.replace("{step_name}", step_name)
            report = report.replace("{error_type}", log_info.get('type', 'error'))
            report = report.replace("{error_category}", category.value)
            report = report.replace("{error_message}", error_msg[:500])
            report = report.replace("{conclusion_title}", conclusion_title)
            report = report.replace("{conclusion_content}", conclusion_content)
            report = report.replace("{extra_content}", "")
            report = report.replace("{status}", status)
            report = report.replace("{analysis_path}", f"Fast Path - {analysis_path_label}")
            report = report.replace("{timestamp}", timestamp)
        else:
            # 无模板时的 fallback
            report = f"""# 分析报告

## 失败用例: {scenario}
## 模块: {module}
## 步骤: {step_name}
## 错误类型: {category.value}

## 错误信息
{error_msg[:500]}

## 结论
**{conclusion_title}**: {conclusion_content}

---
状态: {status}
"""

        # 保存报告 - 文件名包含 case_id，根据类型保存到对应子目录
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        ticket_dir = Path("tickets")

        # 根据错误类型确定子目录
        if category == ErrorCategory.SCRIPT_ERROR:
            sub_dir = ticket_dir / "script_error"
        elif category == ErrorCategory.RESOURCE_INSUFFICIENT:
            sub_dir = ticket_dir / "resource_insufficient"
        elif category == ErrorCategory.ENV_FAULT:
            sub_dir = ticket_dir / "env_fault"
        elif category == ErrorCategory.VERSION_BUG:
            sub_dir = ticket_dir / "version_bug"
        else:
            sub_dir = ticket_dir

        sub_dir.mkdir(exist_ok=True)
        ticket_path = sub_dir / f"{case_id}_{timestamp_str}.md" if case_id else sub_dir / f"TICKET_{timestamp_str}.md"

        with open(ticket_path, "w", encoding="utf-8") as f:
            f.write(report)

        return str(ticket_path)

    def _build_conclusion(self, error_msg: str, category: ErrorCategory,
                          scenario: str, module: str) -> Tuple[str, str, str, str]:
        """根据错误类型构建结论"""

        if category == ErrorCategory.RESOURCE_INSUFFICIENT:
            return (
                "资源不足",
                "当前测试环境配额或底层资源不足，无法完成操作。建议：(1) 联系云平台提升配额；(2) 清理不需要的资源；(3) 调整测试用例规格。",
                "待处理-资源",
                "资源不足"
            )

        elif category == ErrorCategory.SCRIPT_ERROR:
            # 细化脚本问题类型
            if "KeyError" in error_msg:
                key_match = re.search(r"KeyError: ['\"]([^'\"]+)['\"]", error_msg)
                missing_key = key_match.group(1) if key_match else "未知字段"
                return (
                    "脚本错误",
                    f"BDD Step 中访问了不存在的 context 字段 '{missing_key}'。"
                    f"可能原因：(1) 重置节点后 context 被清空；(2) 前置 Step 未正确设置该字段。"
                    f"建议检查 'When 触发重置节点操作' 后的 context 清理逻辑。",
                    "待修复-脚本",
                    "脚本错误"
                )
            elif "deprecated" in error_msg.lower() or "removed" in error_msg.lower():
                sc_match = re.search(r"StorageClass ['\"]([^'\"]+)['\"]", error_msg)
                old_sc = sc_match.group(1) if sc_match else "csi-disk-extreme"
                new_sc = "csi-disk-ssd-everest"
                return (
                    "脚本错误",
                    f"使用的 StorageClass '{old_sc}' 已在目标集群版本 (v1.27+) 中被移除。"
                    f"需要将 BDD 脚本中的 storageClassName 从 '{old_sc}' 更新为 '{new_sc}'。"
                    f"这是 API 版本兼容性问题，属于测试脚本未同步更新，而非平台 bug。",
                    "待修复-脚本",
                    "脚本错误"
                )
            elif "Field is immutable" in error_msg:
                return (
                    "脚本错误",
                    "API 拒绝了请求中的不可变字段。当前测试脚本尝试修改一个在 API v1.27 中已标记为不可变的字段。"
                    "建议检查 BDD 脚本中对该 API 的调用参数，移除对不可变字段的修改。",
                    "待修复-脚本",
                    "脚本错误"
                )
            else:
                return (
                    "脚本错误",
                    f"BDD 脚本执行时发生错误，请检查相关 Step 定义。错误信息：{error_msg[:200]}",
                    "待修复-脚本",
                    "脚本错误"
                )

        elif category == ErrorCategory.ENV_FAULT:
            if "401" in error_msg or "403" in error_msg:
                return (
                    "环境故障",
                    "API 认证失败，可能是 Token 过期或权限不足。请检查：(1) 环境配置中的 API Key 是否有效；(2) 是否具有目标操作的权限。",
                    "待处理-环境",
                    "环境故障"
                )
            elif "timeout" in error_msg.lower():
                return (
                    "环境故障",
                    "请求超时，可能是网络问题或服务端响应过慢。请检查网络连通性或重试。",
                    "待处理-环境",
                    "环境故障"
                )
            else:
                return (
                    "环境故障",
                    f"环境异常导致请求失败。请检查测试环境配置。错误信息：{error_msg[:200]}",
                    "待处理-环境",
                    "环境故障"
                )

        else:
            return (
                "未知错误",
                f"无法自动分类的错误类型，请人工排查。错误信息：{error_msg[:200]}",
                "待处理",
                "未知"
            )
