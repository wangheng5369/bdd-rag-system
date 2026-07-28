"""
Script Verifier — LLM 生成脚本后校验
1. Python 语法校验（AST）
2. API 签名校验（静态）
3. BDD Gherkin 格式校验
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional


@dataclass
class VerificationIssue:
    """单个校验问题"""
    severity: str      # "error" | "warning"
    check: str       # "syntax" | "api" | "bdd"
    message: str
    location: str      # "line 42" 或 "delete_node()"


@dataclass
class VerificationResult:
    """校验结果"""
    passed: bool
    issues: List[VerificationIssue] = field(default_factory=list)

    def add(self, issue: VerificationIssue):
        self.issues.append(issue)
        self.passed = False


class VerificationError(Exception):
    """脚本校验失败时抛出，阻断后续写入和执行"""
    def __init__(self, issues: List[VerificationIssue]):
        self.issues = issues
        super().__init__(str(self))

    def __str__(self):
        lines = [f"脚本校验失败，共 {len(self.issues)} 个问题:"]
        for iss in self.issues:
            lines.append(f"  [{iss.severity}] {iss.check}: {iss.message} (位置: {iss.location})")
        return "\n".join(lines)


class ScriptVerifier:
    """
    校验 LLM 生成的 BDD 脚本
    """

    def __init__(self, sdk_api: Dict[str, Dict]):
        """
        Args:
            sdk_api: SDK API 签名字典
                     格式: {
                       "delete_node": {
                         "params": ["cluster_id", "node_id", "drain_first"],
                         "required": ["cluster_id", "node_id"]
                       }
                     }
        """
        self.sdk_api = sdk_api

    def verify(self, script_content: str) -> VerificationResult:
        """
        执行所有校验

        Args:
            script_content: LLM 生成的 Python 脚本内容

        Returns:
            VerificationResult: 校验结果
        """
        result = VerificationResult(passed=True)

        # 1. Python 语法校验
        result.issues.extend(self._check_python_syntax(script_content))

        # 2. API 签名校验
        result.issues.extend(self._check_api_signatures(script_content))

        # 3. BDD Gherkin 格式校验
        result.issues.extend(self._check_bdd_format(script_content))

        if result.issues:
            result.passed = False
        return result

    def _check_python_syntax(self, script: str) -> List[VerificationIssue]:
        """Python AST 语法校验"""
        issues = []
        try:
            ast.parse(script)
        except SyntaxError as e:
            issues.append(VerificationIssue(
                severity="error",
                check="syntax",
                message=f"Python 语法错误: {e.msg}",
                location=f"line {e.lineno or '?'}"
            ))
        return issues

    def _check_api_signatures(self, script: str) -> List[VerificationIssue]:
        """对比 SDK API 签名：检查函数调用是否存在、参数是否匹配"""
        issues = []

        # 提取所有 cce_sdk.* 调用
        # 匹配形如: cce_sdk.node.delete_node(...) 或 delete_node(...)
        # 以及 from cce_sdk.node import delete_node 后直接调用 delete_node(...)
        calls = self._extract_sdk_calls(script)

        for call in calls:
            func_name = call["func"]
            args = call["args"]
            kwargs = call["kwargs"]
            lineno = call.get("lineno", "?")

            # 跳过已知的 BDD / pytest / Python 内置名
            if func_name in {
                "when", "then", "given", "And", "context",
                "parse", "parsers", "step", "scenario", "feature",
                "assert", "print", "len", "str", "int", "float",
                "list", "dict", "set", "tuple", "bool", "type",
                # 常见的 dict/object 方法
                "get", "items", "keys", "values", "append", "extend",
                "pop", "update", "setdefault", "copy", "clear",
                # 常见的 response/result 方法
                "status", "data", "result", "error", "errors",
                "properties", "parameters", "exist", "cluster",
                # 常见的测试相关
                "sleep", "wait", "timeout", "retry",
                # LLM 生成的内容中的常见词
                "signature", "different", "required", "passed",
                "Assertions", "parameters_list", "missing", "note",
            }:
                continue

            # 检查 API 是否存在
            if func_name not in self.sdk_api:
                issues.append(VerificationIssue(
                    severity="error",
                    check="api",
                    message=f"API '{func_name}()' 不存在，请检查是否虚构了不存在的接口",
                    location=f"line {lineno}"
                ))
                continue

            # 获取签名
            sig = self.sdk_api[func_name]
            expected_params = sig.get("params", [])
            required_params = sig.get("required", expected_params)

            # 检查参数数量
            provided = set(args) | set(kwargs.keys())

            # 缺少必需参数
            missing = set(required_params) - provided
            if missing:
                issues.append(VerificationIssue(
                    severity="error",
                    check="api",
                    message=f"{func_name}() 缺少必需参数: {sorted(missing)}，签名要求: {required_params}",
                    location=f"line {lineno}"
                ))

        return issues

    def _extract_sdk_calls(self, script: str) -> List[Dict]:
        """从脚本中提取所有 SDK 函数调用"""
        calls = []
        lines = script.split("\n")

        # 提取所有形如 xxx_call(...) 的调用（不含模块前缀）
        for i, line in enumerate(lines, 1):
            # 跳过装饰器行和 def 定义行
            stripped = line.strip()
            if stripped.startswith("@") or stripped.startswith("def "):
                continue

            # 跳过注释
            code_part = line.split("#")[0]

            # 移除字符串字面量和格式占位符
            # 只保留实际的函数调用模式
            cleaned = ""
            in_string = False
            string_char = None
            brace_depth = 0  # 跟踪 {} 深度（格式占位符）
            j = 0
            while j < len(code_part):
                ch = code_part[j]
                if not in_string:
                    if ch in ('"', "'", 'f"', "f'", 'r"', "r'", 'b"', "b'"):
                        in_string = True
                        string_char = ch
                        cleaned += "''"
                        j += 1
                        continue
                    elif ch == '{':
                        brace_depth = 1
                        cleaned += '{{'  # 暂时替换
                        j += 1
                        continue
                    elif ch == '}':
                        cleaned += '}}'
                        j += 1
                        continue
                    else:
                        cleaned += ch
                else:
                    if ch == string_char and (j == 0 or code_part[j-1] != '\\'):
                        in_string = False
                        string_char = None
                        cleaned += "''"
                j += 1

            # 移除格式占位符 {}
            cleaned = re.sub(r'\{[^}]*\}', '', cleaned)

            # 现在在清理后的代码上匹配函数调用
            for m in re.finditer(r"(\w+)\s*\(([^)]*)\)", cleaned):
                func = m.group(1)
                args_str = m.group(2)
                args = self._parse_args(args_str)
                calls.append({
                    "func": func,
                    "args": args,
                    "kwargs": {},
                    "lineno": i
                })

        return calls

    def _parse_args(self, args_str: str) -> List[str]:
        """解析函数调用参数列表"""
        if not args_str.strip():
            return []
        args = []
        # 简单解析：用逗号分隔（不考虑嵌套括号）
        depth = 0
        current = ""
        for ch in args_str:
            if ch in "([{":
                depth += 1
                current += ch
            elif ch in ")]}":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current.strip())
        # 清理参数值（只取变量名）
        cleaned = []
        for a in args:
            a = a.strip()
            # 取第一个词作为参数名
            m = re.match(r"([a-zA-Z_]\w*)", a)
            if m:
                cleaned.append(m.group(1))
        return cleaned

    def _check_bdd_format(self, script: str) -> List[VerificationIssue]:
        """BDD Gherkin 格式校验"""
        issues = []

        # 提取所有装饰器行
        decorator_lines = []
        for i, line in enumerate(script.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("@when") or stripped.startswith("@then") or \
               stripped.startswith("@given") or stripped.startswith("@And"):
                decorator_lines.append((i, stripped))

        if not decorator_lines:
            issues.append(VerificationIssue(
                severity="warning",
                check="bdd",
                message="未找到任何 @when/@then/@given 装饰器，可能缺少 BDD Step 定义",
                location="file"
            ))
            return issues

        # 检查第一个 Step 是否是 And（孤立 And 错误）
        first_line = decorator_lines[0][1]
        if first_line.startswith("@And"):
            issues.append(VerificationIssue(
                severity="error",
                check="bdd",
                message="第一个 Step 不能使用 @And，必须以 @Given/@When/@Then 开头",
                location=f"line {decorator_lines[0][0]}"
            ))

        # 检查 Given/When/Then/And 关键字是否在合法位置
        for lineno, line in decorator_lines:
            # 检查是否有未闭合的括号
            open_count = line.count("(") - line.count(")")
            if open_count > 0:
                issues.append(VerificationIssue(
                    severity="warning",
                    check="bdd",
                    message=f"装饰器行存在未闭合的括号: {line[:50]}",
                    location=f"line {lineno}"
                ))

        return issues


def verify_script(script_content: str, sdk_api: Dict[str, Dict]) -> VerificationResult:
    """
    便捷函数：对脚本执行校验

    Args:
        script_content: LLM 生成的 Python 脚本
        sdk_api: SDK API 签名字典

    Returns:
        VerificationResult: 校验结果
    """
    verifier = ScriptVerifier(sdk_api)
    return verifier.verify(script_content)
