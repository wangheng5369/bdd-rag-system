"""
LocalTester 模块
在本地运行生成的 pytest-bdd 脚本，验证 step 实现是否正确
"""

import json
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class StepResult:
    """单个 Step 的执行结果"""
    step_name: str                    # e.g. "step_build_node_req"
    status: str                       # "passed" / "failed" / "skipped" / "error"
    error: Optional[str] = None       # 失败时的错误信息
    duration: Optional[float] = None  # 执行耗时（秒）


@dataclass
class TestResult:
    """完整测试场景的执行结果"""
    scenario: str                     # 场景名
    passed: bool                      # 是否全部通过
    total: int                        # 总 step 数
    passed_count: int                 # 通过数
    failed_count: int                # 失败数
    skipped_count: int = 0           # 跳过数
    step_results: List[StepResult] = field(default_factory=list)
    stdout: str = ""                 # 原始 stdout
    stderr: str = ""                 # 原始 stderr
    duration: Optional[float] = None # 总耗时（秒）

    @property
    def all_passed(self) -> bool:
        return self.passed_count == self.total and self.total > 0

    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "scenario": self.scenario,
            "passed": self.passed,
            "total": self.total,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "duration": self.duration,
            "steps": [
                {
                    "step_name": s.step_name,
                    "status": s.status,
                    "error": s.error,
                    "duration": s.duration
                }
                for s in self.step_results
            ],
            "stderr": self.stderr
        }

    def save_report(self, output_path: str) -> None:
        """
        将测试报告保存为 JSON 文件

        Args:
            output_path: 输出文件路径
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class LocalTester:
    """
    本地运行 pytest-bdd 脚本

    使用 pytest 的 JUnit XML 输出解析测试结果
    """

    def __init__(
        self,
        pytest_args: str = "-v --tb=short",
        default_timeout: int = 120
    ):
        """
        初始化 LocalTester

        Args:
            pytest_args: pytest 默认参数
            default_timeout: 默认超时时间（秒）
        """
        self.pytest_args = pytest_args
        self.default_timeout = default_timeout

    def run(
        self,
        script_path: str,
        pytest_args: Optional[str] = None,
        timeout: Optional[int] = None,
        save_report_to: Optional[str] = None
    ) -> TestResult:
        """
        运行测试脚本

        Args:
            script_path: 测试脚本路径
            pytest_args: 覆盖默认 pytest 参数
            timeout: 覆盖默认超时时间

        Returns:
            TestResult: 测试结果
        """
        script_path = Path(script_path)
        if not script_path.exists():
            return TestResult(
                scenario=script_path.stem,
                passed=False,
                total=0,
                passed_count=0,
                failed_count=0,
                stderr=f"脚本不存在: {script_path}"
            )

        # 构建 pytest 命令
        args = pytest_args or self.pytest_args
        # 生成临时 JUnit XML 报告路径
        junit_report = script_path.parent / f".pytest_{script_path.stem}_report.xml"
        cmd = [
            "pytest",
            str(script_path),
            *args.split(),
            f"--junit-xml={junit_report}",
            "-q"  # 简洁输出
        ]

        effective_timeout = timeout or self.default_timeout

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout
            )
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout or ""
            stderr = e.stderr or f"测试超时（{effective_timeout}秒）"
            returncode = -1
        except Exception as e:
            stdout = ""
            stderr = f"执行失败: {e}"
            returncode = -1

        # 解析 JUnit XML 报告
        test_result = self._parse_junit(junit_report, script_path.stem)
        test_result.stdout = stdout
        test_result.stderr = stderr

        # 清理临时报告文件
        if junit_report.exists():
            junit_report.unlink()

        # 可选：保存 JSON 报告
        if save_report_to:
            test_result.save_report(save_report_to)

        return test_result

    def _parse_junit(self, xml_path: Path, fallback_scenario: str) -> TestResult:
        """解析 JUnit XML 报告"""
        if not xml_path.exists():
            return TestResult(
                scenario=fallback_scenario,
                passed=False,
                total=0,
                passed_count=0,
                failed_count=0,
                stderr="JUnit XML 报告未生成"
            )

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            return TestResult(
                scenario=fallback_scenario,
                passed=False,
                total=0,
                passed_count=0,
                failed_count=0,
                stderr=f"XML 解析失败: {e}"
            )

        # 获取场景名（testcase 的 classname）
        testcases = root.findall(".//testcase")
        scenario = fallback_scenario
        if testcases and testcases[0].get("classname"):
            scenario = testcases[0].get("classname").split(".")[-1]

        step_results: List[StepResult] = []
        passed_count = 0
        failed_count = 0
        skipped_count = 0

        for tc in testcases:
            name = tc.get("name", "unknown")
            time = float(tc.get("time", 0))

            # 检查 failure 或 error
            failure = tc.find("failure")
            error = tc.find("error")
            skipped = tc.find("skipped")

            if failure is not None:
                status = "failed"
                error_msg = failure.get("message") or str(failure.text or "").strip()
                failed_count += 1
            elif error is not None:
                status = "error"
                error_msg = error.get("message") or str(error.text or "").strip()
                failed_count += 1
            elif skipped is not None:
                status = "skipped"
                error_msg = skipped.get("message") or ""
                skipped_count += 1
            else:
                status = "passed"
                error_msg = None
                passed_count += 1

            step_results.append(StepResult(
                step_name=name,
                status=status,
                error=error_msg[:500] if error_msg else None,
                duration=time
            ))

        total = len(testcases)
        passed = failed_count == 0 and skipped_count == 0

        return TestResult(
            scenario=scenario,
            passed=passed,
            total=total,
            passed_count=passed_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            step_results=step_results
        )

    def run_and_wait(
        self,
        script_path: str,
        pytest_args: Optional[str] = None,
        timeout: Optional[int] = None,
        save_report_to: Optional[str] = None
    ) -> TestResult:
        """
        带超时的等待运行

        Args:
            script_path: 测试脚本路径
            pytest_args: 覆盖默认 pytest 参数
            timeout: 最大总超时时间

        Returns:
            TestResult: 测试结果
        """
        return self.run(script_path, pytest_args, timeout, save_report_to)
