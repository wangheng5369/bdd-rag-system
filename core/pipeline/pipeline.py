"""
Pipeline 核心模块
串联 Phase2+3 (Retrieval+Reranker) → StepWriter → LLM → LocalTester
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

from core.retrieval.retrieval import Retrieval
from core.step_writer import StepWriter, Step
from core.llm import LLMGenerator, create_llm
from core.testing import LocalTester, TestResult
from core.pipeline.script_verifier import ScriptVerifier, VerificationError
from self_healing.fast_path.fast_path import FastPathHealer, FastPathErrorType
from self_healing.slow_path.slow_path import SlowPathHealer


@dataclass
class PipelineResult:
    """单个用例的 Pipeline 执行结果"""
    use_case_id: str
    use_case_name: str
    query: str
    hit: bool = False                      # 是否命中 KB2
    matched_bdds: List[Dict] = field(default_factory=list)
    matched_sdks: List[Dict] = field(default_factory=list)
    generated_steps: List[Dict] = field(default_factory=list)   # Step 列表（dict格式）
    script_path: Optional[str] = None
    test_result: Optional[Dict] = None    # TestResult 的 dict 格式

    def to_dict(self) -> dict:
        return {
            "use_case_id": self.use_case_id,
            "use_case_name": self.use_case_name,
            "query": self.query,
            "hit": self.hit,
            "matched_bdds_count": len(self.matched_bdds),
            "matched_sdks_count": len(self.matched_sdks),
            "generated_steps_count": len(self.generated_steps),
            "script_path": self.script_path,
            "test_result": self.test_result
        }


class Pipeline:
    """
    BDD-RAG Pipeline

    串联: Retrieval → Reranker → (MISS? StepWriter+LLM) → Script → LocalTester
    """

    def __init__(
        self,
        kb1_dir: str,
        kb2_dir: str,
        kb3_dir: str = None,
        llm: Optional[LLMGenerator] = None,
        similarity_threshold: float = 0.4,
        reranker_threshold: float = 0.6,
        output_dir: str = "generated_scripts"
    ):
        """
        初始化 Pipeline

        Args:
            kb1_dir: KB1 (SDK) 向量数据库路径
            kb2_dir: KB2 (BDD) 向量数据库路径
            kb3_dir: KB3 (Bug) 向量数据库路径（可选）
            llm: LLMGenerator 实例，默认自动创建
            similarity_threshold: 相似度阈值
            reranker_threshold: Reranker 阈值
            output_dir: 生成脚本的输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 Retrieval（Phase 2+3）
        persist_dirs = {"kb1": kb1_dir, "kb2": kb2_dir}
        if kb3_dir:
            persist_dirs["kb3"] = kb3_dir
        self.retrieval = Retrieval(
            persist_dirs=persist_dirs,
            similarity_threshold=similarity_threshold,
            reranker_threshold=reranker_threshold
        )

        # 初始化 LLMGenerator
        self.llm = llm or create_llm()

        # 初始化 StepWriter
        self.step_writer = StepWriter(llm=self.llm)

        # 初始化 LocalTester
        self.tester = LocalTester()

    def run(self, use_case) -> PipelineResult:
        """
        对单个用例执行完整 Pipeline

        Args:
            use_case: UseCase 实例

        Returns:
            PipelineResult: 执行结果
        """
        result = PipelineResult(
            use_case_id=use_case.id,
            use_case_name=use_case.name,
            query=use_case.query
        )

        print(f"\n{'='*60}")
        print(f"用例: {use_case.id} - {use_case.name}")
        print(f"Query: {use_case.query}")
        print(f"{'='*60}")

        # ===== Phase 2+3: Retrieval + Reranker =====
        matched_bdds, did_fallback = self.retrieval.retrieve_with_fallback(
            query=use_case.query,
            primary_kb="kb2",
            fallback_kb="kb1",
            top_k=5
        )
        result.matched_bdds = matched_bdds

        print(f"[Phase2+3] 召回 {len(matched_bdds)} 条 (降级: {'是' if did_fallback else '否'})")

        # ===== 判断是否命中 =====
        if matched_bdds:
            result.hit = True
            print(f"  ✅ 命中 KB2，直接复用已有 BDD")

            # 从 matched_bdds 中提取 SDK 信息（同一 collection 里的 KB1 结果）
            matched_sdks = self._get_sdk_from_results(matched_bdds)
            result.matched_sdks = matched_sdks

            # 生成脚本（使用命中的 BDD + SDK）
            script_path = self._generate_script(use_case, result)
            result.script_path = script_path

        else:
            result.hit = False
            print(f"  ❌ MISS KB2，需要从 KB1 检索 SDK 并生成 Step")

            # ===== KB1 检索 SDK =====
            kb1_results = self.retrieval.retrieve(
                query=use_case.query,
                kb_name="kb1",
                top_k=3
            )
            result.matched_sdks = kb1_results
            print(f"  [KB1] 召回 {len(kb1_results)} 条 SDK")

            if not kb1_results:
                print(f"  ⚠️ KB1 也无结果，跳过脚本生成")
                return result

            # ===== StepWriter 生成 Step =====
            generated_steps = self.step_writer.write_steps(
                intent=use_case.name,
                kb1_results=kb1_results
            )
            if generated_steps:
                result.generated_steps = [asdict(s) for s in generated_steps]
                print(f"  [StepWriter] 生成 {len(generated_steps)} 条 Step")
                for s in generated_steps:
                    print(f"    - {s.gherkin}")
            else:
                print(f"  [StepWriter] 生成失败")
                return result

            # ===== 生成脚本 =====
            script_path = self._generate_script(use_case, result)
            result.script_path = script_path

        # ===== 运行测试 =====
        if result.script_path and Path(result.script_path).exists():
            test_result = self.tester.run(
                result.script_path,
                save_report_to=result.script_path.replace(".py", "_result.json")
            )
            result.test_result = test_result.to_dict()
            status = "✅" if test_result.passed else "❌"
            print(f"\n  {status} 测试 {test_result.passed_count}/{test_result.total} 通过")
            for s in test_result.step_results:
                if s.status != "passed":
                    print(f"    - {s.step_name}: {s.status} | {s.error[:80] if s.error else ''}")

            # ===== Fast Path 自愈 =====
            if test_result.failed_count > 0:
                healer = FastPathHealer(llm=self.llm)
                healed_steps = []
                for step_result in test_result.step_results:
                    if step_result.status in ('failed', 'error'):
                        log_info = {
                            'type': step_result.status,
                            'message': step_result.error or '',
                            'step_name': step_result.step_name
                        }
                        fixed, msg = healer.diagnose_and_heal(
                            log_info,
                            result.script_path
                        )
                        healed_steps.append({
                            'step': step_result.step_name,
                            'fixed': fixed,
                            'message': msg
                        })
                        print(f"    🔧 自愈: {step_result.step_name} → {msg}")

                # 如果有任何步骤被修复，重新运行测试
                if any(h['fixed'] for h in healed_steps):
                    print(f"\n  🔄 重新运行测试验证修复...")
                    test_result = self.tester.run(
                        result.script_path,
                        save_report_to=result.script_path.replace(".py", "_result.json")
                    )
                    result.test_result = test_result.to_dict()
                    status = "✅" if test_result.passed else "❌"
                    print(f"\n  {status} 测试 {test_result.passed_count}/{test_result.total} 通过")
                    for s in test_result.step_results:
                        if s.status != "passed":
                            print(f"    - {s.step_name}: {s.status} | {s.error[:80] if s.error else ''}")

                # ===== Slow Path 自愈（Fast Path 无法修复时）=====
                remaining_failures = [s for s in test_result.step_results if s.status in ('failed', 'error')]
                if remaining_failures and self.retrieval.vectorstores.get("kb3"):
                    print(f"\n  🔍 触发 Slow Path 分析...")
                    slow_healer = SlowPathHealer(
                        llm=self.llm,
                        kb3_dir=self.retrieval.persist_dirs.get("kb3")
                    )
                    for step_result in remaining_failures:
                        log_info = {
                            'type': step_result.status,
                            'message': step_result.error or '',
                            'step_name': step_result.step_name
                        }
                        _, msg, ticket_path = slow_healer.diagnose_and_heal(log_info)
                        print(f"    📋 Slow Path: {step_result.step_name} → {msg}")
        else:
            print(f"\n  脚本未生成，跳过测试")

        return result

    def run_batch(self, use_cases: List, verbose: bool = True) -> List[PipelineResult]:
        """
        批量执行 Pipeline

        Args:
            use_cases: UseCase 列表
            verbose: 是否打印详情

        Returns:
            List[PipelineResult]: 结果列表
        """
        results = []
        for uc in use_cases:
            r = self.run(uc)
            results.append(r)
        return results

    def _get_sdk_from_results(self, results: List[Dict]) -> List[Dict]:
        """从召回结果中提取 SDK 信息"""
        # 当前逻辑：直接返回结果，让 LLM 自行判断哪些是 SDK
        return results

    def _generate_script(self, use_case, result: PipelineResult) -> Optional[str]:
        """调用 LLM 生成 BDD 脚本"""
        # HIT 路径时：额外从 KB1 查询 SDK 签名
        if result.hit and result.matched_bdds:
            kb1_results = self.retrieval.retrieve(
                query=use_case.query,
                kb_name="kb1",
                top_k=5,
                threshold=0.3  # KB1 查询用较低阈值
            )
            sdk_examples = [{"content": s.get("content", "")} for s in kb1_results]
            matched_sdks_for_verify = kb1_results
            print(f"\n  [DEBUG] KB1 召回 {len(kb1_results)} 条")
            for i, r in enumerate(kb1_results[:3]):
                content = r.get('content', '')
                print(f"    KB1-{i+1} (len={len(content)}): {content[:300]}...")
                # 统计函数签名数量
                import re
                sigs = re.findall(r'def\s+\w+\s*\(', content)
                print(f"      -> Found {len(sigs)} function definitions: {[s for s in sigs]}")
        else:
            sdk_examples = [{"content": s.get("content", "")} for s in result.matched_sdks]
            matched_sdks_for_verify = result.matched_sdks

        # 组装 prompt
        prompt = self.llm.build_script_prompt(
            scenario=use_case.name,
            steps=result.generated_steps if not result.hit else [],
            sdk_examples=sdk_examples,
            existing_bdds=result.matched_bdds if result.hit else None,
        )

        # 从 matched_sdks 提取 SDK 签名字典（提前，以便重试时复用）
        sdk_api = self._extract_sdk_api(matched_sdks_for_verify)
        print(f"\n  [DEBUG] KB1 SDK 提取结果: {list(sdk_api.keys())}")
        for api_name, api_info in sdk_api.items():
            params = api_info.get('params', [])
            required = api_info.get('required', [])
            print(f"    {api_name}:")
            print(f"      params ({len(params)}): {params}")
            print(f"      required ({len(required)}): {required}")

        # 调用 LLM（可能多次：校验失败时重试）
        script_content = self._llm_generate_with_retry(
            prompt=prompt,
            sdk_api=sdk_api,
            use_case=use_case,
            max_retries=3
        )

        if script_content is None:
            print(f"  ⚠️ LLM 生成失败，跳过")
            return None

        # 保存脚本
        script_path = self.output_dir / f"{use_case.id}.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        print(f"  📄 脚本已生成: {script_path}")
        return str(script_path)

    def _extract_sdk_api(self, matched_sdks: List[Dict]) -> Dict[str, Dict]:
        """从 matched_sdks 提取 SDK 签名字典，供 ScriptVerifier 使用
        
        处理两种 chunk：
        1. API 级别 chunk（level=3）：只包含单个 API，直接提取
        2. Module 级别 chunk（level=2）：包含多个 API，需要拆分处理
        """
        import re
        api = {}
        
        for item in matched_sdks:
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            level = metadata.get("level", 3)
            
            # 找出所有 API 名称和它们的参数
            apis_in_chunk = self._extract_apis_from_chunk(content)
            
            for api_name, (params, required) in apis_in_chunk.items():
                if api_name not in api:
                    api[api_name] = {"params": params, "required": required}
        
        return api

    def _extract_apis_from_chunk(self, content: str) -> Dict[str, tuple]:
        """从 chunk 内容中提取所有 API 的参数信息"""
        import re
        result = {}
        
        # 匹配 "接口名称" 行及其后续内容，直到下一个 "接口名称" 或文档结束
        # 模式: "接口名称：XXX" 后跟内容
        api_pattern = r'接口名称[：:]\s*(\S+)'
        
        for m_api in re.finditer(api_pattern, content):
            api_name = m_api.group(1).strip()
            # 清理 api_name（比如 "4. 接口名称：create_node" -> "create_node"）
            api_name = re.sub(r'^\d+\.\s*', '', api_name)
            
            # 确定这个 API 的范围（到下一个接口名称或文档结束）
            start_pos = m_api.start()
            next_api_m = re.search(api_pattern, content[start_pos + len(m_api.group(0)):])
            if next_api_m:
                end_pos = start_pos + len(m_api.group(0)) + next_api_m.start()
            else:
                end_pos = len(content)
            
            api_content = content[start_pos:end_pos]
            
            # 在这个 API 的范围内提取参数
            params = []
            required = []
            
            # 方法1：从参数表格提取
            for pm in re.finditer(r'`?(\w+)`?\s*\|\s*`(?:str|int|bool|list|dict|float)`', api_content):
                param_name = pm.group(1)
                params.append(param_name)
                required.append(param_name)
            
            # 方法2：如果没有表格，用函数签名
            if not params:
                m_sig = re.search(r'def\s+' + re.escape(api_name) + r'\s*\(([^)]*)\)', api_content)
                if m_sig:
                    for arg in m_sig.group(1).split(','):
                        arg = arg.strip()
                        if arg:
                            name = arg.split(':')[0].strip().split('=')[0].strip()
                            if name and name not in ('self', 'cls'):
                                params.append(name)
                                if '=' not in arg:
                                    required.append(name)
            
            if params:
                result[api_name] = (list(dict.fromkeys(params)), list(dict.fromkeys(required)))
        
        return result

    def _strip_markdown(self, raw_output: str) -> str:
        """从 LLM 原始输出中提取纯 Python 代码

        移除：
        - 所有 markdown 代码块标记 (```python ... ```, ``` ... ```)
        - 所有思考过程标签 (<think> ... 、</think>)
        - 所有行内 markdown 标记
        """
        import re

        if not raw_output:
            return raw_output

        # 移除  思考过程标签
        output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
        output = re.sub(r'<think>.*', '', output)

        # 移除 markdown 代码块
        # ```python ... ``` 或 ``` ... ```
        output = re.sub(r'```python\s*(.*?)```', r'\1', output, flags=re.DOTALL)
        output = re.sub(r'```\s*(.*?)```', r'\1', output, flags=re.DOTALL)

        # 移除 markdown 标题 (##, ### 等)
        output = re.sub(r'^#{1,6}\s+', '', output, flags=re.MULTILINE)

        # 移除 markdown 列表 (- , *, 等)
        output = re.sub(r'^[-*+]\s+', '', output, flags=re.MULTILINE)

        # 移除 markdown 强调 (**, __)
        output = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', output)
        output = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', output)

        # 移除 markdown 链接
        output = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', output)

        # 移除 "Output:", "Result:", "Python:" 等前缀
        output = re.sub(r'^(Output|Result|Python|代码|输出|结果)[:：]\s*', '', output, flags=re.MULTILINE)

        # 移除空行和前后空白
        lines = output.split('\n')
        code_lines = []
        in_code_block = False
        for line in lines:
            stripped = line.strip()
            # 跳过明显的解释性文本行
            if not in_code_block:
                if stripped.startswith('from ') or stripped.startswith('@') or stripped.startswith('def ') or stripped.startswith('import '):
                    in_code_block = True
                elif stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                    # 可能还在解释区域，跳过短行
                    if len(stripped) < 200 and not any(c in stripped for c in ['@', 'def ', 'class ', 'from ', 'import ']):
                        continue
            code_lines.append(line)

        result = '\n'.join(code_lines).strip()

        # 如果结果仍然不是有效的 Python（没有 import 或 @ 装饰器），尝试更激进的方式
        if not any(kw in result for kw in ['from ', 'import ', '@given', '@when', '@then', 'def ']):
            # 提取所有看起来像 Python 代码的行
            code_lines = [l for l in lines if l.strip() and
                         (l.strip().startswith(('from ', 'import ', '@', 'def ', '#', '"""', "'''", 'if ', 'for ', 'while ', 'return ', 'pass', 'context')))]
            result = '\n'.join(code_lines)

        return result

    def _fix_python_format(self, code: str) -> str:
        """修复常见的 Python 格式问题"""
        import re

        if not code:
            return code

        # 修复模块名：pytestbdd -> pytest_bdd
        code = re.sub(r'from\s+pytestbdd\s+import', 'from pytest_bdd import', code)
        code = re.sub(r'import\s+pytestbdd', 'import pytest_bdd', code)

        # 修复 SDK 模块名：ccesdk -> cce_sdk, cce-sdk -> cce_sdk
        code = re.sub(r'from\s+ccesdk', 'from cce_sdk', code)
        code = re.sub(r'import\s+ccesdk', 'import cce_sdk', code)
        code = re.sub(r'from\s+cce-sdk', 'from cce_sdk', code)
        code = re.sub(r'import\s+cce-sdk', 'import cce_sdk', code)

        # 修复函数名：修复无下划线的函数名 (stepbuildnodereq -> step_build_node_req)
        # 只处理连续字母开头的函数名，添加缺失的下划线
        def add_underscores(name):
            # 在连续大写字母前添加下划线，然后在所有大写字母后添加下划线
            # stepBuildNodeReq -> step_Build_Node_Req -> step_build_node_req
            if '_' in name or name.islower():
                return name
            result = name
            # 处理 CamelCase -> snake_case
            result = re.sub(r'([a-z])([A-Z])', r'\1_\2', result)
            result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', result)
            return result.lower()

        # 修复函数定义名
        code = re.sub(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                      lambda m: f'def {add_underscores(m.group(1))}(', code)

        # 修复字典 key 中的下划线
        # context["nodereq"] -> context["node_req"]
        # clusterid -> cluster_id, datavolumesize -> data_volume_size
        KNOWN_KEYS = {
            'clusterid': 'cluster_id',
            'nodeid': 'node_id',
            'nodereq': 'node_req',
            'nodepoolid': 'nodepool_id',
            'datavolumesize': 'data_volume_size',
            'rootvolumesize': 'root_volume_size',
            'ostype': 'os_type',
            'volumetype': 'volume_type',
            'subnetid': 'subnet_id',
            'vpcid': 'vpc_id',
            'deleteefs': 'delete_efs',
            'deleteeni': 'delete_eni',
            'deletenet': 'delete_net',
            'initialnodes': 'initial_nodes',
            'maxnodes': 'max_nodes',
            'minnodes': 'min_nodes',
            'desirednodes': 'desired_nodes',
            'nodepoolid': 'nodepool_id',
            'clusterid': 'cluster_id',
        }

        def fix_dict_key(name):
            # 先检查已知的问题字符串
            if name.lower() in KNOWN_KEYS:
                return KNOWN_KEYS[name.lower()]
            # 如果已经有下划线，直接返回
            if '_' in name:
                return name
            # 否则尝试转换驼峰
            result = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
            return result.lower()

        code = re.sub(r'\["([a-zA-Z_][a-zA-Z0-9_]*)"\]',
                      lambda m: f'["{fix_dict_key(m.group(1))}"]', code)

        # 修复 API 调用中的参数名 (不带下划线的)
        # clusterid -> cluster_id
        code = re.sub(r'\b([a-z]+)([A-Z][a-z])=', r'\1_\2=', code)

        # 修复纯小写的 API 调用名 (createnode -> create_node)
        # 匹配常见的 API 动作后缀
        code = re.sub(r'\b(create|delete|get|list|update|reset|add|put|patch)(node|cluster|pool|volume|network)\b',
                      lambda m: f'{m.group(1)}_{m.group(2)}', code, flags=re.IGNORECASE)

        # 修复混合情况：stepcallcreatenode -> step_call_create_node
        def fix_mixed_case(name):
            if '_' in name:
                return name
            # 查找所有 API 动作
            for action in ['create', 'delete', 'get', 'list', 'update', 'reset', 'add', 'put', 'patch']:
                if action in name.lower():
                    idx = name.lower().find(action)
                    prefix = name[:idx]
                    rest = name[idx:]
                    # 在 rest 中也插入下划线
                    fixed_rest = rest
                    for a in ['Node', 'Cluster', 'Pool', 'Volume', 'Network']:
                        if a.lower() in fixed_rest.lower():
                            fixed_rest = re.sub(r'([A-Z])', r'_\1', fixed_rest)
                            break
                    return (prefix + fixed_rest).lower()
            return name

        # 修复函数名中混合大小写的情况
        code = re.sub(r'def\s+([a-zA-Z][a-zA-Z0-9]*)\s*\(',
                      lambda m: f'def {fix_mixed_case(m.group(1))}(', code)

        # 移除中文警告行
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            stripped = line.strip()
            # 跳过只包含中文且不包含 Python 关键字的行
            if re.search(r'^[一-鿿]+$', stripped):
                continue
            # 也跳过包含中文警告的行（但保留包含代码的行）
            if re.match(r'^[^a-zA-Z]*警告[:：]', stripped) or \
               (re.match(r'^[^a-zA-Z]*#\s*警告', stripped)):
                continue
            fixed_lines.append(line)
        code = '\n'.join(fixed_lines)

        return code

    def _llm_generate_with_retry(
        self,
        prompt: str,
        sdk_api: Dict,
        use_case,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        调用 LLM 生成脚本，校验失败时将错误反馈给 LLM 重试
        """
        current_prompt = prompt

        for attempt in range(1, max_retries + 1):
            raw_output = self.llm.generate(current_prompt)

            # DEBUG: 打印原始输出
            print(f"\n  [DEBUG] LLM 原始输出 (尝试 {attempt}):")
            print("-" * 40)
            print(raw_output[:2000] if raw_output else "(空)")
            print("-" * 40)

            # 后处理：提取纯 Python 代码
            script_content = self._strip_markdown(raw_output)

            # 修复常见的 Python 格式问题
            script_content = self._fix_python_format(script_content)

            if script_content.startswith("[LLMGenerator"):
                return None

            if not sdk_api:
                return script_content

            verifier = ScriptVerifier(sdk_api)
            vr = verifier.verify(script_content)

            if vr.passed:
                return script_content

            error_lines = [
                f"- [{iss.severity}] {iss.check}: {iss.message} (位置: {iss.location})"
                for iss in vr.issues
            ]

            if attempt == max_retries:
                print(f"  ⚠️ 脚本校验失败，已重试 {attempt} 次")
                for line in error_lines[:5]:
                    print(f"     {line[:120]}")
                # 保存失败版本以便调试
                failed_path = self.output_dir / f"{use_case.id}_failed.py"
                with open(failed_path, "w", encoding="utf-8") as f:
                    f.write(script_content)
                print(f"  📄 失败脚本已保存: {failed_path}")
                raise VerificationError(vr.issues)

            print(f"  ⚠️ 校验失败（尝试 {attempt}/{max_retries}），反馈给 LLM 修复...")

            current_prompt = self._build_fix_prompt(
                original_prompt=prompt,
                script=script_content,
                issues=vr.issues
            )

        return None

    def _build_fix_prompt(
        self,
        original_prompt: str,
        script: str,
        issues: List
    ) -> str:
        issue_text = "\n".join([
            f"- [{iss.severity}] {iss.check}: {iss.message} (位置: {iss.location})"
            for iss in issues
        ])
        return (
            original_prompt
            + "\n\n---\n\n## 修正要求\n\n"
            + "上一轮生成的脚本存在以下问题，请修正后重新输出**完整的修正后脚本**（不要只输出修改部分）：\n\n"
            + issue_text
            + "\n\n请重新生成完整的 Python 脚本，确保：\n"
            + "1. 修复以上所有 error 级别问题\n"
            + "2. 保持原有的 BDD Step 结构（Given/When/Then/And）\n"
            + "3. 修复后的脚本可直接通过语法和 API 签名校验\n"
        )




def run_pipeline(
    use_cases: List,
    kb1_dir: str,
    kb2_dir: str,
    llm: Optional[LLMGenerator] = None,
    output_dir: str = "generated_scripts"
) -> List[PipelineResult]:
    """
    便捷函数：运行 Pipeline

    Args:
        use_cases: UseCase 列表
        kb1_dir: KB1 路径
        kb2_dir: KB2 路径
        llm: LLMGenerator
        output_dir: 输出目录

    Returns:
        List[PipelineResult]: 结果列表
    """
    pipeline = Pipeline(
        kb1_dir=kb1_dir,
        kb2_dir=kb2_dir,
        llm=llm,
        output_dir=output_dir
    )
    return pipeline.run_batch(use_cases)
