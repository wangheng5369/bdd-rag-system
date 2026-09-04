"""
LLM生成模块
支持 Claude (Anthropic) 和 OpenAI 兼容 API (MiniMax 等)
"""

import os
from pathlib import Path
from typing import List, Dict, Optional


class LLMGenerator:
    """LLM 脚本生成模块"""

    # System Prompt 文件路径
    SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompt" / "system_prompt.md"

    def __init__(
        self,
        provider: str = "openai",
        base_url: str = "",
        api_key: Optional[str] = None,
        model: str = "MiniMax-M3",
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化 LLMGenerator

        Args:
            provider: 提供商，"openai"（兼容）或 "anthropic"
            base_url: API 地址（OpenAI 兼容用，MiniMax 等填 https://...）
            api_key: API Key，默认从环境变量读取
            model: 模型名称
            max_tokens: 最大输出 token 数
            system_prompt: System Prompt 内容，None 时自动从文件加载
        """
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.client = None
        # System Prompt：优先使用传入值，其次从文件加载
        self.system_prompt = system_prompt or self._load_system_prompt()
        self._init_client()

    def _load_system_prompt(self) -> str:
        """从文件加载 System Prompt"""
        try:
            with open(self.SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"System Prompt 已加载: {self.SYSTEM_PROMPT_PATH}")
            return content
        except FileNotFoundError:
            print(f"警告: System Prompt 文件未找到 ({self.SYSTEM_PROMPT_PATH})，使用内置默认")
            return self._get_default_system_prompt()
        except Exception as e:
            print(f"警告: System Prompt 加载失败 ({e})，使用内置默认")
            return self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """获取默认 System Prompt（当文件不存在时）"""
        return """# Role
你是一名资深的 K8s / CCE 云平台自动化测试专家，精通 BDD（行为驱动开发）规范与 Python pytest-bdd 自动化测试。

# 绝对约束
1. 只能使用 Context 2 中列出的 SDK API，禁止调用任何其他函数
2. 必须优先复用 Context 1 中的已有 BDD 语句，原封不动
3. 输出必须是纯文本，禁止 markdown 格式
4. 缺失 API 时，函数体只写 pass 或 TODO

# 输出格式
1. .feature 文件（Gherkin 规范）
2. .py 文件（pytest-bdd 实现）
"""

    def _init_client(self):
        """初始化客户端"""
        if not self.api_key:
            print("警告: API Key 未设置，LLMGenerator 将返回占位内容")
            self.client = None
            return

        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                print(f"LLMGenerator 初始化成功: Anthropic/{self.model}")
            except ImportError:
                print("警告: anthropic 包未安装，运行: pip install anthropic")
                self.client = None
        else:
            # OpenAI 兼容（MiniMax 等）
            try:
                from openai import OpenAI
                extra_kwargs = {"base_url": self.base_url} if self.base_url else {}
                self.client = OpenAI(api_key=self.api_key, **extra_kwargs)
                print(f"LLMGenerator 初始化成功: OpenAI 兼容/{self.model} ({self.base_url})")
            except ImportError:
                print("警告: openai 包未安装，运行: pip install openai")
                self.client = None

    def generate(self, prompt: str, use_system: bool = True) -> str:
        """
        调用 LLM 生成内容

        Args:
            prompt: User Prompt 内容
            use_system: 是否使用 System Prompt，默认为 True

        Returns:
            LLM 生成的文本内容
        """
        if self.client is None:
            return "[LLMGenerator 未配置 API Key，无法生成内容]"

        messages = []
        # System Prompt
        if use_system and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        # User Prompt
        messages.append({"role": "user", "content": prompt})

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages
                )
                return response.content[0].text
            else:
                # OpenAI 兼容（MiniMax 等）
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return f"[LLM 调用失败: {e}]"

    def build_script_prompt(
        self,
        scenario: str,
        steps: List[dict],
        sdk_examples: List[dict] = None,
        existing_bdds: List[dict] = None,
    ) -> str:
        """组装脚本生成 prompt（兼容旧接口）"""
        from core.llm.llm import build_script_prompt_impl
        return build_script_prompt_impl(
            scenario=scenario,
            steps=steps,
            sdk_examples=sdk_examples,
            existing_bdds=existing_bdds,
        )

    def build_step_prompt(self, intent: str, sdk_info: dict) -> str:
        """组装 Step 生成 prompt（兼容旧接口）"""
        from core.llm.llm import build_step_prompt_impl
        return build_step_prompt_impl(intent=intent, sdk_info=sdk_info)


def build_script_prompt_impl(
    scenario: str,
    steps: List[dict],
    sdk_examples: List[dict] = None,
    existing_bdds: List[dict] = None,
) -> str:
    """组装脚本生成 prompt（核心实现）"""
    import re
    from pathlib import Path

    # 读取 Prompt 模板文件
    template_path = Path(__file__).parent.parent / "docs" / "prompt-template.md"
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        # 如果模板文件不存在，使用内联模板
        template = _get_inline_script_template()

    # Context 1：已有 BDD
    context1 = ""
    if existing_bdds:
        lines = []
        for bdd in existing_bdds:
            content = bdd.get("content", "")
            for block in re.findall(r"```gherkin\s*\n(.*?)```", content, re.DOTALL):
                for line in block.split("\n"):
                    line = line.strip()
                    if line and line.startswith(("Given ", "When ", "Then ", "And ")):
                        lines.append(f"- {line}")
        if lines:
            context1 = "\n".join(dict.fromkeys(lines))

    # Context 2：SDK 示例（只包含 KB1 中的 API）
    context2 = ""
    available_apis = []
    if sdk_examples:
        chunks = []
        for i, ex in enumerate(sdk_examples or []):
            code = ex.get("content", "") or ex.get("code", "")
            # 提取 API 名称
            api_match = re.search(r'接口名称[：:]\s*(\S+)', code)
            api_name = api_match.group(1) if api_match else f"API_{i+1}"
            available_apis.append(api_name)

            # 提取函数签名
            m = re.search(r"```python\s*\n(.*?)```", code, re.DOTALL)
            if m:
                chunks.append(f"【{api_name}】\n```python\n{m.group(1).strip()}\n```")
        if chunks:
            context2 = "\n\n".join(chunks)

    # Context 3：待生成 Steps
    steps_text = ""
    if steps:
        lines = [f"- {s.get('gherkin', '')}" for s in steps if s.get("gherkin")]
        steps_text = "\n".join(lines)

    ctx1_block = f"\n# Context 1: 已有的 BDD 步骤（优先复用）\n{context1}\n" if context1 else "\n# Context 1: （无）\n"
    ctx2_block = f"\n# Context 2: 可用的 SDK API（只能使用这些 API，禁止虚构）\n可用 API 列表：{', '.join(available_apis)}\n\n{context2}\n" if context2 else "\n# Context 2: （无 SDK 信息）\n"
    ctx3_block = f"\n# Context 3: 待生成的 Steps\n{steps_text}\n"

    # 替换模板中的占位符
    prompt = template.replace("{scenario}", scenario)
    prompt = prompt.replace("{ctx1_block}", ctx1_block)
    prompt = prompt.replace("{ctx2_block}", ctx2_block)
    prompt = prompt.replace("{ctx3_block}", ctx3_block)

    return prompt


def _get_inline_script_template() -> str:
    """内联备用模板（当模板文件不存在时使用）"""
    return """# Task
根据以下上下文信息，生成 BDD Feature 文件和 Python pytest-bdd 实现代码。

{ctx1_block}{ctx2_block}{ctx3_block}
---

## 输出要求

### 板块一：.feature 文件
使用标准 Gherkin 格式输出完整的 Feature 文件。

### 板块二：.py 文件
输出新写步骤的 Python pytest-bdd 实现代码。
"""


def build_step_prompt_impl(intent: str, sdk_info: dict) -> str:
    """组装 Step 生成 prompt（兼容旧接口）"""
    api_name = sdk_info.get("api_name", "unknown")
    signature = sdk_info.get("signature", "")
    params_table = sdk_info.get("params_table", "")
    example = sdk_info.get("example", "")
    description = sdk_info.get("description", "")

    return f"""你是一个 BDD Step 生成器。根据以下信息生成 When/And/Then 步骤链。

## 背景
{description}

## 意图
{intent}

## API 信息
- API 名称：{api_name}
- 函数签名：{signature}
- 参数表：
{params_table}
- 调用示例：{example}

## Step 生成规则
1. 第一步：When 构造{intent}请求，名称为"xxx"
2. 中间步骤：And 设置xxx参数
3. 最后一步：When 基于构造的请求，{intent}
4. 仅输出 Step 列表，每行一条 Gherkin 语句

## 输出格式
When 构造{intent}请求，名称为"node-01"
And 设置xxx参数
When 基于构造的请求，{intent}
"""


def create_llm(
    model: str = "MiniMax-M3",
    api_key: Optional[str] = None,
    provider: str = "openai",
    base_url: str = "",
) -> LLMGenerator:
    """
    工厂函数：创建 LLMGenerator

    默认从 config/config.yaml 读取配置
    """
    # 如果没有传入参数，从 config.yaml 读取
    if api_key is None and not base_url and provider == "openai":
        print("[create_llm] 尝试从 config.yaml 读取配置...")
        try:
            # 使用 Path 获取 config 目录的绝对路径
            from pathlib import Path
            config_dir = Path(__file__).parent.parent.parent / "config"
            print(f"[create_llm] config_dir: {config_dir}")
            print(f"[create_llm] config_dir exists: {config_dir.exists()}")
            import sys
            print(f"[create_llm] current sys.path[0]: {sys.path[0]}")
            sys.path.insert(0, str(config_dir.parent))
            print(f"[create_llm] after insert sys.path[0]: {sys.path[0]}")
            from config.config_loader import load_config, get_llm_config
            config = load_config(str(config_dir / "config.yaml"))
            print(f"[create_llm] config loaded, api_key: {config.get('api_key', '')[:20]}...")
            llm_config = get_llm_config(config)
            api_key = llm_config.get("api_key")
            base_url = llm_config.get("base_url")
            model = llm_config.get("model", model)
            provider = llm_config.get("provider", provider)
            print(f"[create_llm] api_key after get_llm_config: {api_key[:20] if api_key else None}...")
        except Exception as e:
            import traceback
            print(f"[create_llm] 从 config 读取配置失败: {e}")
            traceback.print_exc()

    return LLMGenerator(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
