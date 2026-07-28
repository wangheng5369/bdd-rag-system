# BDD Pipeline Skill

## 用途

将自然语言用例文本转换为可执行的 pytest-bdd 自动化脚本。

## 入口

```bash
# 设置 API Key（可选，未设置则跳过 LLM 生成步骤）
export ANTHROPIC_API_KEY="your-key"

# 运行 Pipeline
python main.py
python main.py --use-case-file docs/用例示例.md
```

## 完整流程

```
文本用例 (docs/用例示例.md)
  │
  ▼
parse_use_cases()  ──→ UseCase{id, name, query, steps, expected}
  │
  ▼ 对每个用例
  │
  ├── Phase 2: retrieval.retrieve()  ──→ KB2 向量检索
  │              ↓
  ├── Phase 2.5: filter_by_action_conflict()  ──→ 硬规则熔断
  │              ↓
  ├── Phase 3: reranker.rerank()  ──→ Cross-Attention 重排
  │              ↓
  │          matched_bdds (List[Dict])
  │
  ├── if matched_bdds 非空:
  │     ✅ 命中 → matched_bdds + matched_sdks → LLM 生成脚本
  │
  ├── if matched_bdds 为空:
  │     ❌ MISS → 查 KB1 (SDK) → StepWriter.write_steps()
  │                → LLM 生成 BDD Step 链
  │                → LLM 生成脚本
  │
  ▼
LLMGenerator.generate_script()  ──→ pytest-bdd Python 文件
  │
  ▼
LocalTester.run()  ──→ TestResult (passed/failed/steps)
```

## 模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| **用例解析** | `core/pipeline/use_case_parser.py` | 解析用例文档为 `UseCase` 结构 |
| **Pipeline** | `core/pipeline/pipeline.py` | 串联 Phase2-5，复用已有模块 |
| **Retrieval** | `core/retrieval/retrieval.py` | Phase2+3，向量检索 + 重排 |
| **StepWriter** | `core/step_writer/step_writer.py` | MISS 时从 KB1 生成 BDD Step |
| **LLMGenerator** | `core/llm/llm.py` | 调用 Claude 生成 Step 和脚本 |
| **LocalTester** | `core/testing/testing.py` | 运行 pytest-bdd 并解析结果 |

## 数据结构

### UseCase（用例）
```python
@dataclass
class UseCase:
    id: str           # "Tc_Func_Node_001"
    name: str        # "创建100G数据盘的节点"
    query: str       # 用于向量检索的 query
    steps: List[str] # ["创建集群", "创建节点..."]
    expected: str     # 预期结果
```

### PipelineResult（执行结果）
```python
@dataclass
class PipelineResult:
    use_case_id: str
    hit: bool                      # 是否命中 KB2
    matched_bdds: List[Dict]       # 命中的 BDD Step
    matched_sdks: List[Dict]        # 命中的 SDK
    generated_steps: List[Dict]     # 生成的 Step（非 MISS 时为空）
    script_path: Optional[str]      # 生成脚本路径
    test_result: Optional[Dict]     # 测试结果
```

## 输出

```
generated_scripts/
├── Tc_Func_Node_001.py         # 生成的 pytest-bdd 脚本
├── Tc_Func_Node_001_result.json # 测试报告
├── Tc_Func_Node_002.py
├── Tc_Func_Node_002_result.json
├── ...
└── pipeline_summary.json         # 汇总报告
```

## 代码调用示例

```python
from core.pipeline import load_use_cases, run_pipeline

# 方式1: 快捷函数
results = run_pipeline(
    use_cases=use_cases,
    kb1_dir="knowledge-base/kb1_aw_sdk/indexed/chroma_db",
    kb2_dir="knowledge-base/kb2_bdd_scenarios/indexed/chroma_db"
)

# 方式2: Pipeline 类（更多控制）
from core.pipeline import Pipeline
from core.llm import create_llm

pipeline = Pipeline(
    kb1_dir="...",
    kb2_dir="...",
    llm=create_llm(),
    output_dir="generated_scripts"
)
for result in pipeline.run_batch(use_cases):
    print(result.use_case_id, result.hit, result.script_path)
```

## Prompt 设计

### Step 生成 Prompt（StepWriter → LLM）

```
你是一个 BDD Step 生成器。根据以下信息生成 When/And/Then 步骤链。

意图：{intent}
API 信息：
- 函数名：{func_name}
- 参数表：
{params_table}
- 调用示例：{example}

要求：
1. 第一步：When 构造{intent}请求，名称为"xxx"
2. 中间步骤：And 设置xxx参数
3. 最后一步：When 基于构造的请求，{intent}
4. 仅输出 Step 列表，每行一条 Gherkin 语句
```

### 脚本生成 Prompt（LLMGenerator → LLM）

```
你是一个 BDD 自动化测试脚本生成器。根据以下 Step 定义生成 Python pytest-bdd 脚本。

场景：{scenario}
Step 定义：
{steps_text}

要求：
1. 使用 pytest-bdd 框架
2. 每个 Step 对应一个 @when/@then/@given 装饰器函数
3. 使用 context 字典在步骤间传递数据
4. 仅输出 Python 代码
```
