# BDD-RAG-System

将自然语言用例文本转换为可执行的 pytest-bdd 自动化脚本的 RAG 系统。

## 特性

- **LLM + RAG**：基于大语言模型和检索增强生成技术
- **自动化脚本生成**：将自然语言用例自动转换为 pytest-bdd 测试脚本
- **多知识库支持**：SDK 知识库、BDD 场景库、Bug 报告库
- **自愈能力**：测试失败后自动诊断和修复

## 快速开始

### 前置条件

- Python 3.8+
- API Key（MiniMax 或 OpenAI 兼容接口）

### 安装依赖

```bash
pip install chromadb langchain langchain-core langchain-community
pip install sentence-transformers markdown-it-py PyYAML numpy
pip install bge-reranker huggingface_hub
```

### 配置

编辑 `config/config.yaml`，或设置环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
```

### 构建知识库（首次运行需要）

```bash
# KB1: SDK 文档嵌入
python scripts/embedding_sdk.py

# KB2: BDD 场景嵌入
python scripts/embedding_bdd.py

# KB3: Bug 报告嵌入
python scripts/embedding_bug.py
```

---

## 一、自动生成 BDD 脚本

### 运行命令

```bash
# 使用默认用例文件
python main.py

# 指定用例文件
python main.py --use-case-file docs/用例示例.md

# 指定输出目录
python main.py --output-dir generated_scripts

# 命令行指定 API Key
python main.py --api-key "your-key"
```

### 输入

- **用例文件**：`docs/用例示例.md` - 包含自然语言测试用例的 Markdown 文件

### 输出

- **生成脚本**：`generated_scripts/` 目录下的 `.py` 文件
- **汇总报告**：`generated_scripts/pipeline_summary.json`

### 代码调用示例

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

---

## 二、自动分析失败脚本

当 BDD 测试失败后，可以使用自愈流程进行自动诊断和生成分析报告。

### 运行命令

```bash
python3 scripts/test_self_healing_flow.py
```

### 输入

- **失败日志文件**：`docs/脚本失败日志示例.md` - 包含测试失败用例的日志

### 输出

- **分析报告**：`tickets/` 目录下的 Markdown 报告，按类型分类存放：
  - `tickets/script_error/` - 脚本问题报告
  - `tickets/env_fault/` - 环境故障报告
  - `tickets/resource_insufficient/` - 资源不足报告
  - `tickets/version_bug/` - 版本Bug报告
- **文件名格式**：`{case_id}_{timestamp}.md`（如 `Tc_Func_Node_001_20260728_113155.md`）

### 工作流程

```
失败日志输入
    │
    ▼
┌─────────────────────────────────────────────┐
│  解析失败日志，提取每个失败用例的 log_info   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           Fast Path 快速分类                 │
│  • 脚本错误 (SCRIPT_ERROR)                  │
│  • 资源不足 (RESOURCE_INSUFFICIENT)         │
│  • 版本Bug (VERSION_BUG)                    │
│  • 环境故障 (ENV_FAULT)                     │
│  • 未知 (UNKNOWN)                           │
└─────────────────────────────────────────────┘
    │
    ├── SCRIPT_ERROR / RESOURCE_INSUFFICIENT
    │    │  直接生成报告 ✅
    │    ▼
    │    tickets/{case_id}_{timestamp}.md
    │
    └── VERSION_BUG / ENV_FAULT / UNKNOWN
         │  移交给 Slow Path
         ▼
    ┌─────────────────────────────────────────────┐
    │           Slow Path 深度分析                 │
    │  • RAG 在 KB3 中检索相似 Bug                │
    │  • Rerank 取 Top-3 最相关记录               │
    │  • 判断检索是否命中（top-1 相似度 >= 0.4）  │
    │  • LLM 生成诊断结论                         │
    └─────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  tickets/{case_id}_{timestamp}.md           │
    │  - 检索命中：疑似已知 bug（Bug ID）         │
    │  - 检索 MISS：找不到已知 bug，疑似新版本问题 │
    └─────────────────────────────────────────────┘
```

### 错误分类说明

| 错误类型 | 关键词特征 | 处理路径 |
|----------|-----------|----------|
| SCRIPT_ERROR | `KeyError`, `TypeError`, `deprecated`, `Field is immutable` | Fast Path 直接生成报告 |
| RESOURCE_INSUFFICIENT | `Quota`, `exceeded`, `out of stock`, `No Space` | Fast Path 直接生成报告 |
| VERSION_BUG | `cgroup`, `containerd` | Slow Path RAG + LLM 分析 |
| ENV_FAULT | `401`, `403`, `404`, `Timeout`, `Connection refused` | Slow Path RAG + LLM 分析 |
| UNKNOWN | 无法匹配上述关键词 | Slow Path RAG + LLM 分析 |

### 检索 MISS 判断

当 RAG 检索结果满足以下任一条件时，判定为检索 MISS：
- **召回层 MISS**：`bug_matches` 为空，知识库无相关内容
- **精排层 MISS**：`top_bugs` 为空，或 top-1 相似度 < 0.4

检索 MISS 时，报告结论为"找不到已知 bug，疑似新版本问题"。

### 代码调用示例

```python
from self_healing.fast_path import FastPathHealer
from self_healing.slow_path import SlowPathHealer
from core.llm import create_llm

# 初始化
llm = create_llm()
fast_healer = FastPathHealer(llm=llm)
slow_healer = SlowPathHealer(
    llm=llm,
    kb3_dir="knowledge-base/kb3_bug_reports/indexed/chroma_db"
)

# 模拟失败日志信息
log_info = {
    'case_id': 'Tc_Func_Node_001',
    'type': 'error',
    'message': 'KeyError: node_req...',
    'step_name': 'When 基于构造的请求，创建node',
    'scenario': '重置节点并重新校验节点磁盘配置',
    'module': 'Node'
}

# Fast Path 诊断（返回 4 个值）
fixed, msg, ticket_path, category = fast_healer.diagnose_and_heal(log_info)

if fixed:
    print(f"Fast Path 直接修复: {msg}")
    print(f"报告路径: {ticket_path}")
else:
    print(f"Fast Path 无法修复: {msg}，移交给 Slow Path")
    # Slow Path 深度分析（传入 Fast Path 分类结果）
    _, report_msg, ticket_path = slow_healer.diagnose_and_heal(
        log_info,
        fast_path_category=category,
        case_id=log_info.get('case_id', '')
    )
    print(f"分析报告: {ticket_path}")
```

## 系统架构

BDD-RAG-System 包含三大核心阶段：

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           BDD-RAG-System 架构图                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    第一阶段：知识库构建 (Knowledge Base)                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │   │
│  │  │   KB1        │  │   KB2        │  │   KB3        │                   │   │
│  │  │   SDK接口    │  │   BDD场景    │  │   Bug报告    │                   │   │
│  │  │   定义       │  │   步骤定义   │  │   故障诊断   │                   │   │
│  │  │              │  │              │  │   知识       │                   │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │   │
│  │         │                 │                 │                            │   │
│  │         ▼                 ▼                 ▼                            │   │
│  │  ┌─────────────────────────────────────────────────────────────┐        │   │
│  │  │           BGE-M3 Embedding + ChromaDB 向量存储               │        │   │
│  │  └─────────────────────────────────────────────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                  第二阶段：BDD脚本自动生成 (Auto Script Generation)        │   │
│  │                                                                         │   │
│  │   用例文本 ──→ 解析 ──→ KB2向量检索 ──→ Reranker重排 ──┐                 │   │
│  │                                                 │                          │   │
│  │                              ┌──────────────────┴──────────────────┐      │   │
│  │                              │                                     │      │   │
│  │                         ┌────▼────┐                        ┌──────▼────┐  │   │
│  │                         │ HIT路径  │                        │ MISS路径  │  │   │
│  │                         │ KB2命中  │                        │ KB2未命中  │  │   │
│  │                         └────┬────┘                        └──────┬────┘  │   │
│  │                              │                                     │      │   │
│  │                              │                            ┌───────▼──────┐│  │
│  │                              │                            │  查KB1获取SDK ││  │
│  │                              │                            │  StepWriter  ││  │
│  │                              │                            └──────────────┘│  │
│  │                              │                                     │      │   │
│  │                              └─────────────┬───────────────────────┘      │   │
│  │                                            │                              │   │
│  │                                            ▼                              │   │
│  │                                   ┌─────────────────┐                     │   │
│  │                                   │  LLM生成脚本    │                     │   │
│  │                                   │  (MiniMax-M3)   │                     │   │
│  │                                   └────────┬────────┘                     │   │
│  │                                            │                              │   │
│  │                                            ▼                              │   │
│  │                                   ┌─────────────────┐                     │   │
│  │                                   │ ScriptVerifier  │                     │   │
│  │                                   │ 语法+API校验    │                     │   │
│  │                                   └────────┬────────┘                     │   │
│  │                                            │                              │   │
│  │                                            ▼                              │   │
│  │                                   ┌─────────────────┐                     │   │
│  │                                   │  LocalTester    │                     │   │
│  │                                   │  pytest-bdd运行 │                     │   │
│  │                                   └─────────────────┘                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                              测试结果 ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │              第三阶段：自动分析自愈 (Auto Failure Analysis & Self-Healing) │   │
│  │                                                                         │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │                      故障分类引擎                                 │   │   │
│  │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │   │
│  │   │  │ 脚本错误    │  │ 环境故障    │  │ 资源不足    │              │   │   │
│  │   │  │ Syntax/     │  │ 401/403/    │  │ Disk/Memory/│              │   │   │
│  │   │  │ Indent/Type │  │ Timeout     │  │ Quota       │              │   │   │
│  │   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │   │   │
│  │   └─────────┼────────────────┼────────────────┼──────────────────────┘   │   │
│  │             │                │                │                           │   │
│  │             ▼                ▼                ▼                           │   │
│  │   ┌─────────────────────────────────────────────────────────────┐      │   │
│  │   │                      自愈处理                                │      │   │
│  │   │  ┌─────────────────────────┐  ┌─────────────────────────┐   │      │   │
│  │   │  │    Fast Path 快速路径    │  │   Slow Path 慢速路径    │   │      │   │
│  │   │  │    (确定性错误自动修复)   │  │   (复杂问题RAG+LLM分析) │   │      │   │
│  │   │  │  • 脚本语法错误自动修复   │  │  • KB3 Bug知识库检索    │   │      │
│  │   │  │  • 环境故障通知Oncall   │  │  • LLM深度分析根因      │   │      │
│  │   │  │  • 资源问题清理/调整    │  │  • 生成排查单           │   │      │
│  │   │  └─────────────────────────┘  └─────────────────────────┘   │      │   │
│  │   └─────────────────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 数据流概览

| 阶段 | 输入 | 处理 | 输出 |
|------|------|------|------|
| **第一阶段** | SDK/BDD/Bug 原始文档 | BGE-M3 Embedding + ChromaDB | 向量化的知识库 |
| **第二阶段** | 自然语言用例 | RAG检索 → LLM生成 → 验证 | pytest-bdd 脚本 |
| **第三阶段** | 测试执行结果/错误日志 | 分类 → Fast/Slow Path | 自愈结果/排查单 |

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| **用例解析** | `core/pipeline/use_case_parser.py` | 解析用例文档为 `UseCase` 结构 |
| **Pipeline** | `core/pipeline/pipeline.py` | 串联 Phase2-5，复用已有模块 |
| **Retrieval** | `core/retrieval/retrieval.py` | Phase2+3，向量检索 + 重排 |
| **StepWriter** | `core/step_writer/step_writer.py` | MISS 时从 KB1 生成 BDD Step |
| **LLMGenerator** | `core/llm/llm.py` | 调用 LLM 生成 Step 和脚本 |
| **LocalTester** | `core/testing/testing.py` | 运行 pytest-bdd 并解析结果 |
| **ScriptVerifier** | `core/pipeline/script_verifier.py` | 语法 + API 签名校验 |

### 自愈模块 (Self-Healing)

| 模块 | 文件 | 职责 |
|------|------|------|
| **Fast Path** | `self-healing/fast_path/fast_path.py` | 快速路径 - 确定性错误自动修复 |
| **Slow Path** | `self-healing/slow_path/slow_path.py` | 慢速路径 - 复杂问题 RAG + LLM 分析 |

#### Fast Path 快速路径

处理**确定性高、简单的错误**，无需 LLM 介入：

| 错误类型 | 关键词特征 | 处理方式 |
|----------|-----------|----------|
| 脚本错误 | `SyntaxError`, `IndentationError`, `NameError`, `TypeError` | 自动修复脚本 |
| 环境故障 | `401 Unauthorized`, `403 Forbidden`, `Timeout`, `Connection Refused` | 通知 Oncall |
| 资源不足 | `No Space Left`, `Out of Memory`, `Quota Exceeded` | 清理资源/调整配额 |

#### Slow Path 慢速路径

处理**复杂逻辑、未知分支的疑难问题**，通过 RAG + LLM 深度分析：

1. **RAG 检索**：在 KB3 (Bug 知识库) 中查找相似故障记录
2. **Rerank 重排**：获取 Top-3 最相关的 Bug 记录
3. **LLM 分析**：深度分析根因和解决方案
4. **生成排查单**：创建工单等待人工确认
5. **知识库更新**：问题解决后回写知识库

### 自愈工作流

```
测试失败日志
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   故障分类引擎                        │
│  提取关键词 → 匹配错误类型 → 确定处理路径              │
└─────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  Fast Path    │       │  Slow Path    │
│  确定性错误    │       │  复杂问题      │
└───────┬───────┘       └───────┬───────┘
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ 自动修复脚本   │       │ RAG + LLM 分析 │
│ 或通知 Oncall │       │ 生成排查单     │
└───────────────┘       └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ 知识库更新     │
                        │ (学到的经验)   │
                        └───────────────┘
```

## 目录结构

```
bdd-rag-system/
├── main.py                      # 主入口
├── config/
│   ├── config.yaml              # LLM 配置
│   └── config_loader.py         # 配置加载工具
├── core/
│   ├── pipeline/                # Pipeline 编排器
│   │   ├── pipeline.py
│   │   ├── use_case_parser.py
│   │   └── step_prompt_builder.py
│   ├── retrieval/               # 向量检索 (ChromaDB + BGE-M3)
│   │   └── retrieval.py
│   ├── rerank/                  # 重排模块 (bge-reranker)
│   │   └── rerank.py
│   ├── step_writer/             # Step 生成
│   │   └── step_writer.py
│   ├── llm/                     # LLM 调用
│   │   └── llm.py
│   └── testing/                 # 本地测试执行
│       └── testing.py
├── knowledge-base/              # 三个知识库
│   ├── kb1_aw_sdk/              # KB1: SDK 接口定义
│   ├── kb2_bdd_scenarios/       # KB2: BDD 场景
│   └── kb3_bug_reports/         # KB3: Bug 报告
├── scripts/                     # Embedding 脚本
│   ├── embedding_sdk.py
│   ├── embedding_bdd.py
│   └── embedding_bug.py
├── self_healing/                # 自愈模块
│   ├── fast_path/               # 快速修复
│   └── slow_path/               # 复杂问题处理
├── docs/                        # 文档
│   └── 用例示例.md              # 示例用例
└── generated_scripts/           # 生成脚本输出目录
```

## 配置说明

`config/config.yaml` 配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `llm_provider` | LLM 提供商 (`anthropic` / `openai`) | `openai` |
| `base_url` | API 地址 | `https://api.minimaxi.com/v1` |
| `api_key` | API Key | 环境变量 `OPENAI_API_KEY` |
| `model` | 模型名称 | `MiniMax-M3` |
| `max_tokens` | 最大 token 数 | `4096` |

## 输出

```
generated_scripts/
├── Tc_Func_Node_001.py         # 生成的 pytest-bdd 脚本
├── Tc_Func_Node_001_result.json
├── Tc_Func_Node_002.py
├── Tc_Func_Node_002_result.json
└── pipeline_summary.json         # 汇总报告
```

## 技术栈

| 组件 | 技术 |
|------|------|
| Embedding | BAAI/bge-m3, BAAI/bge-large-zh-v1.5 |
| Reranker | bge-reranker |
| Vector DB | ChromaDB |
| LLM | OpenAI 兼容接口 (MiniMax-M3) |
| Framework | LangChain |
| Parser | markdown-it-py |

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

## 许可证

MIT License
