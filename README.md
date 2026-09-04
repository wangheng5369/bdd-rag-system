# BDD-RAG-System

A RAG system that converts natural language test case descriptions into executable pytest-bdd automation scripts.

## Features

- **LLM + RAG**: Powered by Large Language Models and Retrieval-Augmented Generation
- **Automated Script Generation**: Automatically converts natural language test cases into pytest-bdd test scripts
- **Multi-Knowledge Base Support**: SDK knowledge base, BDD scenario library, and Bug report library
- **Self-Healing**: Automatically diagnose and fix failures after test runs

## Quick Start

### Prerequisites

- Python 3.8+
- API Key (MiniMax or OpenAI-compatible endpoint)

### Install Dependencies

```bash
pip install chromadb langchain langchain-core langchain-community
pip install sentence-transformers markdown-it-py PyYAML numpy
pip install bge-reranker huggingface_hub
```

### Configuration

Edit `config/config.yaml`, or set environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
```

### Build Knowledge Bases (required for first run)

```bash
# KB1: SDK Documentation Embedding
python scripts/embedding_sdk.py

# KB2: BDD Scenario Embedding
python scripts/embedding_bdd.py

# KB3: Bug Report Embedding
python scripts/embedding_bug.py
```

---

## I. Automated BDD Script Generation

### Run Command

```bash
# Use default use case file
python main.py

# Specify use case file
python main.py --use-case-file docs/用例示例.md

# Specify output directory
python main.py --output-dir generated_scripts

# Specify API Key via command line
python main.py --api-key "your-key"
```

### Input

- **Use Case File**: `docs/用例示例.md` - Markdown file containing natural language test cases

### Output

- **Generated Scripts**: `.py` files in `generated_scripts/`
- **Summary Report**: `generated_scripts/pipeline_summary.json`

### Code Usage Example

```python
from core.pipeline import load_use_cases, run_pipeline

# Method 1: Convenience function
results = run_pipeline(
    use_cases=use_cases,
    kb1_dir="knowledge-base/kb1_aw_sdk/indexed/chroma_db",
    kb2_dir="knowledge-base/kb2_bdd_scenarios/indexed/chroma_db"
)

# Method 2: Pipeline class (more control)
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

## II. Automated Failure Analysis

When a BDD test fails, use the self-healing workflow for automatic diagnosis and report generation.

### Self-Healing Run Command

```bash
python3 scripts/test_self_healing_flow.py
```

### Self-Healing Input

- **Failure Log File**: `docs/脚本失败日志示例.md` - Log containing test failure cases

### Self-Healing Output

- **Analysis Reports**: Markdown reports in `tickets/`, organized by type:
  - `tickets/script_error/` - Script error reports
  - `tickets/env_fault/` - Environment fault reports
  - `tickets/resource_insufficient/` - Insufficient resource reports
  - `tickets/version_bug/` - Version bug reports
- **Filename Format**: `{case_id}_{timestamp}.md` (e.g., `Tc_Func_Node_001_20260728_113155.md`)

### Workflow

```
Failure Log Input
    │
    ▼
┌─────────────────────────────────────────────┐
│  Parse failure log, extract log_info for    │
│  each failed test case                       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           Fast Path Quick Classification     │
│  • Script Error (SCRIPT_ERROR)              │
│  • Resource Insufficient (RESOURCE_INSUFFICIENT)
│  • Version Bug (VERSION_BUG)                │
│  • Environment Fault (ENV_FAULT)           │
│  • Unknown (UNKNOWN)                        │
└─────────────────────────────────────────────┘
    │
    ├── SCRIPT_ERROR / RESOURCE_INSUFFICIENT
    │    │  Generate report directly ✅
    │    ▼
    │    tickets/{case_id}_{timestamp}.md
    │
    └── VERSION_BUG / ENV_FAULT / UNKNOWN
         │  Escalate to Slow Path
         ▼
    ┌─────────────────────────────────────────────┐
    │           Slow Path Deep Analysis            │
    │  • RAG searches KB3 for similar Bugs       │
    │  • Rerank selects Top-3 most relevant      │
    │  • Check hit (top-1 similarity >= 0.4)     │
    │  • LLM generates diagnosis                 │
    └─────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  tickets/{case_id}_{timestamp}.md           │
    │  - Hit: Suspected known bug (Bug ID)       │
    │  - Miss: No known bug found, likely new    │
    └─────────────────────────────────────────────┘
```

### Error Classification

| Error Type | Keyword Features | Processing Path |
|------------|------------------|----------------|
| SCRIPT_ERROR | `KeyError`, `TypeError`, `deprecated`, `Field is immutable` | Fast Path - direct report |
| RESOURCE_INSUFFICIENT | `Quota`, `exceeded`, `out of stock`, `No Space` | Fast Path - direct report |
| VERSION_BUG | `cgroup`, `containerd` | Slow Path - RAG + LLM |
| ENV_FAULT | `401`, `403`, `404`, `Timeout`, `Connection refused` | Slow Path - RAG + LLM |
| UNKNOWN | Cannot match any keywords above | Slow Path - RAG + LLM |

### Retrieval MISS Detection

A retrieval MISS is判定 when any of the following conditions are met:
- **Recall Layer MISS**: `bug_matches` is empty, no relevant content in knowledge base
- **Rerank Layer MISS**: `top_bugs` is empty, or top-1 similarity < 0.4

On retrieval MISS, the report concludes: "No known bug found, suspected new version issue".

### Code Usage Example

```python
from self_healing.fast_path import FastPathHealer
from self_healing.slow_path import SlowPathHealer
from core.llm import create_llm

# Initialize
llm = create_llm()
fast_healer = FastPathHealer(llm=llm)
slow_healer = SlowPathHealer(
    llm=llm,
    kb3_dir="knowledge-base/kb3_bug_reports/indexed/chroma_db"
)

# Simulated failure log info
log_info = {
    'case_id': 'Tc_Func_Node_001',
    'type': 'error',
    'message': 'KeyError: node_req...',
    'step_name': 'When create node with constructed request',
    'scenario': 'Reset node and verify disk configuration',
    'module': 'Node'
}

# Fast Path diagnosis (returns 4 values)
fixed, msg, ticket_path, category = fast_healer.diagnose_and_heal(log_info)

if fixed:
    print(f"Fast Path fixed directly: {msg}")
    print(f"Report path: {ticket_path}")
else:
    print(f"Fast Path cannot fix: {msg}, escalate to Slow Path")
    # Slow Path deep analysis (pass Fast Path classification result)
    _, report_msg, ticket_path = slow_healer.diagnose_and_heal(
        log_info,
        fast_path_category=category,
        case_id=log_info.get('case_id', '')
    )
    print(f"Analysis report: {ticket_path}")
```

## System Architecture

BDD-RAG-System consists of three core phases:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           BDD-RAG-System Architecture                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │               Phase 1: Knowledge Base Construction                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │   │
│  │  │   KB1       │  │   KB2       │  │   KB3       │                   │   │
│  │  │   SDK API   │  │   BDD       │  │   Bug       │                   │   │
│  │  │   Definitions│  │   Scenarios │  │   Reports   │                   │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │   │
│  │         │                 │                 │                            │   │
│  │         ▼                 ▼                 ▼                            │   │
│  │  ┌─────────────────────────────────────────────────────────────┐        │   │
│  │  │           BGE-M3 Embedding + ChromaDB Vector Store          │        │   │
│  │  └─────────────────────────────────────────────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │               Phase 2: BDD Script Auto Generation                        │   │
│  │                                                                         │   │
│  │   Use Case Text ──→ Parse ──→ KB2 Vector Search ──→ Rerank ──┐           │   │
│  │                                                 │                          │   │
│  │                              ┌──────────────────┴──────────────────┐      │   │
│  │                              │                                     │      │   │
│  │                         ┌────▼────┐                        ┌───────▼────┐│  │
│  │                         │  HIT   │                        │   MISS    ││  │
│  │                         │  Path  │                        │   Path    ││  │
│  │                         └────┬────┘                        └───────┬────┘│  │
│  │                              │                                     │      │  │
│  │                              │                            ┌───────▼──────┐│ │
│  │                              │                            │ Query KB1   ││ │
│  │                              │                            │ for SDK     ││ │
│  │                              │                            │ StepWriter  ││ │
│  │                              │                            └─────────────┘│ │
│  │                              │                                     │      │  │
│  │                              └─────────────┬───────────────────────┘      │  │
│  │                                            │                              │  │
│  │                                            ▼                              │  │
│  │                                   ┌─────────────────┐                     │  │
│  │                                   │  LLM Generate   │                     │  │
│  │                                   │  Script        │                     │  │
│  │                                   │  (MiniMax-M3)  │                     │  │
│  │                                   └────────┬────────┘                     │  │
│  │                                            │                              │  │
│  │                                            ▼                              │  │
│  │                                   ┌─────────────────┐                     │  │
│  │                                   │ ScriptVerifier  │                     │  │
│  │                                   │ Syntax+API      │                     │  │
│  │                                   └────────┬────────┘                     │  │
│  │                                            │                              │  │
│  │                                            ▼                              │  │
│  │                                   ┌─────────────────┐                     │  │
│  │                                   │  LocalTester    │                     │  │
│  │                                   │  pytest-bdd     │                     │  │
│  │                                   └─────────────────┘                     │  │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                              Test Results ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │       Phase 3: Auto Failure Analysis & Self-Healing                       │   │
│  │                                                                         │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │                      Fault Classification Engine                   │   │   │
│  │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │   │
│  │   │  │ Script Err │  │  Env Fault │  │ Resource    │              │   │   │
│  │   │  │ Syntax/    │  │ 401/403/   │  │ Insufficient│              │   │   │
│  │   │  │ Indent/Type│  │  Timeout   │  │ Disk/Memory/│              │   │   │
│  │   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │   │   │
│  │   └─────────┼────────────────┼────────────────┼──────────────────────┘   │   │
│  │             │                │                │                           │   │
│  │             ▼                ▼                ▼                           │   │
│  │   ┌─────────────────────────────────────────────────────────────┐      │   │
│  │   │                      Self-Healing Processing                │      │   │
│  │   │  ┌─────────────────────────┐  ┌─────────────────────────┐   │      │   │
│  │   │  │    Fast Path           │  │    Slow Path            │   │      │   │
│  │   │  │    (Deterministic Err  │  │    (Complex Issues      │   │      │   │
│  │   │  │     Auto-Fix)         │  │     RAG+LLM Analysis)   │   │      │   │
│  │   │  │  • Script syntax fix  │  │  • KB3 Bug search      │   │      │
│  │   │  │  • Env fault notify  │  │  • LLM root cause      │   │      │
│  │   │  │  • Resource cleanup  │  │  • Generate ticket     │   │      │
│  │   │  └─────────────────────────┘  └─────────────────────────┘   │      │   │
│  │   └─────────────────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Overview

| Phase | Input | Processing | Output |
|-------|-------|------------|--------|
| **Phase 1** | SDK/BDD/Bug raw docs | BGE-M3 Embedding + ChromaDB | Vectorized knowledge base |
| **Phase 2** | Natural language use cases | RAG retrieval → LLM generation → Verification | pytest-bdd scripts |
| **Phase 3** | Test execution results/error logs | Classification → Fast/Slow Path | Self-healing results/tickets |

### Core Modules

| Module | File | Responsibility |
|--------|------|----------------|
| **UseCase Parser** | `core/pipeline/use_case_parser.py` | Parse use case documents into `UseCase` structures |
| **Pipeline** | `core/pipeline/pipeline.py` | Orchestrate Phase 2-5, reuse existing modules |
| **Retrieval** | `core/retrieval/retrieval.py` | Phase 2+3, vector search + reranking |
| **StepWriter** | `core/step_writer/step_writer.py` | Generate BDD Steps from KB1 on MISS |
| **LLMGenerator** | `core/llm/llm.py` | Call LLM to generate Steps and scripts |
| **LocalTester** | `core/testing/testing.py` | Run pytest-bdd and parse results |
| **ScriptVerifier** | `core/pipeline/script_verifier.py` | Syntax + API signature verification |

### Self-Healing Modules

| Module | File | Responsibility |
|--------|------|----------------|
| **Fast Path** | `self-healing/fast_path/fast_path.py` | Fast path - deterministic error auto-fix |
| **Slow Path** | `self-healing/slow_path/slow_path.py` | Slow path - complex issues via RAG + LLM |

#### Fast Path

Handles **high-confidence, simple errors** without LLM intervention:

| Error Type | Keyword Features | Handling |
|------------|------------------|----------|
| Script Error | `SyntaxError`, `IndentationError`, `NameError`, `TypeError` | Auto-fix script |
| Environment Fault | `401 Unauthorized`, `403 Forbidden`, `Timeout`, `Connection Refused` | Notify Oncall |
| Resource Insufficient | `No Space Left`, `Out of Memory`, `Quota Exceeded` | Clean up resources/adjust quota |

#### Slow Path

Handles **complex logic and unknown branches** via RAG + LLM deep analysis:

1. **RAG Retrieval**: Search KB3 (Bug knowledge base) for similar failure records
2. **Rerank**: Get Top-3 most relevant Bug records
3. **LLM Analysis**: Deep analysis of root cause and solution
4. **Generate Ticket**: Create ticket for human confirmation
5. **Knowledge Base Update**: Write back after resolution

### Self-Healing Workflow

```
Test Failure Log
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Fault Classification Engine            │
│  Extract keywords → Match error type → Determine   │
│  processing path                                    │
└─────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  Fast Path    │       │  Slow Path    │
│  Deterministic│       │  Complex      │
└───────┬───────┘       └───────┬───────┘
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Auto-fix or   │       │ RAG + LLM     │
│ notify Oncall │       │ Generate      │
└───────────────┘       │ ticket        │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Knowledge Base│
                        │ Update        │
                        │ (learned)     │
                        └───────────────┘
```

## Directory Structure

```
bdd-rag-system/
├── main.py                      # Main entry point
├── config/
│   ├── config.yaml              # LLM configuration
│   └── config_loader.py         # Config loader utility
├── core/
│   ├── pipeline/                # Pipeline orchestrator
│   │   ├── pipeline.py
│   │   ├── use_case_parser.py
│   │   └── step_prompt_builder.py
│   ├── retrieval/               # Vector search (ChromaDB + BGE-M3)
│   │   └── retrieval.py
│   ├── rerank/                  # Rerank module (bge-reranker)
│   │   └── rerank.py
│   ├── step_writer/             # Step generation
│   │   └── step_writer.py
│   ├── llm/                     # LLM invocation
│   │   └── llm.py
│   └── testing/                 # Local test execution
│       └── testing.py
├── knowledge-base/              # Three knowledge bases
│   ├── kb1_aw_sdk/              # KB1: SDK interface definitions
│   ├── kb2_bdd_scenarios/       # KB2: BDD scenarios
│   └── kb3_bug_reports/         # KB3: Bug reports
├── scripts/                     # Embedding scripts
│   ├── embedding_sdk.py
│   ├── embedding_bdd.py
│   └── embedding_bug.py
├── self_healing/                # Self-healing modules
│   ├── fast_path/               # Fast fix
│   └── slow_path/               # Complex issue handling
├── docs/                        # Documentation
│   └── 用例示例.md              # Sample use cases
└── generated_scripts/           # Generated script output directory
```

## Configuration

`config/config.yaml` options:

| Config | Description | Default |
|--------|-------------|---------|
| `llm_provider` | LLM provider (`anthropic` / `openai`) | `openai` |
| `base_url` | API endpoint | `https://api.minimaxi.com/v1` |
| `api_key` | API Key | Environment variable `OPENAI_API_KEY` |
| `model` | Model name | `MiniMax-M3` |
| `max_tokens` | Max tokens | `4096` |

## Output

```
generated_scripts/
├── Tc_Func_Node_001.py         # Generated pytest-bdd script
├── Tc_Func_Node_001_result.json
├── Tc_Func_Node_002.py
├── Tc_Func_Node_002_result.json
└── pipeline_summary.json         # Summary report
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embedding | BAAI/bge-m3, BAAI/bge-large-zh-v1.5 |
| Reranker | bge-reranker |
| Vector DB | ChromaDB |
| LLM | OpenAI-compatible interface (MiniMax-M3) |
| Framework | LangChain |
| Parser | markdown-it-py |

## Code Usage Example

```python
from core.pipeline import load_use_cases, run_pipeline

# Method 1: Convenience function
results = run_pipeline(
    use_cases=use_cases,
    kb1_dir="knowledge-base/kb1_aw_sdk/indexed/chroma_db",
    kb2_dir="knowledge-base/kb2_bdd_scenarios/indexed/chroma_db"
)

# Method 2: Pipeline class (more control)
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

## License

MIT License
