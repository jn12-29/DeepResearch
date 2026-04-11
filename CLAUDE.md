# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tongyi DeepResearch** is an agentic LLM system (30B-A3B parameters) for long-horizon deep information-seeking tasks. It uses a ReAct inference paradigm where the model iteratively calls tools (web search, page visits, Python interpreter, file parsing, Google Scholar) until it reaches a final `<answer>`.

## Environment Setup

Requires Python **3.10.0** (other versions may cause dependency issues).

```bash
conda create -n react_infer_env python=3.10
conda activate react_infer_env
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys and model path
```

## Key Configuration (`.env`)

| Variable                  | Purpose                                                   |
| ------------------------- | --------------------------------------------------------- |
| `MODEL_PATH`              | Path to local model weights (Tongyi-DeepResearch-30B-A3B) |
| `DATASET`                 | Path to evaluation data file (.json or .jsonl)            |
| `OUTPUT_PATH`             | Directory for saving results                              |
| `SERPER_KEY_ID`           | Serper.dev API key for web search                         |
| `JINA_API_KEYS`           | Jina.ai API key for page reading                          |
| `API_KEY` / `API_BASE`    | OpenAI-compatible API for page summarization              |
| `DASHSCOPE_API_KEY`       | Alibaba Dashscope for file parsing                        |
| `SANDBOX_FUSION_ENDPOINT` | Python sandbox endpoints (comma-separated)                |

## Running Inference

```bash
# Full pipeline: starts vLLM server + runs inference
bash inference/run_react_infer.sh
```

## Running Evaluation

```bash
cd evaluation

# HLE benchmark
export API_KEY=... BASE_URL=...
python evaluate_hle_official.py --input_fp <output_folder> --model_path <qwen_model_path>

# Other benchmarks (GAIA, BrowseComp, FRAMES, etc.)
export OPENAI_API_KEY=... OPENAI_API_BASE=... API_KEY=... BASE_URL=... Qwen2_5_7B_PATH=...
python evaluate_deepsearch_official.py --input_fp <output_folder> --dataset <dataset_name>
```

## Architecture

### Core Inference Loop (`inference/`)

- **`react_agent.py`** — `MultiTurnReactAgent` drives the ReAct loop: calls vLLM server via OpenAI-compatible API, parses `<tool_call>`/`<tool_response>` XML tags, enforces token budget (110K context limit), retries with backoff, terminates on `<answer>` tags.
- **`run_multi_react.py`** — Orchestrator: loads dataset, manages multiple rollouts, distributes tasks across vLLM ports via round-robin, runs inference in parallel with `ThreadPoolExecutor`, writes results as JSONL.
- **`run_react_infer.sh`** — Entry point: validates config, starts vLLM servers per-GPU, waits for readiness, invokes `run_multi_react.py`.
- **`prompt.py`** — System prompt defining tool signatures in XML format and the `EXTRACTOR_PROMPT` for page summarization.

### Tools (`inference/tool_*.py`)

| Tool                | File              | Purpose                                    |
| ------------------- | ----------------- | ------------------------------------------ |
| `search`            | `tool_search.py`  | Batched Google web search via Serper API   |
| `visit`             | `tool_visit.py`   | Fetch + summarize web pages via Jina       |
| `google_scholar`    | `tool_scholar.py` | Academic search via Serper Scholar API     |
| `PythonInterpreter` | `tool_python.py`  | Execute Python in SandboxFusion            |
| `parse_file`        | `tool_file.py`    | Parse PDFs, DOCX, XLSX, etc. via Dashscope |

Files for `parse_file` must be placed in `inference/eval_data/file_corpus/` and referenced by filename in the question.

### Tool Call Format

The model outputs tool calls in this XML format (not standard OpenAI function calling):

```xml
<tool_call>
{"name": "search", "arguments": {"query": ["query1", "query2"]}}
</tool_call>
```

Python calls use a special code-block syntax:

```xml
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
print("hello")
</code>
</tool_call>
```

Tool results are injected as `<tool_response>...</tool_response>` user messages.

### Output Format

Results are written as JSONL with fields: `question`, `answer` (ground truth), `messages` (full conversation), `prediction`, `termination` (reason for stopping).

Multiple rollouts produce `iter1.jsonl`, `iter2.jsonl`, etc. Distributed splits produce `iter1_split1of2.jsonl`, etc. The system resumes from partial output (skips already-processed questions).

### WebAgent Sub-projects (`WebAgent/`)

The repo contains predecessor research agents as reference implementations:

- **WebSailor** — Earlier ReAct-based search agent
- **WebWalker** — Web traversal benchmark agent
- **WebDancer** — Autonomous information-seeking agent with Gradio demo
- **NestBrowse** — Browser-use learning with MCP/async tools
- **ParallelMuse** — Parallel reasoning aggregation
- **AgentFold** — Proactive context management
- **WebResummer** — Context summarization agent
- **WebWatcher** — Vision-language deep research agent

### OpenRouter Usage

To use OpenRouter instead of a local model, edit `inference/react_agent.py`:

1. Set API key/base to OpenRouter credentials in `call_server()`
2. Change model name to `alibaba/tongyi-deepresearch-30b-a3b`
3. Uncomment lines 87–88 to prepend `reasoning_content` to `content`
