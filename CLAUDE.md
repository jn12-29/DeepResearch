# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Personal Preferences

- Use English for code and annotations, but communicate with me in Chinese.
- Update README.md, CLAUDE.md, and any \*.sh files when modifications are needed.

## Project Overview

**Tongyi DeepResearch** is an agentic LLM system (30B-A3B parameters) for long-horizon deep information-seeking tasks. It uses a ReAct inference paradigm where the model iteratively calls tools (web search, page visits, Python interpreter, file parsing, Google Scholar) until it reaches a final `<answer>`.

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

## Evaluation Results (`inference/outputs/` & `summary.jsonl`)

### Models Evaluated

| Directory | Model |
|-----------|-------|
| `Tongyi-DeepResearch-30B-A3B_sglang` | FP16 full-precision |
| `Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang` | Int4 W4A16 quantized |

### Datasets

| Dataset key | Directory | Description |
|-------------|-----------|-------------|
| `xbench-deepsearch` | `DeepSearch-2510` | 2510-question Chinese deep search benchmark |
| `gaia` | `gaia_2023_validation` | GAIA 2023 validation set |
| `hle` | `hle_text_200` | Humanity's Last Exam, 200 text-only questions |

### Directory Layout

```
inference/outputs/<model>/<dataset>/
    iter{N}.jsonl              # Raw inference output
    iter{N}_scored.jsonl       # Scored (DeepSearch / GAIA); adds is_correct, judgement
    iter{N}.eval_details.jsonl # Detailed eval metadata (HLE); fields: acc, turns,
                               #   token_usage, tool_usage, item, context_length,
                               #   dollars_o4mini, is_answer
    statistics.md              # Optional human-readable stats summary
```

Timestamped sibling directories (e.g. `DeepSearch-2510_20260418_205255`) contain **re-scored results** using a different judge model or evaluation method on the **same inference outputs** — the questions and model predictions are identical across base and timestamped directories.

### `summary.jsonl` Schema

Each line is a JSON object representing one evaluation run (model × dataset × judge):

```jsonc
{
  "dataset": "xbench-deepsearch" | "gaia" | "hle",
  "files": { "round1": "…/iter1.jsonl", "round2": "…/iter2.jsonl", "round3": "…/iter3.jsonl" },
  "overall": {
    "avg_pass_at_3": float,   // mean of 3 rounds' Pass@1
    "best_pass_at_1": float,  // best single-round Pass@1
    "pass_at_3": float        // oracle: correct if any round correct
  },
  "individual": { "Round1_Pass@1": float, "Round2_Pass@1": float, "Round3_Pass@1": float },
  "statistics": {
    "avg_action": float,                // total tool calls per question
    "avg_search_action": float,
    "avg_visit_action": float,
    "avg_other_action": float,
    "avg_ans_length": float,            // tokens in final answer
    "avg_think_length": float,
    "avg_assistant_tokens_per_question": float,
    "avg_assistant_tokens_per_message": float,
    "termination_freq": { "<reason>": float, … },
    "num_invalid": float,
    "extra_length": float,
    "avg_tool_calls_per_question_correctly_solved": float,
    "avg_assistant_tokens_per_question_correctly_solved": float
  }
}
```
