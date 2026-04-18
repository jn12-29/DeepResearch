# Evaluation

这里提供两个评测脚本，分别对应 **HLE** 和 **DeepSearch**（GAIA / BrowseComp / WebWalker / XBench）系列基准。

---

## 前置配置

在项目根目录的 `.env` 文件中设置 Judge 模型的 API 凭证：

```bash
API_KEY=your_api_key
API_BASE=https://your-api-endpoint/v1
```

两个脚本启动时都会自动 `source` 该文件。

---

## run_dr.sh — DeepSearch 系列评测

用于评测 GAIA、BrowseComp、WebWalker、XBench-DeepSearch 等数据集，计算 **Pass@1 / Pass@3 / Avg. Pass@3** 等指标。

**要求**：输入目录下必须同时存在 `iter1.jsonl`、`iter2.jsonl`、`iter3.jsonl`（三轮推理结果）。

### 用法

```bash
bash evaluation/run_dr.sh <input_folder> [dataset] [extra_args...]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_folder` | 包含 iter1/2/3.jsonl 的目录（必填） | — |
| `dataset` | 数据集名称（见下表） | `gaia` |
| `--judge_model` | Judge 模型名称 | `qwen-flash-2025-07-28` |
| `--base_url` | Judge 模型的 API Base URL（覆盖 .env 中的 API_BASE） | — |
| `--max_workers` | 并发评测线程数 | `64` |
| `--max_calls_per_minute` | Judge API 限速（次/分钟） | `30` |
| `--restore_result_path` | 汇总结果追加写入的 JSONL 文件 | `summary.jsonl` |
| `--debug` | 打印样本数据并调用一次 Judge 后退出 | — |

**支持的 dataset 值**：

| 值 | 对应基准 |
|----|----------|
| `gaia` | GAIA 2023 Validation |
| `browsecomp_en` | BrowseComp English |
| `browsecomp_en_full` | BrowseComp English Full |
| `browsecomp_zh` | BrowseComp Chinese |
| `webwalker` | WebWalker |
| `xbench-deepsearch` | XBench-DeepSearch |

### 示例

```bash
# 基本用法
bash evaluation/run_dr.sh /path/to/outputs gaia

# 指定 Judge 模型和 Base URL
bash evaluation/run_dr.sh /path/to/outputs browsecomp_en \
    --judge_model gpt-4o \
    --base_url https://api.openai.com/v1

# Debug 模式（验证配置是否正确）
bash evaluation/run_dr.sh /path/to/outputs gaia --debug
```

### 输出

- 每轮的评分结果保存为 `iter1_scored.jsonl` / `iter2_scored.jsonl` / `iter3_scored.jsonl`（与输入文件同目录）
- 汇总指标追加写入 `summary.jsonl`
- 已存在的 `_scored.jsonl` 文件会被直接复用，无需重复调用 Judge API

---

## run_hle.sh — HLE 评测

用于评测 HLE（Humanity's Last Exam），支持**单文件**和**目录**两种模式。

### 用法

```bash
bash evaluation/run_hle.sh <input_folder_or_file> [tokenizer_path] [extra_args...]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_folder_or_file` | 单个 .jsonl 文件或包含 iter*.jsonl 的目录 | `$OUTPUT_PATH`（来自 .env） |
| `tokenizer_path` | 用于统计 token 数的分词器路径（HuggingFace 格式） | `$MODEL_PATH`（来自 .env） |
| `--judge_model` | Judge 模型名称 | `qwen-flash-2025-07-28` |
| `--base_url` | Judge 模型的 API Base URL（覆盖 .env 中的 API_BASE） | — |
| `--max_workers` | 并发评测线程数 | `10` |
| `--max_calls_per_minute` | Judge API 限速（次/分钟） | `30` |
| `--repeat_times` | 单文件模式：重复评测次数（用于估算方差） | `1` |

### 两种运行模式

**单文件模式**（输入为 `.jsonl` 文件）：

对单轮推理结果做一次性评测，输出 Pass@1。

```bash
bash evaluation/run_hle.sh /path/to/iter1.jsonl /path/to/tokenizer
```

生成文件：
- `iter1.eval_details.jsonl`：每条样本的详细判断
- `iter1.report.json`：汇总指标

**目录模式**（输入为目录）：

自动枚举目录下所有 `iter*.jsonl` 文件并逐一评测（与 run_dr.sh 不同，HLE 不做跨轮 Pass@k 聚合）。

```bash
bash evaluation/run_hle.sh /path/to/outputs /path/to/tokenizer
```

### 示例

```bash
# 单文件，使用默认 Judge 模型
bash evaluation/run_hle.sh outputs/iter1.jsonl /model/Tongyi-DeepResearch-30B-A3B

# 指定 Judge 模型和 Base URL
bash evaluation/run_hle.sh outputs/iter1.jsonl /model/Tongyi-DeepResearch-30B-A3B \
    --judge_model gpt-4o-mini \
    --base_url https://api.openai.com/v1

# 目录模式，批量评测
bash evaluation/run_hle.sh outputs/ /model/Tongyi-DeepResearch-30B-A3B
```

---

## 直接调用 Python 脚本

如需更灵活的控制，可以跳过 shell 脚本直接调用：

```bash
# DeepSearch 系列
python evaluation/evaluate_deepsearch_official.py \
    --input_folder /path/to/outputs \
    --dataset gaia \
    --judge_model gpt-4o \
    --base_url https://api.openai.com/v1 \
    --max_workers 32

# HLE 单文件
python evaluation/evaluate_hle_official.py \
    --input_fp /path/to/iter1.jsonl \
    --tokenizer_path /path/to/tokenizer \
    --judge_model gpt-4o \
    --base_url https://api.openai.com/v1

# HLE 目录（3轮 Pass@k）
python evaluation/evaluate_hle_official.py \
    --input_folder /path/to/outputs \
    --tokenizer_path /path/to/tokenizer \
    --judge_model gpt-4o \
    --base_url https://api.openai.com/v1
```
