import os

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("API_BASE", "")

from pydantic import BaseModel
from typing import Literal
import json
import queue as _queue
from tqdm import tqdm
import threading
from collections import Counter
from transformers import AutoTokenizer
import argparse

try:
    import tiktoken
except ImportError:
    tiktoken = None

from judge_utils import (
    _PriorityQueue,
    SlidingWindowRateLimiter,
    get_client,
    _is_rate_limit_error,
    _is_terminal_error,
    _rate_limit_delay,
)

JUDGE_MODEL = "qwen-flash-2025-07-28"
MAX_WORKERS = 5
MAX_RETRIES = 100
JITTER = 0.5
MAX_CALLS_PER_MINUTE = 3000

rate_limiter = SlidingWindowRateLimiter(MAX_CALLS_PER_MINUTE)


def load_jsonl(fp):
    with open(fp, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(data, fp):
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(line, ensure_ascii=False) for line in data) + "\n")


JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]


def _extract_once(question, correct_answer, response):
    """Single attempt at the judge API (no retry).
    Returns (result_dict | None, exception | None)."""
    rate_limiter.acquire()
    client = get_client()
    prompt = JUDGE_PROMPT.format(
        question=question, correct_answer=correct_answer, response=response
    )
    try:
        response_obj = client.beta.chat.completions.parse(
            model=JUDGE_MODEL,
            max_completion_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
            timeout=60.0,
        )
        content = response_obj.choices[0].message.parsed
        return {
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence,
        }, None
    except Exception as e:
        return None, e


def _process_item_once(item, tokenizer):
    """Single attempt to fully process one item.
    Returns (report | None, exception | None, terminal: bool)."""
    response = item["prediction"]
    question, correct_answer = item["question"], item["answer"]
    token_usage = item.get("usage", "")
    tool_usage = Counter()

    judge_result, exc = _extract_once(question, correct_answer, response)
    if exc is not None:
        terminal = _is_terminal_error(exc)
        return None, exc, terminal

    acc = 1 if judge_result["correct"] in ("y", "yes", "true", "positive") else 0

    messages = item.get("messages", [])
    context_text = " ".join(m.get("content", "") for m in messages)
    context_length = count_tokens(context_text, tokenizer)

    report = {
        "acc": acc,
        "turns": 0,
        "token_usage": token_usage,
        "tool_usage": tool_usage,
        "item": item,
        "context_length": context_length,
        "dollars_o4mini": 0,
        "is_answer": 1,
    }
    return report, None, False


def _queue_worker(q, results, lock, pending, pbar, tokenizer):
    global MAX_RETRIES, JITTER
    while True:
        with lock:
            if pending[0] == 0:
                return
        try:
            retry_count, item = q.get(timeout=0.5)
        except _queue.Empty:
            continue

        report, exc, terminal = _process_item_once(item, tokenizer)

        if exc is None:
            with lock:
                results.append(report)
                pending[0] -= 1
                pbar.update(1)
        elif terminal or retry_count >= MAX_RETRIES - 1:
            if terminal:
                print(f"Terminal error, skipping: {item['question'][:60]}: {exc}")
            else:
                print(f"Max retries exceeded: {item['question'][:60]}: {exc}")
            with lock:
                results.append(
                    {
                        "acc": 0,
                        "turns": 0,
                        "token_usage": "",
                        "tool_usage": Counter(),
                        "item": item,
                        "context_length": 0,
                        "dollars_o4mini": 0,
                        "is_answer": 0,
                    }
                )
                pending[0] -= 1
                pbar.update(1)
        else:
            delay = _rate_limit_delay(retry_count, JITTER)
            if _is_rate_limit_error(exc):
                print(
                    f"Rate limit, requeueing in {delay:.1f}s (retry {retry_count+1}/{MAX_RETRIES})"
                )
            else:
                print(
                    f"Error, requeueing in {delay:.1f}s (retry {retry_count+1}/{MAX_RETRIES}): {exc}"
                )
            q.put(item, retry_count=retry_count + 1, delay=delay)


def run_process_queue(items, tokenizer):
    """Run judge calls via priority queue."""
    global MAX_WORKERS
    q = _PriorityQueue()
    for item in items:
        q.put(item)

    results = []
    lock = threading.Lock()
    pending = [len(items)]

    with tqdm(total=len(items), desc="Processing") as pbar:
        workers = [
            threading.Thread(
                target=_queue_worker,
                args=(q, results, lock, pending, pbar, tokenizer),
                daemon=True,
            )
            for _ in range(MAX_WORKERS)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

    return results


# ── Statistics helpers (aligned with evaluate_deepsearch_official.py) ──────────


def get_termination_value(item):
    if "termination" in item:
        return item["termination"]
    messages = item.get("messages", [])
    if not messages:
        return "unknown"
    last_message = messages[-1]["content"] if messages else ""
    if "max_turns_reached" in last_message.lower():
        return "max_turns_reached"
    elif "max_tokens_reached" in last_message.lower():
        return "max_tokens_reached"
    elif "<answer>" in last_message and "</answer>" in last_message:
        return "answered"
    else:
        return "unknown"


def count_tokens(text, tokenizer):
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text) // 4


def single_round_statistics(items, tokenizer):
    num_invalid, num_extra = 0, 0
    tool_use_cnt, visit_tool_cnt, search_tool_cnt, other_tool_cnt = [], [], [], []
    all_ans_lengths, all_think_lengths = [], []
    all_assistant_tokens_per_question, all_assistant_tokens_per_message = [], []
    termination_counts = {}

    for item in items:
        messages = item["messages"]
        final_msg = messages[-1]["content"] if messages else ""

        if "<answer>" not in final_msg or "</answer>" not in final_msg:
            num_invalid += 1
            answer_length = 0
        else:
            answer_length = len(
                final_msg.split("<answer>")[1].split("</answer>")[0].strip()
            )

        num_tool_use, num_visit_tool, num_search_tool, num_other_tool = 0, 0, 0, 0
        think_lengths = []
        question_assistant_tokens = 0
        total_tokens = 0

        for msg in messages:
            if msg["role"] == "assistant":
                content = msg["content"]
                remaining = content
                while "<tool_call>" in remaining and "</tool_call>" in remaining:
                    s = remaining.find("<tool_call>")
                    e = remaining.find("</tool_call>")
                    if s != -1 and e != -1 and e > s:
                        tc = remaining[s + 11 : e].strip()
                        if tc:
                            num_tool_use += 1
                            try:
                                name = json.loads(tc).get("name", "")
                                if name == "search":
                                    num_search_tool += 1
                                elif "visit" in name:
                                    num_visit_tool += 1
                                else:
                                    num_other_tool += 1
                            except Exception:
                                if "visit" in tc:
                                    num_visit_tool += 1
                                elif "search" in tc:
                                    num_search_tool += 1
                                else:
                                    num_other_tool += 1
                        remaining = remaining[e + 12 :]
                    else:
                        break

                think_content = (
                    content.split("<think>")[-1].split("</think>")[0]
                    if "<think>" in content
                    else content
                )
                think_lengths.append(len(think_content))

                atokens = count_tokens(content, tokenizer)
                question_assistant_tokens += atokens
                total_tokens += atokens
                all_assistant_tokens_per_message.append(atokens)
            else:
                total_tokens += count_tokens(msg["content"], tokenizer)

        tool_use_cnt.append(num_tool_use)
        visit_tool_cnt.append(num_visit_tool)
        search_tool_cnt.append(num_search_tool)
        other_tool_cnt.append(num_other_tool)
        all_ans_lengths.append(answer_length)
        all_think_lengths.append(
            sum(think_lengths) / len(think_lengths) if think_lengths else 0
        )
        all_assistant_tokens_per_question.append(question_assistant_tokens)

        termination = get_termination_value(item)
        termination_counts[termination] = termination_counts.get(termination, 0) + 1

        if total_tokens > 30000:
            num_extra += 1

    n = len(items)
    return {
        "extra_length": num_extra,
        "num_invalid": num_invalid,
        "avg_action": sum(tool_use_cnt) / n,
        "avg_visit_action": sum(visit_tool_cnt) / n,
        "avg_search_action": sum(search_tool_cnt) / n,
        "avg_other_action": sum(other_tool_cnt) / n,
        "avg_ans_length": sum(all_ans_lengths) / n,
        "avg_think_length": sum(all_think_lengths) / n,
        "avg_assistant_tokens_per_question": (
            sum(all_assistant_tokens_per_question)
            / len(all_assistant_tokens_per_question)
            if all_assistant_tokens_per_question
            else 0
        ),
        "avg_assistant_tokens_per_message": (
            sum(all_assistant_tokens_per_message)
            / len(all_assistant_tokens_per_message)
            if all_assistant_tokens_per_message
            else 0
        ),
        "termination_freq": {k: round(v / n, 3) for k, v in termination_counts.items()},
    }


def aggregate_statistics(round_items_dict, tokenizer):
    stats = {
        rn: single_round_statistics(items, tokenizer)
        for rn, items in round_items_dict.items()
    }
    keys = list(next(iter(stats.values())).keys())
    avg = {}
    for key in keys:
        sample = stats[list(stats.keys())[0]][key]
        if isinstance(sample, dict):
            all_keys = set().union(*(s[key].keys() for s in stats.values()))
            avg[key] = {
                k: round(sum(s[key].get(k, 0) for s in stats.values()) / len(stats), 3)
                for k in all_keys
            }
        else:
            avg[key] = round(sum(s[key] for s in stats.values()) / len(stats), 3)
    return avg


def calculate_enhanced_statistics(round_results, round_items_dict, tokenizer):
    correct_tool_calls, correct_assistant_tokens = [], []

    for round_name, results in round_results.items():
        items = round_items_dict[round_name]
        q_to_item = {}
        for item in items:
            try:
                q_to_item[item["question"]] = item
            except KeyError:
                pass

        for result in results:
            if result["acc"] != 1:
                continue
            item = q_to_item.get(result["item"]["question"])
            if item is None:
                continue

            num_tool_use = 0
            question_assistant_tokens = 0
            for msg in item["messages"]:
                if msg["role"] != "assistant":
                    continue
                content = msg["content"]
                remaining = content
                while "<tool_call>" in remaining and "</tool_call>" in remaining:
                    s = remaining.find("<tool_call>")
                    e = remaining.find("</tool_call>")
                    if s != -1 and e != -1 and e > s:
                        if remaining[s + 11 : e].strip():
                            num_tool_use += 1
                        remaining = remaining[e + 12 :]
                    else:
                        break
                question_assistant_tokens += count_tokens(content, tokenizer)

            correct_tool_calls.append(num_tool_use)
            correct_assistant_tokens.append(question_assistant_tokens)

    return {
        "avg_tool_calls_per_question_correctly_solved": round(
            (
                sum(correct_tool_calls) / len(correct_tool_calls)
                if correct_tool_calls
                else 0
            ),
            3,
        ),
        "avg_assistant_tokens_per_question_correctly_solved": round(
            (
                sum(correct_assistant_tokens) / len(correct_assistant_tokens)
                if correct_assistant_tokens
                else 0
            ),
            3,
        ),
    }


# ── Pass@k helpers ──────────────────────────────────────────────────────────────


def build_query_results(round_results):
    """Map question → {round1: 'Correct'|other, round2: ..., round3: ..., answer: str}."""
    query_results = {}
    for round_name, results in round_results.items():
        for r in results:
            q = r["item"]["question"]
            if q not in query_results:
                query_results[q] = {rn: None for rn in round_results}
                query_results[q]["answer"] = r["item"]["answer"]
            query_results[q][round_name] = "Correct" if r["acc"] == 1 else "Incorrect"
    return query_results


def calculate_pass_at_k(query_results, round_names, k=3):
    total = sum(
        1
        for v in query_results.values()
        if "Correct" in [v[rn] for rn in round_names[:k]]
    )
    return round(total / len(query_results) * 100, 2)


def calculate_best_pass_at_1(query_results, round_names):
    best = max(
        sum(1 for v in query_results.values() if v[rn] == "Correct")
        / len(query_results)
        for rn in round_names
    )
    return round(best * 100, 2)


def calculate_avg_pass_at_n(query_results, round_names):
    avg = sum(
        sum(1 for v in query_results.values() if v[rn] == "Correct")
        / len(query_results)
        for rn in round_names
    ) / len(round_names)
    return round(avg * 100, 2)


# ── Single-file mode (backward compat) ─────────────────────────────────────────


def run_single_file(input_fp, tokenizer, repeat_times=1):
    d = load_jsonl(input_fp) * repeat_times
    res = run_process_queue(d, tokenizer)
    write_jsonl(res, input_fp.replace(".jsonl", ".eval_details.jsonl"))

    metrics = sum(i["acc"] for i in res) / len(res) if res else 0
    is_answer_rate = sum(1 for i in res if i["is_answer"]) / len(res)

    report = {
        "evaluated_nums": len(d),
        "valid_nums": len(res),
        "metrics": metrics * 100,
        "judge_model": JUDGE_MODEL,
        "is_answer_rate": is_answer_rate,
        "repeat_times": repeat_times,
    }
    print(report)
    report_path = input_fp.replace(".jsonl", ".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)


# ── Folder mode ─────────────────────────────────────────────────────────────────


def load_or_judge_round(jsonl_path, eval_details_path, tokenizer):
    """Load existing eval_details if available, otherwise run judge and save."""
    if os.path.exists(eval_details_path):
        print(f"Loading existing {eval_details_path}")
        return load_jsonl(eval_details_path)

    print(f"Judging {jsonl_path} ...")
    items = load_jsonl(jsonl_path)
    results = run_process_queue(items, tokenizer)
    write_jsonl(results, eval_details_path)
    return results


def run_folder_mode(input_folder, tokenizer, restore_result_path):
    round_files = {
        f"round{i}": os.path.join(input_folder, f"iter{i}.jsonl") for i in [1, 2, 3]
    }
    for rn, fp in round_files.items():
        assert os.path.exists(fp), f"{fp} not found; all 3 rounds are required"

    round_names = ["round1", "round2", "round3"]

    # Load or judge each round
    round_results = {}
    for rn, jsonl_path in round_files.items():
        eval_details_path = jsonl_path.replace(".jsonl", ".eval_details.jsonl")
        round_results[rn] = load_or_judge_round(
            jsonl_path, eval_details_path, tokenizer
        )

    # Load original items for statistics
    round_items = {rn: load_jsonl(fp) for rn, fp in round_files.items()}

    # ── Accuracy metrics ──
    query_results = build_query_results(round_results)
    pass_at_3 = calculate_pass_at_k(query_results, round_names, k=3)
    best_pass_at_1 = calculate_best_pass_at_1(query_results, round_names)
    avg_pass_at_3 = calculate_avg_pass_at_n(query_results, round_names)
    round_performance = {
        f"Round{i}_Pass@1": round(
            sum(1 for r in round_results[f"round{i}"] if r["acc"] == 1)
            / len(round_results[f"round{i}"])
            * 100,
            2,
        )
        for i in [1, 2, 3]
    }

    print("===========")
    print(f"Avg. Pass@3 {avg_pass_at_3}%")
    print(f"Best Pass@1 {best_pass_at_1}%")
    print(f"Pass@3 {pass_at_3}%")
    print(
        f"Pass@1  Round 1: {round_performance['Round1_Pass@1']}%  "
        f"Round 2: {round_performance['Round2_Pass@1']}%  "
        f"Round 3: {round_performance['Round3_Pass@1']}%"
    )

    # ── Statistics ──
    aggr_stats = aggregate_statistics(round_items, tokenizer)
    enhanced = calculate_enhanced_statistics(round_results, round_items, tokenizer)

    print(
        f"# Invalid {aggr_stats['num_invalid']}  # Extra Length {aggr_stats['extra_length']}"
    )
    print(
        f"Avg. Action {aggr_stats['avg_action']:.2f}  "
        f"Avg. Visit {aggr_stats['avg_visit_action']:.2f}  "
        f"Avg. Search {aggr_stats['avg_search_action']:.2f}  "
        f"Avg. Other {aggr_stats['avg_other_action']:.2f}"
    )
    print(
        f"Avg. Answer Length {aggr_stats['avg_ans_length']:.2f}  "
        f"Avg. Thinking Length {aggr_stats['avg_think_length']:.2f}"
    )
    print("\n=== ADDITIONAL STATISTICS ===")
    print(f"Avg. Tool Calls per Question: {aggr_stats['avg_action']:.2f}")
    print(
        f"Avg. Tool Calls per Question (Correctly Solved): "
        f"{enhanced['avg_tool_calls_per_question_correctly_solved']:.2f}"
    )
    print(
        f"Avg. Assistant Tokens per Question: "
        f"{aggr_stats['avg_assistant_tokens_per_question']:.2f}"
    )
    print(
        f"Avg. Assistant Tokens per Question (Correctly Solved): "
        f"{enhanced['avg_assistant_tokens_per_question_correctly_solved']:.2f}"
    )
    print(
        f"Avg. Assistant Tokens per Message: "
        f"{aggr_stats['avg_assistant_tokens_per_message']:.2f}"
    )
    print("\n=== TERMINATION FREQUENCIES ===")
    for t, freq in aggr_stats["termination_freq"].items():
        print(f"{t}: {freq:.3f}")
    print("===========")

    overall = {
        "dataset": "hle",
        "files": {rn: fp for rn, fp in round_files.items()},
        "overall": {
            "avg_pass_at_3": avg_pass_at_3,
            "best_pass_at_1": best_pass_at_1,
            "pass_at_3": pass_at_3,
        },
        "individual": round_performance,
        "statistics": {**aggr_stats, **enhanced},
    }

    with open(restore_result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(overall, ensure_ascii=False) + "\n")
    print(f"\nSummary appended to {restore_result_path}")

    return overall


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Folder mode (3-round Pass@k evaluation)
    parser.add_argument(
        "--input_folder",
        type=str,
        default="",
        help="Folder containing iter1/2/3.jsonl; enables Pass@k mode",
    )
    parser.add_argument(
        "--restore_result_path",
        type=str,
        default="summary.jsonl",
        help="JSONL file to append the summary report to",
    )
    # Single-file mode (backward compat)
    parser.add_argument(
        "--input_fp",
        type=str,
        default="",
        help="Single JSONL file for one-round evaluation",
    )
    parser.add_argument("--repeat_times", type=int, default=1)
    # Common
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--jitter", type=float, default=JITTER)
    parser.add_argument(
        "--max_calls_per_minute", type=int, default=MAX_CALLS_PER_MINUTE
    )
    parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL)
    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help="Override API_BASE for the judge model endpoint",
    )

    args = parser.parse_args()
    MAX_WORKERS = args.max_workers
    MAX_RETRIES = args.max_retries
    JITTER = args.jitter
    rate_limiter._max_calls = args.max_calls_per_minute
    JUDGE_MODEL = args.judge_model

    import judge_utils as _ju

    if args.base_url:
        _ju.BASE_URL = args.base_url

    # Load tokenizer (fall back to tiktoken or char-count)
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    elif tiktoken is not None:
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    else:
        tokenizer = None

    if args.input_folder:
        run_folder_mode(args.input_folder, tokenizer, args.restore_result_path)
    elif args.input_fp:
        run_single_file(args.input_fp, tokenizer, args.repeat_times)
    else:
        parser.error("Provide either --input_folder or --input_fp")
