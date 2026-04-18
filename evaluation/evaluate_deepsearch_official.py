import queue as _queue
import os
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from prompt import *
import traceback

try:
    import tiktoken
except ImportError:
    tiktoken = None
import random
import threading

from judge_utils import (
    _PriorityQueue,
    SlidingWindowRateLimiter,
    get_client,
    _is_rate_limit_error,
    _is_terminal_error,
    _rate_limit_delay,
)

MAX_JUDGE_RETRIES = 100
MAX_WORKERS = 64
JITTER = 0.5
MAX_CALLS_PER_MINUTE = 3000

rate_limiter = SlidingWindowRateLimiter(MAX_CALLS_PER_MINUTE)

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.getenv("API_BASE", "")

JUDGE_MODEL = "qwen-flash-2025-07-28"


extracted_answer_format_for_confidence = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "number"},
                "strict": {"type": "boolean"},
            },
            "required": [
                "extracted_final_answer",
                "reasoning",
                "correct",
                "confidence",
                "strict",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

extracted_answer_format_for_xbench = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "最终答案": {"type": "string"},
                "解释": {"type": "string"},
                "结论": {"type": "string", "enum": ["正确", "错误"]},
            },
            "required": ["最终答案", "解释", "结论"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def is_correct_judgement(judgement):
    return judgement.lower() == "correct" or (judgement and judgement.lower()[0] == "a")


def _judge_once(item, judge_prompt, dataset):
    """Single attempt at the judge API (no retry).
    Returns (result_dict | None, exception | None).
    json.JSONDecodeError in result means the response is unrecoverable."""
    rate_limiter.acquire()
    question = item["question"]
    correct_answer = item["answer"]
    response = item["prediction"].strip()
    prompt = judge_prompt.format(
        question=question, correct_answer=correct_answer, response=response
    )
    try:
        client = get_client()
        if dataset == "xbench-deepsearch":
            response_obj = client.beta.chat.completions.parse(
                model=JUDGE_MODEL,
                max_completion_tokens=16 * 1024,
                messages=[{"role": "user", "content": prompt}],
                response_format=extracted_answer_format_for_xbench,
                temperature=0,
                timeout=600.0,
            )
            raw_judge = json.loads(response_obj.choices[0].message.content)
            judgement = "Correct" if raw_judge["结论"].lower() == "正确" else ""
        elif "browsecomp" in dataset:
            resp = client.beta.chat.completions.parse(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16 * 1024,
                temperature=0,
                response_format=extracted_answer_format_for_confidence,
                timeout=600.0,
            )
            raw_judge = json.loads(resp.choices[0].message.content)
            judgement = "Correct" if raw_judge["correct"].lower() == "yes" else ""
        else:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16 * 1024,
                temperature=0,
                timeout=600.0,
            )
            judgement = resp.choices[0].message.content
        return {
            "question": question,
            "answer": correct_answer,
            "judgement": judgement,
        }, None
    except json.JSONDecodeError as e:
        return {
            "question": question,
            "answer": correct_answer,
            "judgement": "Error",
            "error": str(e),
        }, e
    except Exception as e:
        return None, e


def _queue_worker(q, results, lock, pending, pbar, judge_prompt, dataset):
    global MAX_JUDGE_RETRIES, JITTER
    while True:
        with lock:
            if pending[0] == 0:
                return
        try:
            retry_count, item = q.get(timeout=0.5)
        except _queue.Empty:
            continue

        result, exc = _judge_once(item, judge_prompt, dataset)

        if exc is None or isinstance(exc, json.JSONDecodeError):
            with lock:
                results.append(result)
                pending[0] -= 1
                pbar.update(1)
        elif _is_terminal_error(exc) or retry_count >= MAX_JUDGE_RETRIES - 1:
            if _is_terminal_error(exc):
                print(f"Terminal error, skipping: {item['question'][:60]}: {exc}")
            else:
                print(f"Max retries exceeded: {item['question'][:60]}: {exc}")
            with lock:
                results.append(
                    {
                        "question": item["question"],
                        "answer": item["answer"],
                        "judgement": "Error",
                        "error": str(exc),
                    }
                )
                pending[0] -= 1
                pbar.update(1)
        else:
            delay = _rate_limit_delay(retry_count, JITTER)
            if _is_rate_limit_error(exc):
                print(
                    f"Rate limit, requeueing in {delay:.1f}s (retry {retry_count+1}/{MAX_JUDGE_RETRIES})"
                )
            else:
                print(
                    f"Error, requeueing in {delay:.1f}s (retry {retry_count+1}/{MAX_JUDGE_RETRIES}): {exc}"
                )
            q.put(item, retry_count=retry_count + 1, delay=delay)


def run_judge_queue(items, judge_prompt, dataset, desc="Evaluating"):
    global MAX_WORKERS
    q = _PriorityQueue()
    for item in items:
        q.put(item)

    results = []
    lock = threading.Lock()
    pending = [len(items)]

    with tqdm(total=len(items), desc=desc) as pbar:
        workers = [
            threading.Thread(
                target=_queue_worker,
                args=(q, results, lock, pending, pbar, judge_prompt, dataset),
                daemon=True,
            )
            for _ in range(MAX_WORKERS)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

    return results


def process_single_round(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def single_round_statistics(input_file, tokenizer):
    contents = process_single_round(input_file)

    num_invalid, num_extra = 0, 0
    tool_use_cnt, visit_tool_cnt, search_tool_cnt, other_tool_cnt = [], [], [], []
    all_ans_lengths, all_think_lengths = [], []
    all_assistant_tokens_per_question, all_assistant_tokens_per_message = [], []
    termination_counts = {}

    for item in contents:
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
                remaining_content = content
                while (
                    "<tool_call>" in remaining_content
                    and "</tool_call>" in remaining_content
                ):
                    start_idx = remaining_content.find("<tool_call>")
                    end_idx = remaining_content.find("</tool_call>")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        tool_call_content = remaining_content[
                            start_idx + 11 : end_idx
                        ].strip()
                        if tool_call_content:
                            num_tool_use += 1
                            try:
                                tool_name = json.loads(tool_call_content).get(
                                    "name", ""
                                )
                                if tool_name == "search":
                                    num_search_tool += 1
                                elif "visit" in tool_name:
                                    num_visit_tool += 1
                                else:
                                    num_other_tool += 1
                            except Exception:
                                if "visit" in tool_call_content:
                                    num_visit_tool += 1
                                elif "search" in tool_call_content:
                                    num_search_tool += 1
                                else:
                                    num_other_tool += 1
                        remaining_content = remaining_content[end_idx + 12 :]
                    else:
                        break

                think_content = (
                    content.split("<think>")[-1].split("</think>")[0]
                    if "<think>" in content
                    else content
                )
                think_lengths.append(len(think_content))

                assistant_tokens = count_tokens(content, tokenizer)
                question_assistant_tokens += assistant_tokens
                total_tokens += assistant_tokens
                all_assistant_tokens_per_message.append(assistant_tokens)
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

    total_questions = len(contents)
    return {
        "extra_length": num_extra,
        "num_invalid": num_invalid,
        "avg_action": sum(tool_use_cnt) / len(tool_use_cnt),
        "avg_visit_action": sum(visit_tool_cnt) / len(visit_tool_cnt),
        "avg_search_action": sum(search_tool_cnt) / len(search_tool_cnt),
        "avg_other_action": sum(other_tool_cnt) / len(other_tool_cnt),
        "avg_ans_length": sum(all_ans_lengths) / len(all_ans_lengths),
        "avg_think_length": sum(all_think_lengths) / len(all_think_lengths),
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
        "termination_freq": {
            k: round(v / total_questions, 3) for k, v in termination_counts.items()
        },
    }


def aggregate_statistics(round1_file, round2_file, round3_file, tokenizer):
    stats = [
        single_round_statistics(f, tokenizer)
        for f in [round1_file, round2_file, round3_file]
    ]
    keys = stats[0].keys()
    avg_stats = {}
    for key in keys:
        if isinstance(stats[0][key], dict):
            all_keys = set().union(*(s[key].keys() for s in stats))
            avg_stats[key] = {
                k: round(sum(s[key].get(k, 0) for s in stats) / 3, 3) for k in all_keys
            }
        else:
            avg_stats[key] = round(sum(s[key] for s in stats) / 3, 3)
    return avg_stats


def calculate_enhanced_statistics(round_results, round_items, tokenizer):
    correct_tool_calls = []
    correct_assistant_tokens = []

    for round_name in ["round1", "round2", "round3"]:
        results = round_results[round_name]
        items = round_items[round_name]

        question_to_item = {}
        for item in items:
            try:
                q = item["messages"][1]["content"]
                question_to_item[q] = item
            except (IndexError, KeyError):
                pass

        for result in results:
            if not is_correct_judgement(result["judgement"]):
                continue
            item = question_to_item.get(result["question"])
            if item is None:
                continue

            messages = item["messages"]
            num_tool_use = 0
            question_assistant_tokens = 0

            for msg in messages:
                if msg["role"] == "assistant":
                    content = msg["content"]
                    remaining_content = content
                    while (
                        "<tool_call>" in remaining_content
                        and "</tool_call>" in remaining_content
                    ):
                        start_idx = remaining_content.find("<tool_call>")
                        end_idx = remaining_content.find("</tool_call>")
                        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                            tool_call_content = remaining_content[
                                start_idx + 11 : end_idx
                            ].strip()
                            if tool_call_content:
                                num_tool_use += 1
                            remaining_content = remaining_content[end_idx + 12 :]
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


def aggregate_results(round1_results, round2_results, round3_results):
    query_results = {}
    for results, round_name in zip(
        [round1_results, round2_results, round3_results], ["round1", "round2", "round3"]
    ):
        for result in results:
            query = result["question"]
            if query not in query_results:
                query_results[query] = {
                    "round1": None,
                    "round2": None,
                    "round3": None,
                    "answer": result["answer"],
                }
            if is_correct_judgement(result["judgement"]):
                query_results[query][round_name] = "Correct"
            else:
                query_results[query][round_name] = result["judgement"].capitalize()
    return query_results


def calculate_pass_at_k(query_results, k=3):
    total_correct = sum(
        1
        for results in query_results.values()
        if "Correct" in [results["round1"], results["round2"], results["round3"]][:k]
    )
    return round(total_correct / len(query_results) * 100, 2)


def calculate_best_pass_at_1(query_results):
    overall_best = max(
        sum(1 for r in query_results.values() if r[rn] == "Correct")
        / len(query_results)
        for rn in ["round1", "round2", "round3"]
    )
    return round(overall_best * 100, 2)


def calculate_avg_pass_at_3(query_results):
    round_names = ["round1", "round2", "round3"]
    avg_overall = sum(
        sum(1 for r in query_results.values() if r[rn] == "Correct")
        / len(query_results)
        for rn in round_names
    ) / len(round_names)
    return round(avg_overall * 100, 2)


# ── Auto-reuse existing scored files ───────────────────────────────────────────


def _load_results_from_scored(scored_file):
    """Reconstruct judge results list from an existing _scored.jsonl."""
    results = []
    with open(scored_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            results.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "judgement": item.get("judgement", "Error"),
                }
            )
    return results


def load_or_judge_round(jsonl_path, judge_prompt, dataset, round_name):
    """Load existing _scored.jsonl if available, otherwise run judge and save."""
    scored_path = jsonl_path.replace(".jsonl", "_scored.jsonl")
    if os.path.exists(scored_path):
        print(f"Loading existing {scored_path}")
        return _load_results_from_scored(scored_path)

    print(f"Judging {jsonl_path} ...")
    items = process_single_round(jsonl_path)
    results = run_judge_queue(
        items, judge_prompt, dataset, desc=f"Evaluating {round_name}"
    )

    # Save scored file
    index_map = {item["question"]: i for i, item in enumerate(items)}
    sorted_results = sorted(
        results, key=lambda x: index_map.get(x["question"], float("inf"))
    )
    with open(scored_path, "w", encoding="utf-8") as f:
        for orig_item, scored_result in zip(items, sorted_results):
            scored_item = {
                "is_correct": is_correct_judgement(scored_result["judgement"]),
                "judgement": scored_result["judgement"],
            }
            if "error" in scored_result:
                scored_item["error"] = scored_result["error"]
            scored_item.update(orig_item)
            f.write(json.dumps(scored_item, ensure_ascii=False) + "\n")

    return results


# ── Debug helper ───────────────────────────────────────────────────────────────


def visualize_debug(items, judge_prompt, dataset, n=3):
    print("=" * 70)
    print(f"[DEBUG] Loaded {len(items)} items. Showing first {n}:")
    print("=" * 70)
    for i, item in enumerate(items[:n]):
        print(f"\n--- Item {i} ---")
        print(f"  question   : {item['question'][:120]}...")
        print(f"  answer     : {item['answer']}")
        print(f"  prediction : {item['prediction'][:120]}...")

    print("\n" + "=" * 70)
    print("[DEBUG] Formatted judge prompt for item 0:")
    print("=" * 70)
    q = items[0]["question"]
    a = items[0]["answer"]
    p = items[0]["prediction"].strip()
    formatted = judge_prompt.format(question=q, correct_answer=a, response=p)
    print(formatted[:2000])

    print("\n" + "=" * 70)
    print("[DEBUG] Calling judge for item 0...")
    print("=" * 70)
    result, _ = _judge_once(items[0], judge_prompt, dataset)
    print(f"  judgement  : {result.get('judgement')}")
    print(f"  is_correct : {is_correct_judgement(result.get('judgement', ''))}")
    if "error" in result:
        print(f"  error      : {result['error']}")
    print("=" * 70)
    print("[DEBUG] Done. Exiting (remove --debug to run full evaluation).")
    print("=" * 70)


# ── Entry point ────────────────────────────────────────────────────────────────


def main():
    global MAX_JUDGE_RETRIES, MAX_WORKERS, JITTER, JUDGE_MODEL
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions across multiple rounds"
    )
    parser.add_argument("--input_folder", help="Path to prediction files")
    parser.add_argument(
        "--restore_result_path", default="summary.jsonl", help="record result"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="browsecomp_en",
        choices=[
            "gaia",
            "browsecomp_zh",
            "browsecomp_en",
            "browsecomp_en_full",
            "webwalker",
            "xbench-deepsearch",
        ],
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print sample data and one judge call, then exit.",
    )
    parser.add_argument(
        "--debug_n",
        type=int,
        default=3,
        help="Number of sample items to display in debug mode.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=MAX_JUDGE_RETRIES,
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=JITTER,
    )
    parser.add_argument(
        "--max_calls_per_minute",
        type=int,
        default=MAX_CALLS_PER_MINUTE,
    )
    parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL)
    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help="Override API_BASE for the judge model endpoint",
    )
    args = parser.parse_args()

    MAX_JUDGE_RETRIES = args.max_retries
    MAX_WORKERS = args.max_workers
    JITTER = args.jitter
    rate_limiter._max_calls = args.max_calls_per_minute
    JUDGE_MODEL = args.judge_model

    import judge_utils as _ju

    if args.base_url:
        _ju.BASE_URL = args.base_url

    dataset = args.dataset
    if dataset in ["gaia", "webwalker"]:
        judge_prompt = JUDGE_PROMPT_GAIA
    elif dataset == "xbench-deepsearch":
        judge_prompt = JUDGE_PROMPT_XBENCH
    elif dataset.startswith("browsecomp"):
        judge_prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL
    else:
        judge_prompt = JUDGE_PROMPT_GAIA
    print(f"Using {dataset} judge prompt ...")
    print(f"Judge prompt:\n {judge_prompt}")
    print(f"Judge model:\n {JUDGE_MODEL}")

    round1_file = os.path.join(args.input_folder, "iter1.jsonl")
    round2_file = os.path.join(args.input_folder, "iter2.jsonl")
    round3_file = os.path.join(args.input_folder, "iter3.jsonl")
    for file in [round1_file, round2_file, round3_file]:
        assert os.path.exists(
            file
        ), f"Prediction {file} not found, three rounds are required"

    round_items = {
        "round1": process_single_round(round1_file),
        "round2": process_single_round(round2_file),
        "round3": process_single_round(round3_file),
    }

    # Load tokenizer (fall back to tiktoken or char-count)
    try:
        tokenizer = AutoTokenizer.from_pretrained(os.getenv("TOKENIZER_PATH", ""))
    except Exception:
        if tiktoken is not None:
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            tokenizer = None

    if args.debug:
        debug_items = round_items.get("round1", [])
        if not debug_items:
            print("[DEBUG] No items found in round1. Check --input_folder path.")
        else:
            visualize_debug(debug_items, judge_prompt, dataset, n=args.debug_n)
        return

    # ── Judge (or load existing scored files) ──
    round_results = {
        "round1": load_or_judge_round(round1_file, judge_prompt, dataset, "round1"),
        "round2": load_or_judge_round(round2_file, judge_prompt, dataset, "round2"),
        "round3": load_or_judge_round(round3_file, judge_prompt, dataset, "round3"),
    }

    # ── Pass@k metrics ──
    aggr_results = aggregate_results(
        round_results["round1"], round_results["round2"], round_results["round3"]
    )
    pass_at_3 = calculate_pass_at_k(aggr_results, k=3)
    best_pass_at_1 = calculate_best_pass_at_1(aggr_results)
    avg_pass_at_3 = calculate_avg_pass_at_3(aggr_results)

    round_performance = {
        f"Round{i}_Pass@1": round(
            sum(
                1
                for r in round_results[f"round{i}"]
                if is_correct_judgement(r["judgement"])
            )
            / len(round_results[f"round{i}"])
            * 100,
            2,
        )
        for i in [1, 2, 3]
    }

    print(f"===========")
    print(f"Avg. Pass@3 {avg_pass_at_3}%")
    print(f"Best Pass@1 {best_pass_at_1}%")
    print(f"Pass@3 {pass_at_3}%")
    print(
        f"Pass@1 Round 1: {round_performance['Round1_Pass@1']}%  "
        f"Round 2: {round_performance['Round2_Pass@1']}%  "
        f"Round 3: {round_performance['Round3_Pass@1']}%\n"
    )

    # ── Statistics ──
    aggr_statistics = aggregate_statistics(
        round1_file, round2_file, round3_file, tokenizer
    )
    print(
        f"# Invalid {aggr_statistics['num_invalid']}  # Extra Length {aggr_statistics['extra_length']}"
    )
    print(
        f"Avg. Action {aggr_statistics['avg_action']:.2f}  "
        f"Avg. Visit Action {aggr_statistics['avg_visit_action']:.2f}  "
        f"Avg. Search Action {aggr_statistics['avg_search_action']:.2f}  "
        f"Avg. Other Action {aggr_statistics['avg_other_action']:.2f}"
    )
    print(
        f"Avg. Answer Length {aggr_statistics['avg_ans_length']:.2f}  "
        f"Avg. Thinking Length {aggr_statistics['avg_think_length']:.2f}"
    )

    enhanced_statistics = calculate_enhanced_statistics(
        round_results, round_items, tokenizer
    )
    print(f"\n=== ADDITIONAL STATISTICS ===")
    print(f"Avg. Tool Calls per Question: {aggr_statistics['avg_action']:.2f}")
    print(
        f"Avg. Tool Calls per Question (Correctly Solved): "
        f"{enhanced_statistics['avg_tool_calls_per_question_correctly_solved']:.2f}"
    )
    print(
        f"Avg. Assistant Tokens per Question: "
        f"{aggr_statistics['avg_assistant_tokens_per_question']:.2f}"
    )
    print(
        f"Avg. Assistant Tokens per Question (Correctly Solved): "
        f"{enhanced_statistics['avg_assistant_tokens_per_question_correctly_solved']:.2f}"
    )
    print(
        f"Avg. Assistant Tokens per Message: "
        f"{aggr_statistics['avg_assistant_tokens_per_message']:.2f}"
    )

    print(f"\n=== TERMINATION FREQUENCIES ===")
    for termination_type, frequency in aggr_statistics["termination_freq"].items():
        print(f"{termination_type}: {frequency:.3f}")

    print(f"===========")

    overall_eval_dict = {
        "dataset": dataset,
        "files": {"round1": round1_file, "round2": round2_file, "round3": round3_file},
        "overall": {
            "avg_pass_at_3": avg_pass_at_3,
            "best_pass_at_1": best_pass_at_1,
            "pass_at_3": pass_at_3,
        },
        "individual": round_performance,
        "statistics": {**aggr_statistics, **enhanced_statistics},
    }

    with open(args.restore_result_path, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(overall_eval_dict, ensure_ascii=False) + "\n")
    print(f"\nSummary appended to {args.restore_result_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"Evaluation Failed: {e}")
        print("Trace Back", error_str)
