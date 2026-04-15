import os
from pydantic import BaseModel
from typing import Literal
import json
import queue as _queue
from tqdm import tqdm
import threading
import random
import re
from collections import Counter, defaultdict
from transformers import AutoTokenizer
import argparse

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
MAX_CALLS_PER_MINUTE = 30

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

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]


def _extract_once(question, correct_answer, response):
    """Single attempt at the judge API (no retry).
    Returns (result_dict | None, exception | None).
    Caller checks exc type: 'length limit' str → terminal, else retryable."""
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
    report = {
        "acc": acc,
        "turns": 0,
        "token_usage": token_usage,
        "tool_usage": tool_usage,
        "item": item,
        "context_length": len(tokenizer.encode("")),
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
                pending[0] -= 1
                pbar.update(1)
        else:
            if _is_rate_limit_error(exc):
                delay = _rate_limit_delay(retry_count, JITTER)
                print(
                    f"Rate limit, requeueing in {delay:.1f}s (retry {retry_count+1}/{MAX_RETRIES})"
                )
            else:
                delay = random.uniform(1 - JITTER, 1 + JITTER)
                print(
                    f"Error, requeueing in {delay:.1f}s (retry {retry_count+1}/{MAX_RETRIES}): {exc}"
                )
            q.put(item, retry_count=retry_count + 1, delay=delay)


def run_process_queue(items, tokenizer):
    """Run judge calls via priority queue — workers never sleep,
    rate-limited items are re-scheduled so fresh items aren't starved."""
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


def extract_response(messages):
    ANSWER_TAG = "answer"

    def get_answers(text: str) -> dict | None:
        pattern = r"<{TAG}>(.*?)</{TAG}>".format(TAG=ANSWER_TAG)
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answer_output = match.group(1).strip()
            return answer_output, 1
        return text, 0

    response, success_flag = get_answers(messages["records"][-1]["content"])

    return response, success_flag


def process_item(item, tokenizer):

    success_flag = 1

    response = item["prediction"]
    question, correct_answer = item["question"], item["answer"]
    token_usage = item.get("usage", "")
    tool_usage = Counter()

    judge_result, _ = _extract_once(question, correct_answer, response)
    acc = 1 if judge_result["correct"] in ("y", "yes", "true", "positive") else 0
    context = ""

    report = {
        "acc": acc,
        "turns": 0,
        "token_usage": token_usage,
        "tool_usage": tool_usage,
        "item": item,
        "context_length": len(tokenizer.encode(context)),
        "dollars_o4mini": 0,
        "is_answer": success_flag,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", type=str, default="")
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of parallel judge threads (default: %(default)s).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=MAX_RETRIES,
        help="Max retry attempts per judge call (default: %(default)s).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=JITTER,
        help="Retry backoff jitter as a fraction, e.g. 0.25 means ±25%% (default: %(default)s).",
    )
    parser.add_argument(
        "--max_calls_per_minute",
        type=int,
        default=MAX_CALLS_PER_MINUTE,
        help="Max API calls per minute via sliding window (default: %(default)s).",
    )

    args = parser.parse_args()
    MAX_WORKERS = args.max_workers
    MAX_RETRIES = args.max_retries
    JITTER = args.jitter
    rate_limiter._max_calls = args.max_calls_per_minute

    input_fp = args.input_fp
    d = load_jsonl(input_fp) * args.repeat_times
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    res = run_process_queue(d, tokenizer)

    metrics = [i["acc"] for i in res]
    metrics = sum(metrics) / len(res) if res else 0
    is_answer_rate = len([i for i in res if i["is_answer"]]) / len(res)
    completion_tokens = 0
    prompt_tokens = 0
    for i in res:
        if i["token_usage"]:
            completion_tokens += i["token_usage"]["completion_tokens"]
            prompt_tokens += i["token_usage"]["prompt_tokens"]
    avg_completion_tokens = completion_tokens / len(res)
    avg_prompt_tokens = prompt_tokens / len(res)
    acc_under_turns, turns_dist, context_length_under_turns, dollars_under_turn = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    for i in res:
        for turn in range(31):
            if i["turns"] <= turn and i["acc"] == 1:
                acc_under_turns[turn] += 1
        turns_dist[i["turns"]] += 1
        context_length_under_turns[i["turns"]] += i["context_length"]
        dollars_under_turn[i["turns"]] += i["dollars_o4mini"]

    for i in context_length_under_turns:
        context_length_under_turns[i] /= turns_dist[i]
    context_length_under_turns = sorted(
        context_length_under_turns.items(), key=lambda x: x[0], reverse=False
    )
    for i in dollars_under_turn:
        dollars_under_turn[i] /= turns_dist[i]
    dollars_under_turn = sorted(
        dollars_under_turn.items(), key=lambda x: x[0], reverse=False
    )

    token_usage_under_turns = Counter()
    for i in res:
        for turn in range(31):
            if i["turns"] <= turn and i["acc"] == 1:
                acc_under_turns[turn] += 1

    tool_usage = defaultdict(list)
    tool_usage_correct = defaultdict(list)
    for i in res:
        cur_usage = {
            "google_search": 0,
            "google_scholar": 0,
            "Visit": 0,
            "PythonInterpreter": 0,
        }
        for tool_use in i["tool_usage"]:
            if tool_use in cur_usage:
                cur_usage[tool_use] += i["tool_usage"][tool_use]
        for tool_use in cur_usage:
            if tool_use in cur_usage:
                tool_usage[tool_use].append(cur_usage[tool_use])
            if tool_use in cur_usage and i["acc"] == 1:
                tool_usage_correct[tool_use].append(cur_usage[tool_use])
    tool_usage = {k: sum(v) / len(res) for k, v in tool_usage.items()}
    correct_num = len([i for i in res if i["acc"] == 1])
    tool_usage_correct = {
        k: sum(v) / correct_num for k, v in tool_usage_correct.items()
    }

    avg_turns = sum([i["turns"] for i in res]) / len(res)
    avg_turns_correct = sum([i["turns"] for i in res if i["acc"] == 1]) / len(
        [i["turns"] for i in res if i["acc"] == 1]
    )

    report = {
        "evaluated_nums": len(d),
        "valid_nums": len(res),
        "metrics": metrics * 100,
        "judge_model": JUDGE_MODEL,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "avg_dollars_o4mini": avg_prompt_tokens / 1000000 * 1.1
        + avg_completion_tokens / 1000000 * 4.4,
        "avg_dollars_claude": avg_prompt_tokens / 1000000 * 3
        + avg_completion_tokens / 1000000 * 15,
        "tool_usage": tool_usage,
        "tool_usage_correct": tool_usage_correct,
        "is_answer_rate": is_answer_rate,
        "repeat_times": args.repeat_times,
        "avg_turns": avg_turns,
        "avg_turns_correct": avg_turns_correct,
        "turns_dist": turns_dist,
    }
    print(report)
    eval_details_path = input_fp.replace(".jsonl", ".eval_details.jsonl")
    write_jsonl(res, eval_details_path)
    report_path = input_fp.replace(".jsonl", ".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
