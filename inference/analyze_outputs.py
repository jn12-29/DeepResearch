import json
import os
import re
from collections import defaultdict
from pathlib import Path

outputs_dir = Path(__file__).parent / "outputs"

jsonl_files = sorted(outputs_dir.rglob("*.jsonl"))
if not jsonl_files:
    print("No JSONL files found in outputs/")
    exit(0)

for filepath in jsonl_files:
    rel_path = filepath.relative_to(outputs_dir)
    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    total_messages = 0
    total_tool_calls = 0
    tool_counts = defaultdict(int)

    for sample in samples:
        messages = sample.get("messages", [])
        total_messages += len(messages)
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            for tc in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
                total_tool_calls += 1
                try:
                    obj = json.loads(tc.strip())
                    name = obj.get("name", "unknown")
                except Exception:
                    # PythonInterpreter uses non-standard format
                    if "python" in tc.lower():
                        name = "PythonInterpreter"
                    else:
                        name = "unknown"
                tool_counts[name] += 1

    print(f"\n{'='*60}")
    print(f"File : {rel_path}")
    print(f"Samples       : {len(samples)}")
    print(f"Total messages: {total_messages}")
    print(f"Total tool calls: {total_tool_calls}")
    if tool_counts:
        print("Tool call breakdown:")
        for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {name:25s} {count}")
    else:
        print("  (no tool calls found)")
