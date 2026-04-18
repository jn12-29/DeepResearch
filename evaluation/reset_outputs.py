import argparse
import glob
import os
import re
import shutil
from datetime import datetime

ARCHIVE_PATTERN = re.compile(r"_\d{6,}$|_old$")


def is_archived(name):
    return bool(ARCHIVE_PATTERN.search(name))


def get_raw_iters(folder):
    """Return iter*.jsonl paths that are raw (no _scored, no .eval_details)."""
    candidates = glob.glob(os.path.join(folder, "iter*.jsonl"))
    return [
        p for p in candidates
        if "_scored" not in os.path.basename(p)
        and ".eval_details" not in os.path.basename(p)
    ]


def reset_folder(subfolder, timestamp, dry_run):
    name = os.path.basename(subfolder)
    parent = os.path.dirname(subfolder)
    archived = os.path.join(parent, f"{name}_{timestamp}")

    raw_files = get_raw_iters(subfolder)
    if not raw_files:
        print(f"  [skip] {name}  (no iter*.jsonl found)")
        return

    print(f"  [archive] {name}  →  {name}_{timestamp}")
    if not dry_run:
        os.rename(subfolder, archived)

    for src in raw_files:
        dst = os.path.join(subfolder, os.path.basename(src))
        print(f"  [copy]   {os.path.basename(src)}  →  {name}/")
        if not dry_run:
            os.makedirs(subfolder, exist_ok=True)
            shutil.copy2(os.path.join(archived, os.path.basename(src)), dst)


def main():
    parser = argparse.ArgumentParser(
        description="Archive evaluated output folders and recreate clean copies for re-evaluation."
    )
    parser.add_argument("model_folder", help="Model output directory (e.g. inference/outputs/Tongyi-...)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()

    model_folder = os.path.abspath(args.model_folder)
    if not os.path.isdir(model_folder):
        print(f"Error: {model_folder} is not a directory")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dry_run:
        print(f"[DRY RUN] model_folder: {model_folder}")

    entries = sorted(
        e for e in os.listdir(model_folder)
        if os.path.isdir(os.path.join(model_folder, e))
    )

    processed = 0
    for name in entries:
        if is_archived(name):
            print(f"  [skip] {name}  (already archived)")
            continue
        reset_folder(os.path.join(model_folder, name), timestamp, args.dry_run)
        processed += 1

    if processed == 0:
        print("No eligible folders found.")
    elif args.dry_run:
        print("\n[DRY RUN] No changes made. Remove --dry-run to apply.")
    else:
        print(f"\nDone. {processed} folder(s) archived. Run temp.sh to re-evaluate.")


if __name__ == "__main__":
    main()
