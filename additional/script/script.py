import os
import json
import regex as re
from collections import defaultdict
from tqdm import tqdm

INPUT_FILE = "combined/nmt/indictrans2_1b_outputs.jsonl"

OUTPUT_BASE = "additional/script/indictrans2_1b"

OUTPUT_ROWS_FILE = f"{OUTPUT_BASE}_rows.jsonl"
OUTPUT_STATS_FILE = f"{OUTPUT_BASE}_stats.jsonl"

TARGET_COLUMN = "gen"

SCRIPT_PATTERNS = {
    "latin": re.compile(r"\p{Script=Latin}"),
    "bengali": re.compile(r"\p{Script=Bengali}"),
    "gujarati": re.compile(r"\p{Script=Gujarati}"),
    "gurmukhi": re.compile(r"\p{Script=Gurmukhi}"),
    "oriya": re.compile(r"\p{Script=Oriya}"),
    "tamil": re.compile(r"\p{Script=Tamil}"),
    "telugu": re.compile(r"\p{Script=Telugu}"),
    "kannada": re.compile(r"\p{Script=Kannada}"),
    "malayalam": re.compile(r"\p{Script=Malayalam}"),
    "urdu_arabic": re.compile(r"\p{Script=Arabic}"),
    "thai": re.compile(r"\p{Script=Thai}"),
    "cyrillic": re.compile(r"\p{Script=Cyrillic}"),
    "han": re.compile(r"\p{Script=Han}"),
    "hiragana": re.compile(r"\p{Script=Hiragana}"),
    "katakana": re.compile(r"\p{Script=Katakana}")
}

os.makedirs(os.path.dirname(OUTPUT_ROWS_FILE), exist_ok=True)

dataset_script_counts = defaultdict(lambda: defaultdict(int))
dataset_total_counts = defaultdict(int)

hallucinated_rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        line = line.strip()

        if not line:
            continue

        try:
            data = json.loads(line)
        except:
            continue

        dataset_name = str(data.get("file", "")).strip()

        if not dataset_name:
            continue

        text = str(data.get(TARGET_COLUMN, "")).strip()

        if not text:
            continue

        present_scripts = set()

        for script_name, pattern in SCRIPT_PATTERNS.items():
            if pattern.search(text):
                present_scripts.add(script_name)

        if not present_scripts:
            continue

        dataset_total_counts[dataset_name] += 1

        for script_name in present_scripts:
            dataset_script_counts[dataset_name][script_name] += 1

        row_data = dict(data)

        row_data["scripts"] = sorted(list(present_scripts))
        row_data["num_scripts"] = len(present_scripts)

        hallucinated_rows.append(row_data)

with open(OUTPUT_ROWS_FILE, "w", encoding="utf-8") as out_f:
    for row in hallucinated_rows:
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as stats_f:
    for dataset_name in sorted(dataset_total_counts.keys()):
        stats_entry = {
            "file": dataset_name,
            "total": dataset_total_counts[dataset_name]
        }

        for script_name in sorted(SCRIPT_PATTERNS.keys()):
            stats_entry[script_name] = dataset_script_counts[dataset_name].get(script_name, 0)

        stats_f.write(json.dumps(stats_entry, ensure_ascii=False) + "\n")

print("Done.")