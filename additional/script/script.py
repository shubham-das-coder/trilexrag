import os
import json
import regex as re
from collections import defaultdict
from tqdm import tqdm

INPUT_ROOT = "/home/shubhamdas-pg/tlr/zs/qwen3-4b-instruct"

OUTPUT_ROOT = "/home/shubhamdas-pg/tlr/additional/script/qwen3-4b-instruct"

OUTPUT_STATS_FILE = os.path.join(OUTPUT_ROOT, "stats.jsonl")

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

DEVANAGARI_PATTERN = re.compile(r"\p{Script=Devanagari}")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

all_stats = []

for root, _, files in os.walk(INPUT_ROOT):
    for file in tqdm(files):
        if not file.endswith(".jsonl"):
            continue

        input_path = os.path.join(root, file)

        relative_path = os.path.relpath(root, INPUT_ROOT)

        output_dir = os.path.join(OUTPUT_ROOT, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        output_rows_path = os.path.join(output_dir, file)

        script_row_counts = defaultdict(int)
        total_hallucinated_rows = 0

        hallucinated_rows = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)
                except:
                    continue

                text = str(data.get(TARGET_COLUMN, "")).strip()

                if not text:
                    continue

                present_scripts = set()

                for script_name, pattern in SCRIPT_PATTERNS.items():
                    if pattern.search(text):
                        present_scripts.add(script_name)

                if present_scripts:
                    total_hallucinated_rows += 1

                    for script_name in present_scripts:
                        script_row_counts[script_name] += 1

                    data["detected_scripts"] = sorted(list(present_scripts))
                    data["source_file"] = file
                    data["line_number"] = line_num

                    hallucinated_rows.append(data)

        with open(output_rows_path, "w", encoding="utf-8") as out_f:
            for row in hallucinated_rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats_entry = {
            "file": os.path.relpath(input_path, INPUT_ROOT),
            "total_hallucinated_rows": total_hallucinated_rows
        }

        for script_name in sorted(SCRIPT_PATTERNS.keys()):
            stats_entry[f"{script_name}_rows"] = script_row_counts.get(script_name, 0)

        all_stats.append(stats_entry)

with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as stats_f:
    for row in all_stats:
        stats_f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Done.")