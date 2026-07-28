import os
import json
from tqdm import tqdm

INPUT_ROOT = "she/r1/gpt-oss-20b"

OUTPUT_BASE = "combined/she/gpt_oss_20b"

OUTPUT_OUTPUTS = f"{OUTPUT_BASE}_outputs.jsonl"
OUTPUT_SCORES = f"{OUTPUT_BASE}_scores.jsonl"

os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)

with open(OUTPUT_OUTPUTS, "w", encoding="utf-8") as out_f:

    for root, _, files in os.walk(INPUT_ROOT):

        for file in sorted(files):

            if not file.endswith(".jsonl"):
                continue

            if file == "scores.jsonl":
                continue

            input_path = os.path.join(root, file)

            if os.path.abspath(input_path) == os.path.abspath(OUTPUT_OUTPUTS):
                continue

            dataset_name = os.path.splitext(file)[0]

            row_id = 1

            with open(input_path, "r", encoding="utf-8") as in_f:

                for line in tqdm(in_f, desc=dataset_name):

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    output_data = {
                        "file": dataset_name,
                        "row": row_id,
                        **data
                    }

                    out_f.write(
                        json.dumps(output_data, ensure_ascii=False) + "\n"
                    )

                    row_id += 1

with open(OUTPUT_SCORES, "w", encoding="utf-8") as score_out_f:

    for root, _, files in os.walk(INPUT_ROOT):

        for file in sorted(files):

            if file != "scores.jsonl":
                continue

            input_path = os.path.join(root, file)

            with open(input_path, "r", encoding="utf-8") as score_in_f:

                for line in tqdm(score_in_f, desc=f"scores: {root}"):

                    line = line.strip()

                    if not line:
                        continue

                    score_out_f.write(line + "\n")

print(f"Done.")
print(f"Outputs saved to: {OUTPUT_OUTPUTS}")
print(f"Scores saved to: {OUTPUT_SCORES}")