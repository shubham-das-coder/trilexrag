import os
import json
from tqdm import tqdm

INPUT_ROOT = "data"
OUTPUT_FILE = "data/data.jsonl"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for root, _, files in os.walk(INPUT_ROOT):
        for file in sorted(files):
            if not file.endswith(".jsonl"):
                continue

            if os.path.abspath(os.path.join(root, file)) == os.path.abspath(OUTPUT_FILE):
                continue

            input_path = os.path.join(root, file)
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
                        "san": data.get("san", ""),
                        "hin": data.get("hin", "")
                    }

                    out_f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

                    row_id += 1

print(f"Done. Output saved to: {OUTPUT_FILE}")