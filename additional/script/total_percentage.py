import os
import json
from decimal import Decimal, ROUND_HALF_UP

INPUT_FOLDER = "additional/script"
OUTPUT_FILE = "additional/script/total_percentage.jsonl"

DATASET_TOTALS = {
    "flores_plus": 1012,
    "in22_conv": 1503,
    "in22_gen": 1024,
    "nios": 1743
}

def emnlp_round(value):
    return format(
        Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        ".2f"
    )

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for filename in sorted(os.listdir(INPUT_FOLDER)):

        if not filename.endswith("_stats.jsonl"):
            continue

        input_path = os.path.join(INPUT_FOLDER, filename)

        model_name = filename.replace("_stats.jsonl", "")

        with open(input_path, "r", encoding="utf-8") as fin:

            for line in fin:
                row = json.loads(line)

                dataset_total = DATASET_TOTALS[row["file"]]
                hallucinated_total = row["total"]

                output = {
                    "model": model_name,
                    "file": row["file"],
                    "hallucinated_rows": hallucinated_total,
                    "dataset_total": dataset_total,
                    "percentage": emnlp_round(
                        (hallucinated_total / dataset_total) * 100
                    )
                }

                fout.write(json.dumps(output, ensure_ascii=False) + "\n")

print(f"Saved percentage jsonl to: {OUTPUT_FILE}")