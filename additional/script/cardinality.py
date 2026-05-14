import json
from collections import Counter
from pathlib import Path

folder_path = "additional/script"
output_path = "additional/script/cardinality.jsonl"

with open(output_path, "w", encoding="utf-8") as out_f:
    for path in sorted(Path(folder_path).glob("*rows.jsonl")):
        model_name = path.stem.replace("_rows", "")

        dataset_counters = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                dataset = row["file"]

                if dataset not in dataset_counters:
                    dataset_counters[dataset] = Counter()

                cardinality = row["num_scripts"]
                dataset_counters[dataset][cardinality] += 1

        for dataset, counter in sorted(dataset_counters.items()):
            output_row = {
                "model": model_name,
                "dataset": dataset,
                "cardinality_distribution": dict(sorted(counter.items()))
            }

            out_f.write(json.dumps(output_row, ensure_ascii=False) + "\n")