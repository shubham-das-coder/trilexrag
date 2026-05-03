from datasets import load_dataset
import json
from tqdm import tqdm

DATASET_NAME = "bhashini-nltm/NIOS_Hindi_Sanskrit"

dataset = load_dataset(DATASET_NAME, split="test")

out_file = "/home/shubhamdas-pg/tlr/data/nios.jsonl"

with open(out_file, "w", encoding="utf-8") as f:
    for row in tqdm(dataset):
        obj = {
            "san": row["Sanskrit"],
            "hin": row["Hindi"]
        }
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")