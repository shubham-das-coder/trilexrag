from datasets import load_dataset
import json
from tqdm import tqdm

DATASET_NAME = "ai4bharat/IN22-Gen"

dataset = load_dataset(DATASET_NAME)

out_file = "/home/shubhamdas-pg/tlr/data/in22_gen.jsonl"

with open(out_file, "w", encoding="utf-8") as f:
    for split in dataset.keys():
        for row in tqdm(dataset[split]):
            obj = {
                "san": row["san_Deva"],
                "hin": row["hin_Deva"]
            }
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")