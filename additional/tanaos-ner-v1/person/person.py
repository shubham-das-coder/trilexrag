import json
from transformers import pipeline
from tqdm import tqdm

INPUT_FILE = "combined/zs/qwen3_4b_instruct_outputs.jsonl"
OUTPUT_FILE = "additional/person/rows.jsonl"

pipe = pipeline(
    "token-classification",
    model="tanaos/tanaos-NER-v1",
    aggregation_strategy="simple"
)

count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in tqdm(fin):
        data = json.loads(line)

        san_text = data["san"]

        entities = pipe(san_text)

        has_person = any(
            ent["entity_group"] == "PERSON"
            for ent in entities
        )

        if has_person:
            fout.write(json.dumps({
                "file": data["file"],
                "row": data["row"]
            }, ensure_ascii=False) + "\n")
            fout.flush()
            count += 1

print(f"Saved {count} rows to {OUTPUT_FILE}")