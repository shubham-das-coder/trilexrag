import json

QWEN_FILE = "combined/zs/qwen3_4b_instruct_outputs.jsonl"
NLLB13_FILE = "combined/nmt/nllb200_1p3b_outputs.jsonl"
NLLB33_FILE = "combined/nmt/nllb200_3p3b_outputs.jsonl"

OUTPUT_FILE = "additional/analysis/location.jsonl"

selected_rows = {
    "flores_plus": 48,
    "in22_conv": 969,
    "in22_gen": 85,
    "nios": 1411
}


def load_jsonl(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            key = (obj["file"], obj["row"])
            data[key] = obj

    return data


qwen_data = load_jsonl(QWEN_FILE)
nllb13_data = load_jsonl(NLLB13_FILE)
nllb33_data = load_jsonl(NLLB33_FILE)

output_rows = []

for dataset_name, row_id in selected_rows.items():

    key = (dataset_name, row_id)

    if key not in qwen_data:
        print(f"Missing in Qwen: {key}")
        continue

    if key not in nllb13_data:
        print(f"Missing in NLLB1.3B: {key}")
        continue

    if key not in nllb33_data:
        print(f"Missing in NLLB3.3B: {key}")
        continue

    qwen_obj = qwen_data[key]
    nllb13_obj = nllb13_data[key]
    nllb33_obj = nllb33_data[key]

    out = {
        "file": dataset_name,
        "row": row_id,

        "san": qwen_obj["san"],
        "hin": qwen_obj["hin"],

        "qwen3_4b_instruct": qwen_obj["gen"],
        "nllb200_1p3b": nllb13_obj["gen"],
        "nllb200_3p3b": nllb33_obj["gen"]
    }

    output_rows.append(out)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in output_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(output_rows)} rows to {OUTPUT_FILE}")