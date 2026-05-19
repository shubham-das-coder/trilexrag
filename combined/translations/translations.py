import json

DATA_FILE = "combined/data/data.jsonl"
QWEN_FILE = "combined/zs/qwen3_4b_instruct_outputs.jsonl"
NLLB13_FILE = "combined/nmt/nllb200_1p3b_outputs.jsonl"
NLLB33_FILE = "combined/nmt/nllb200_3p3b_outputs.jsonl"
OUTPUT_FILE = "combined/translations/translations.jsonl"


def load_as_dict(path):
    """
    Creates a dictionary with key = (file, row)
    and value = full json object.
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            key = (obj.get("file"), obj.get("row"))
            data[key] = obj
    return data


# Load base dataset
base_data = load_as_dict(DATA_FILE)

# Load model outputs
qwen_data = load_as_dict(QWEN_FILE)
nllb13_data = load_as_dict(NLLB13_FILE)
nllb33_data = load_as_dict(NLLB33_FILE)

merged = []

for key, base in base_data.items():
    file_id, row_id = key

    merged_obj = {
        "file": file_id,
        "row": row_id,
        "san": base.get("san"),
        "hin": base.get("hin"),
        "qwen3_4b_instruct": qwen_data.get(key, {}).get("gen"),
        "nllb200_1p3b": nllb13_data.get(key, {}).get("gen"),
        "nllb200_3p3b": nllb33_data.get(key, {}).get("gen"),
    }

    merged.append(merged_obj)


# Write JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for obj in merged:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Saved merged file to {OUTPUT_FILE}")