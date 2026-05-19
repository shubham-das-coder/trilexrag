import json
from collections import defaultdict

DATA_FILE = "combined/data/data.jsonl"
QWEN_FILE = "combined/zs/qwen3_4b_instruct_outputs.jsonl"
NLLB13_FILE = "combined/nmt/nllb200_1p3b_outputs.jsonl"
NLLB33_FILE = "combined/nmt/nllb200_3p3b_outputs.jsonl"
OUTPUT_FILE = "combined/translations/shortest.jsonl"

# enforce dataset order
FILE_ORDER = ["flores_plus", "in22_conv", "in22_gen", "nios"]


def load_as_dict(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            key = (obj.get("file"), obj.get("row"))
            data[key] = obj
    return data


def word_count(text):
    if not text:
        return 10**9
    return len(text.strip().split())


# Load datasets
base_data = load_as_dict(DATA_FILE)
qwen_data = load_as_dict(QWEN_FILE)
nllb13_data = load_as_dict(NLLB13_FILE)
nllb33_data = load_as_dict(NLLB33_FILE)


# -----------------------------
# STEP 1: group by file
# -----------------------------
grouped = defaultdict(list)

for key, base in base_data.items():
    file_id, row_id = key
    grouped[file_id].append((key, base))


# -----------------------------
# STEP 2: select shortest 25 per dataset
# -----------------------------
selected_keys = set()

for file_id in FILE_ORDER:
    items = grouped.get(file_id, [])

    # sort by Sanskrit length, then by row
    items_sorted = sorted(
        items,
        key=lambda x: (word_count(x[1].get("san")), x[0][1])
    )

    for i in range(min(25, len(items_sorted))):
        selected_keys.add(items_sorted[i][0])


# -----------------------------
# STEP 3: sort final output by (file order, row)
# -----------------------------
def sort_key(k):
    file_id, row_id = k
    return (FILE_ORDER.index(file_id), row_id)

selected_keys = sorted(selected_keys, key=sort_key)


# -----------------------------
# STEP 4: merge only selected rows
# -----------------------------
merged = []

for key in selected_keys:
    file_id, row_id = key
    base = base_data[key]

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


# -----------------------------
# STEP 5: write output
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for obj in merged:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Saved filtered ordered file to {OUTPUT_FILE}")