import json

QWEN_FILE = "combined/zs/qwen3_4b_instruct_outputs.jsonl"
NLLB13_FILE = "combined/nmt/nllb200_1p3b_outputs.jsonl"
NLLB33_FILE = "combined/nmt/nllb200_3p3b_outputs.jsonl"

OUTPUT_FILE = "additional/analysis/analysis.jsonl"

categories = {
    "1_negation_preservation": {
        "flores_plus": [1, 2, 3, 4, 5],
        "in22_conv": [2, 13, 15, 16, 17],
        "in22_gen": [1, 5, 6, 9, 12],
        "nios": [8, 13, 14, 21, 22]
    },

    "2_long_complex_sentences": {
        "flores_plus": [4, 341, 549, 162, 692],
        "in22_conv": [1067, 248, 43, 1158, 238],
        "in22_gen": [792, 826, 523, 582, 884],
        "nios": [1561, 977, 1241, 932, 965]
    },

    "3_compound_morphology": {
        "flores_plus": [162, 341, 404, 549, 692],
        "in22_conv": [248, 431, 742, 1067, 1158],
        "in22_gen": [411, 523, 582, 792, 884],
        "nios": [932, 977, 1241, 1561, 1706]
    },

    "4_named_entity_transliteration": {
        "flores_plus": [1, 2, 6, 8, 11],
        "in22_conv": [4, 25, 44, 91, 130],
        "in22_gen": [7, 18, 35, 140, 264],
        "nios": [89, 244, 811, 1310, 1704]
    }
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

for category, datasets in categories.items():

    for dataset_name, rows in datasets.items():

        for row_id in rows:

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
                "type": category,

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