import json

input_file = "dict/filtered.jsonl"
output_file = "dict/plus_removed.jsonl"

kept = 0
removed = 0

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)

        san = data.get("san", "")
        hin = data.get("hin", "")

        if "+" in san or "+" in hin:
            removed += 1
            continue

        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
        kept += 1

print(f"Rows kept: {kept}")
print(f"Rows removed: {removed}")