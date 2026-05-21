import json
from collections import defaultdict

# Input JSONL file
input_file = "combined/she/qwen3_4b_instruct_outputs.jsonl"

# Output JSONL file
output_file = "combined/em/she.jsonl"

# Store statistics separately for each file
total_match_count = defaultdict(int)
total_rows = defaultdict(int)
min_match_count = defaultdict(lambda: float("inf"))
max_match_count = defaultdict(lambda: float("-inf"))

# Read JSONL
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        file_name = data.get("file")
        match_count = data.get("match_count", 0)

        total_match_count[file_name] += match_count
        total_rows[file_name] += 1

        # Update min and max
        min_match_count[file_name] = min(
            min_match_count[file_name],
            match_count
        )

        max_match_count[file_name] = max(
            max_match_count[file_name],
            match_count
        )

# Prepare results
results = []

print("Statistics per file:\n")

for file_name in sorted(total_match_count.keys()):
    avg = total_match_count[file_name] / total_rows[file_name]

    result = {
        "file": file_name,
        "average_match_count": round(avg, 2),
        "min_match_count": min_match_count[file_name],
        "max_match_count": max_match_count[file_name],
        "total_rows": total_rows[file_name]
    }

    results.append(result)

    print(
        f"{file_name} | "
        f"Avg: {avg:.2f} | "
        f"Min: {min_match_count[file_name]} | "
        f"Max: {max_match_count[file_name]} | "
        f"Rows: {total_rows[file_name]}"
    )

# Save results to JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nSaved statistics to: {output_file}")