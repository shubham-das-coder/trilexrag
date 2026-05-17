import json
import re
from pathlib import Path

input_files = [
    "combined/zs/qwen3_4b_instruct_outputs.jsonl",
    "combined/nmt/nllb200_1p3b_outputs.jsonl",
    "combined/nmt/nllb200_3p3b_outputs.jsonl"
]

output_dir = Path("additional/neg")
output_dir.mkdir(parents=True, exist_ok=True)

NEG_WORDS = {
    "न",
    "नहि",
    "नहिं",
    "नो",
    "नौ",
    "नैव",
    "मा",
    "अलम्",
    "नास्ति",
    "नेति",
    "ननु",
    "नूनम्",
    "नोहि",
    "नकिञ्चित्",
    "अकस्मात्",
    "विना",
    "ऋते"
}

def normalize_token(token):
    return re.sub(
        r"^[।॥,.!?;:\"'()\[\]{}]+|[।॥,.!?;:\"'()\[\]{}]+$",
        "",
        token
    )

for input_file in input_files:
    input_path = Path(input_file)
    output_path = output_dir / input_path.name

    matched_rows = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            san_text = data.get("san", "")

            tokens = [
                normalize_token(token)
                for token in san_text.split()
            ]

            matched_negations = [
                token for token in tokens
                if token in NEG_WORDS
            ]

            if matched_negations:
                data["cnt"] = len(matched_negations)
                data["words"] = matched_negations

                outfile.write(
                    json.dumps(data, ensure_ascii=False) + "\n"
                )

                matched_rows += 1

    print(f"Saved: {output_path} | Rows matched: {matched_rows}")