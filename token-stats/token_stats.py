import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm

INPUT_DIR = "data"
OUTPUT_FILE = "token-stats/token_stats.jsonl"

MODEL_NAMES = sorted([
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-4B-Instruct-2507",
    "facebook/nllb-200-1.3B",
    "facebook/nllb-200-3.3B",
], key=str.lower)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

tokenizers = {
    model: AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    for model in MODEL_NAMES
}


def get_token_length(tokenizer, text):
    if text is None:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


results = []

for model_name in MODEL_NAMES:
    tokenizer = tokenizers[model_name]

    print(f"\nProcessing tokenizer: {model_name}")

    for root, _, files in os.walk(INPUT_DIR):
        for file in sorted(files):
            if not file.endswith(".jsonl"):
                continue

            file_path = os.path.join(root, file)
            dataset = os.path.relpath(file_path, INPUT_DIR).replace("\\", "/")

            max_san = 0
            max_hin = 0

            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(
                    f,
                    desc=f"{os.path.basename(model_name)} | {dataset}",
                    leave=False,
                ):
                    data = json.loads(line)

                    san_len = get_token_length(tokenizer, data.get("san", ""))
                    hin_len = get_token_length(tokenizer, data.get("hin", ""))

                    max_san = max(max_san, san_len)
                    max_hin = max(max_hin, hin_len)

            results.append({
                "model": model_name,
                "dataset": dataset,
                "max_san_tokens": max_san,
                "max_hin_tokens": max_hin,
            })

            print(
                f"{model_name} | {dataset} | SAN: {max_san} | HIN: {max_hin}"
            )

results.sort(key=lambda x: (x["model"].lower(), x["dataset"].lower()))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"\nSaved results to {OUTPUT_FILE}")