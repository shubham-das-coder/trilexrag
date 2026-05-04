import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm

INPUT_DIR = "/home/shubhamdas-pg/tlr/data"
OUTPUT_FILE = "/home/shubhamdas-pg/tlr/token-stats/token_stats.jsonl"
MODEL_NAME = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

def get_token_length(text):
    if text is None:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)

                max_san = 0
                max_hin = 0

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in tqdm(f, desc=f"Processing {file}", leave=False):
                        data = json.loads(line.strip())

                        san_text = data.get("san", "")
                        hin_text = data.get("hin", "")

                        san_len = get_token_length(san_text)
                        hin_len = get_token_length(hin_text)

                        if san_len > max_san:
                            max_san = san_len

                        if hin_len > max_hin:
                            max_hin = hin_len

                result = {
                    "file": file,
                    "max_san_tokens": max_san,
                    "max_hin_tokens": max_hin
                }

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

                print(f"{file} | SAN: {max_san} | HIN: {max_hin}")