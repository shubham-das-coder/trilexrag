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
    "ai4bharat/indictrans2-indic-indic-dist-320M"
], key=str.lower)

# FLORES language codes for NLLB and IndicTrans2
LANG_CODES = {
    "san": "san_Deva",
    "hin": "hin_Deva"
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("Loading tokenizers...")
tokenizers = {
    model: AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    for model in MODEL_NAMES
}


def get_token_length(tokenizer, model_name, text, lang_key):
    """
    Calculates token length while properly handling model-specific 
    language tag requirements for IndicTrans2 and NLLB.
    """
    if not text:
        return 0

    lang_code = LANG_CODES.get(lang_key)

    # 1. Handle IndicTrans2 models
    if "indictrans2" in model_name.lower():
        # IndicTrans2 expects format: "<src_lang> <tgt_lang> <text>"
        formatted_text = f"{lang_code} {lang_code} {text}"
        tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
        # Exclude the 2 prefix language tags from the token count calculation
        return max(0, len(tokens) - 2)

    # 2. Handle NLLB-200 models
    elif "nllb" in model_name.lower():
        tokenizer.src_lang = lang_code
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    # 3. Standard tokenizers (Qwen, GPT, etc.)
    else:
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
                    line_str = line.strip()
                    if not line_str:
                        continue
                    
                    data = json.loads(line_str)

                    san_len = get_token_length(tokenizer, model_name, data.get("san", ""), "san")
                    hin_len = get_token_length(tokenizer, model_name, data.get("hin", ""), "hin")

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