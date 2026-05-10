import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

INPUT_DIR = "/home/shubhamdas-pg/tlr/data"
OUTPUT_DIR = "/home/shubhamdas-pg/tlr/zs/nllb200-1p3b"
MODEL_NAME = "facebook/nllb-200-1.3B"

assert torch.cuda.is_available()

DEVICE = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda"
).to(DEVICE)

model.eval()

SRC_LANG = "san_Deva"
TGT_LANG = "hin_Deva"

tokenizer.src_lang = SRC_LANG

@torch.inference_mode()
def translate(san_text):

    inputs = tokenizer(
        san_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        max_new_tokens=2048
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

def process_file(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    open(output_file, "a", encoding="utf-8").close()

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "a", encoding="utf-8") as f:
        for line in tqdm(lines, desc=f"Translating {os.path.basename(input_file)}"):
            data = json.loads(line)

            san_text = data["san"]

            data["gen"] = translate(san_text)

            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()

def process():
    file_list = []

    for root, dirs, files in os.walk(INPUT_DIR):
        dirs.sort()
        files.sort()

        for file in files:
            if file.endswith(".jsonl"):
                file_list.append(os.path.join(root, file))

    file_list.sort()

    for input_file in file_list:
        rel_path = os.path.relpath(input_file, INPUT_DIR)

        output_file = os.path.join(OUTPUT_DIR, rel_path)

        process_file(input_file, output_file)

if __name__ == "__main__":
    process()