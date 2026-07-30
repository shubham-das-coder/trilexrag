import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

INPUT_DIR = "data"
OUTPUT_DIR = "nmt/indictrans2-320m"
MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"
BATCH_SIZE = 4

assert torch.cuda.is_available()

DEVICE = torch.device("cuda")

SRC_LANG = "san_Deva"
TGT_LANG = "hin_Deva"

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
).to(DEVICE)

model.eval()

# Initialize IndicProcessor for text preprocessing and postprocessing
ip = IndicProcessor(inference=True)

@torch.inference_mode()
def translate_batch(san_texts):
    # IndicTrans2 preprocessing (adds lang tags and handles script normalization)
    preprocessed_texts = ip.preprocess_batch(san_texts, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    inputs = tokenizer(
        preprocessed_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=False,  # Fixes the AttributeError with past_key_values
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        max_new_tokens=2048
    )

    # Decode tokens
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    # IndicTrans2 postprocessing
    postprocessed_texts = ip.postprocess_batch(decoded, lang=TGT_LANG)

    return [t.strip() for t in postprocessed_texts]

def process_file(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    open(output_file, "a", encoding="utf-8").close()

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(lines), BATCH_SIZE), desc=f"Translating {os.path.basename(input_file)}"):
            batch_lines = lines[i : i + BATCH_SIZE]
            batch_data = [json.loads(line) for line in batch_lines]

            san_texts = [data["san"] for data in batch_data]
            translations = translate_batch(san_texts)

            for data, gen_text in zip(batch_data, translations):
                data["gen"] = gen_text
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