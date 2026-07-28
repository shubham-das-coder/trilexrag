import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

INPUT_DIR = "data"
OUTPUT_DIR = "os/gpt-oss-20b"
MODEL_NAME = "openai/gpt-oss-20b"

assert torch.cuda.is_available()

DEVICE = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda"
).to(DEVICE)

model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_messages(san_text):
    system_instruction = "You are a professional Sanskrit to Hindi translation system. Translate ONLY into Hindi using STRICT Devanagari script. Output ONLY final translation. No explanation. No repetition."

    one_shot_prompt = f"""Translate Sanskrit to Hindi in Devanagari script.

Example:
Sanskrit: बालकः उद्याने क्रीडति।
Hindi: बच्चा बगीचे में खेलता है।

Now translate:
Sanskrit: {san_text}
Hindi:"""

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": one_shot_prompt}
    ]

@torch.inference_mode()
def translate(san_text):
    messages = build_messages(san_text)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)

    output = model.generate(
        **inputs,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        max_new_tokens=2048,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    gen_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

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