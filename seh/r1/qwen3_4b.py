import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

INPUT_DIR = "/home/shubhamdas-pg/tlr/data"
OUTPUT_DIR = "/home/shubhamdas-pg/tlr/seh/r1/qwen3-4b"
MODEL_NAME = "Qwen/Qwen3-4B"
DICT_FILE = "/home/shubhamdas-pg/tlr/dict/phi4.jsonl"

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

def load_dictionary(dict_file):
    dictionary = []
    with open(dict_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            dictionary.append(entry)
    return dictionary

DICTIONARY = load_dictionary(DICT_FILE)

def tokenize_sanskrit(text):
    return re.findall(r'[\u0900-\u097F]+', text)

def get_exact_matches(san_text):
    matches = []
    words = set(tokenize_sanskrit(san_text))
    for entry in DICTIONARY:
        if entry["san"] in words:
            matches.append(entry)
    return matches

def build_rag_context(matches):
    if not matches:
        return ""
    context_lines = []
    for entry in matches:
        context_lines.append(f'Sanskrit: {entry["san"]}, English: {entry.get("eng","")}, Hindi: {entry["hin"]}')
    context_text = "\n".join(context_lines)
    return context_text

def build_messages(san_text):
    matches = get_exact_matches(san_text)
    rag_context = build_rag_context(matches)

    system_instruction = "You are a professional Sanskrit to Hindi translation system. Translate ONLY into Hindi using STRICT Devanagari script. Output ONLY final translation. No explanation. No repetition."

    user_instruction = f"""Translate the following Sanskrit text into Hindi.

You are given relevant dictionary entries. Use these entries when they are applicable to the input. Prefer meanings corresponding to the specified target language for matched Sanskrit words when available. Ensure that the final translation is fluent, consistent, and entirely in the target language. Do not copy dictionary entries verbatim without considering context.

Sanskrit Text:
{san_text}

Relevant Dictionary Entries:
{rag_context}"""

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction}
    ], matches

@torch.inference_mode()
def translate(san_text):
    messages, matches = build_messages(san_text)

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
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return gen_text, len(matches), matches

def process_file(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "w", encoding="utf-8") as f:
        for line in tqdm(lines, desc=f"Translating {os.path.basename(input_file)}"):
            data = json.loads(line)
            san_text = data["san"]
            gen_text, match_count, match_lines = translate(san_text)
            data["gen"] = gen_text
            data["match_count"] = match_count
            data["matches"] = match_lines
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