import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

INPUT_FILE = "dict/filtered.jsonl"
OUTPUT_FILE = "dict/phi4.jsonl"

MODEL_NAME = "microsoft/phi-4"

dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map="auto"
)

model.eval()

def clean_output(text):
    text = text.strip()
    text = text.split("\n")[0]
    text = text.replace("English:", "").strip()
    return text.strip(" :,-")

def generate_english(sanskrit, hindi):
    prompt = f"""You are a strict English translator.

Task: Convert the following Sanskrit + Hindi dictionary entry into English.

STRICT RULES (MUST FOLLOW):
- Output ONLY English words
- DO NOT output any Hindi or Sanskrit characters
- DO NOT mix languages
- DO NOT include explanations
- DO NOT include slashes (/), translations, or alternatives
- Output a SINGLE concise English phrase
- Length should roughly match Hindi meaning

SELF-CHECK BEFORE OUTPUT:
If your output contains ANY non-English characters, DISCARD it and regenerate internally until it is PURE English.

INPUT:
Sanskrit: {sanskrit}
Hindi: {hindi}

FINAL OUTPUT (English only):"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return clean_output(result)


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in tqdm(infile, total=total_lines, desc="Processing"):
        data = json.loads(line)

        sanskrit = data.get("san", "").strip()
        hindi = data.get("hin", "").strip()

        try:
            eng = generate_english(sanskrit, hindi)
        except Exception as e:
            print(f"Error: {sanskrit} | {hindi} | {e}")
            eng = ""

        data["eng"] = eng

        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")