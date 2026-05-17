import json
import os

from openai import OpenAI
from tqdm import tqdm

INPUT_DIR = "/home/shubhamdas-pg/tlr/data"
OUTPUT_DIR = "/home/shubhamdas-pg/tlr/zs/qwen3.6-27b"

MODEL_NAME = "Qwen/Qwen3.6-27B"

client = OpenAI()

def build_messages(san_text):
    system_instruction = (
        "You are a professional Sanskrit to Hindi translation system. "
        "Translate ONLY into Hindi using STRICT Devanagari script. "
        "Output ONLY final translation. No explanation. No repetition."
    )

    user_instruction = (
        f"Translate the following Sanskrit text into Hindi:\n\n{san_text}"
    )

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_instruction}
    ]

def translate(san_text):
    messages = build_messages(san_text)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        extra_body={
            "top_k": 1,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "chat_template_kwargs": {
                "preserve_thinking": False
            }
        }
    )

    return response.choices[0].message.content.strip()

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