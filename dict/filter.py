import json
import re

INPUT_FILE = "merged.jsonl"
OUTPUT_FILE = "filtered.jsonl"

# regex for ANY English letter
english_pattern = re.compile(r'[a-zA-Z]')

# regex to remove all types of brackets: (), [], {}, <>
bracket_pattern = re.compile(r'[\(\)\[\]\{\}<>]')

def contains_english(text):
    return bool(english_pattern.search(text))

def remove_brackets(text):
    return bracket_pattern.sub('', text)

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    removed_count = 0
    kept_count = 0

    for line in infile:
        data = json.loads(line.strip())

        san_text = data.get("san", "")
        hin_text = data.get("hin", "")

        # 🔹 remove brackets first
        san_text = remove_brackets(san_text)
        hin_text = remove_brackets(hin_text)

        # update cleaned text back
        data["san"] = san_text
        data["hin"] = hin_text

        # remove if ANY English character is present
        if contains_english(san_text) or contains_english(hin_text):
            removed_count += 1
            continue

        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        kept_count += 1

print(f"Kept: {kept_count}, Removed: {removed_count}")