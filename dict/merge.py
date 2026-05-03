import os

BASE_DIR = "/home/shubhamdas-pg/scraping/sa-dot-wiktionary-org"   # change this
OUTPUT_FILE = os.path.join(BASE_DIR, "merged.jsonl")

input_files = [os.path.join(BASE_DIR, f"{i}.jsonl") for i in range(1, 9)]

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for file in input_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        outfile.write(line + "\n")

print(f"Merged file saved at: {OUTPUT_FILE}")