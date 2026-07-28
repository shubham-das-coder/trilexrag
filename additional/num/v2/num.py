import os
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from tqdm import tqdm

INPUT_FILE = "combined/zs/gpt_oss_20b_outputs.jsonl"

OUTPUT_STATS_FILE = "additional/num/v2/gpt_oss_20b_stats.jsonl"

OUTPUT_ROWS_FILE = "additional/num/v2/gpt_oss_20b_rows.jsonl"

os.makedirs(os.path.dirname(OUTPUT_STATS_FILE), exist_ok=True)

devanagari_to_latin_digits = str.maketrans(
    "०१२३४५६७८९",
    "0123456789"
)

def normalize_digits(text):

    return str(text).translate(devanagari_to_latin_digits)

def extract_all_numbers(text):

    normalized_text = normalize_digits(text)

    return re.findall(r"\d+", normalized_text)

def has_numbers(text):

    return len(extract_all_numbers(text)) > 0

def numbers_match(san_text, gen_text):

    san_numbers = extract_all_numbers(san_text)
    gen_numbers = extract_all_numbers(gen_text)

    return san_numbers == gen_numbers

def emnlp_round(value):

    return format(
        Decimal(str(value)).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP
        ),
        ".2f"
    )

stats_by_file = {}

with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as stats_outfile, \
     open(OUTPUT_ROWS_FILE, "w", encoding="utf-8") as rows_outfile, \
     open(INPUT_FILE, "r", encoding="utf-8") as infile:

    print(f"\nProcessing: {INPUT_FILE}")

    for line in tqdm(infile, desc=os.path.basename(INPUT_FILE)):

        try:
            row = json.loads(line)

            file_name = row.get("file", "unknown")

            if file_name not in stats_by_file:

                stats_by_file[file_name] = {
                    "total_rows": 0,
                    "rows_with_numbers": 0,
                    "same_numbers": 0,
                    "different_numbers": 0
                }

            stats = stats_by_file[file_name]

            stats["total_rows"] += 1

            san_text = row.get("san", "")
            gen_text = row.get("gen", "")

            if not has_numbers(san_text):
                continue

            stats["rows_with_numbers"] += 1

            san_numbers = extract_all_numbers(san_text)
            gen_numbers = extract_all_numbers(gen_text)

            same_number = numbers_match(san_text, gen_text)

            if same_number:
                stats["same_numbers"] += 1
            else:
                stats["different_numbers"] += 1

            updated_row = {
                **row,
                "same_number": same_number,
                "san_numbers": san_numbers,
                "gen_numbers": gen_numbers
            }

            rows_outfile.write(
                json.dumps(updated_row, ensure_ascii=False) + "\n"
            )

        except Exception as e:
            print(f"Error processing line: {e}")

    for file_name, stats in stats_by_file.items():

        rows_with_numbers = stats["rows_with_numbers"]

        if rows_with_numbers > 0:
            numeral_accuracy = (
                stats["same_numbers"] / rows_with_numbers
            ) * 100
        else:
            numeral_accuracy = 0.0

        result = {
            "file": file_name,

            "total_rows": stats["total_rows"],

            "rows_with_numbers": rows_with_numbers,

            "same_numbers": stats["same_numbers"],

            "different_numbers": stats["different_numbers"],

            "numeral_accuracy": emnlp_round(numeral_accuracy)
        }

        stats_outfile.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

print(f"\nInput file:\n{INPUT_FILE}")
print(f"\nSaved stats to:\n{OUTPUT_STATS_FILE}")
print(f"Saved rows to:\n{OUTPUT_ROWS_FILE}")