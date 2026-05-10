import os
import json
import re
from tqdm import tqdm

INPUT_ROOT = "/home/shubhamdas-pg/tlr/zs/qwen3-4b-instruct"

OUTPUT_STATS_FILE = "/home/shubhamdas-pg/tlr/additional/qwen3-4b-instruct/stats.jsonl"

OUTPUT_ROWS_FILE = "/home/shubhamdas-pg/tlr/additional/qwen3-4b-instruct/rows.jsonl"

os.makedirs(os.path.dirname(OUTPUT_STATS_FILE), exist_ok=True)

english_digit_pattern = re.compile(r"[0-9]")
hindi_digit_pattern = re.compile(r"[०-९]")

def detect_number_type(text):

    text = str(text)

    has_eng = bool(english_digit_pattern.search(text))
    has_hin = bool(hindi_digit_pattern.search(text))

    if has_hin:
        return "hin"
    elif has_eng:
        return "eng"
    else:
        return "no"

devanagari_to_latin_digits = str.maketrans(
    "०१२३४५६७८९",
    "0123456789"
)

def extract_all_numbers(text):

    text = str(text)

    normalized_text = text.translate(devanagari_to_latin_digits)

    numbers = re.findall(r"\d+", normalized_text)

    return numbers

def numbers_match(san_text, gen_text):

    san_numbers = extract_all_numbers(san_text)
    gen_numbers = extract_all_numbers(gen_text)

    return san_numbers == gen_numbers

all_permutations = [
    "no_no",
    "no_eng",
    "no_hin",
    "eng_no",
    "eng_eng",
    "eng_hin",
    "hin_no",
    "hin_eng",
    "hin_hin",
]

valid_compare_cases = {
    "eng_eng",
    "eng_hin",
    "hin_eng",
    "hin_hin"
}

with open(OUTPUT_STATS_FILE, "w", encoding="utf-8") as stats_outfile, \
     open(OUTPUT_ROWS_FILE, "w", encoding="utf-8") as rows_outfile:

    for root, dirs, files in os.walk(INPUT_ROOT):

        for file in files:

            if not file.endswith(".jsonl"):
                continue

            input_file_path = os.path.join(root, file)

            print(f"\nProcessing: {input_file_path}")

            counts = {perm: 0 for perm in all_permutations}

            total_rows = 0

            comparable_rows = 0
            same_numbers = 0
            different_numbers = 0

            case_stats = {
                "eng_eng": {
                    "same": 0,
                    "different": 0
                },
                "eng_hin": {
                    "same": 0,
                    "different": 0
                },
                "hin_eng": {
                    "same": 0,
                    "different": 0
                },
                "hin_hin": {
                    "same": 0,
                    "different": 0
                }
            }

            relative_file_name = os.path.relpath(
                input_file_path,
                INPUT_ROOT
            )

            with open(input_file_path, "r", encoding="utf-8") as infile:

                for line in tqdm(infile, desc=file):

                    try:
                        row = json.loads(line)

                        san_text = row.get("san", "")
                        gen_text = row.get("gen", "")

                        san_type = detect_number_type(san_text)
                        gen_type = detect_number_type(gen_text)

                        perm = f"{san_type}_{gen_type}"

                        counts[perm] += 1
                        total_rows += 1

                        extra_fields = {}

                        if perm != "no_no":
                            extra_fields["number_type_case"] = perm

                        if perm in valid_compare_cases:

                            comparable_rows += 1

                            if numbers_match(san_text, gen_text):

                                same_numbers += 1
                                case_stats[perm]["same"] += 1

                                extra_fields["same_number"] = True
                                extra_fields["san_numbers"] = extract_all_numbers(san_text)
                                extra_fields["gen_numbers"] = extract_all_numbers(gen_text)

                            else:

                                different_numbers += 1
                                case_stats[perm]["different"] += 1

                                extra_fields["same_number"] = False
                                extra_fields["san_numbers"] = extract_all_numbers(san_text)
                                extra_fields["gen_numbers"] = extract_all_numbers(gen_text)

                        if len(extra_fields) > 0:

                            updated_row = {
                                "file": relative_file_name,
                                **row,
                                **extra_fields
                            }

                            rows_outfile.write(
                                json.dumps(updated_row, ensure_ascii=False) + "\n"
                            )

                    except Exception as e:
                        print(f"Error processing line: {e}")

            result = {
                "file": relative_file_name,

                "total_rows": total_rows,

                "comparable_rows": comparable_rows,

                "same_numbers": same_numbers,

                "different_numbers": different_numbers,

                "eng_eng_same": case_stats["eng_eng"]["same"],
                "eng_eng_different": case_stats["eng_eng"]["different"],

                "eng_hin_same": case_stats["eng_hin"]["same"],
                "eng_hin_different": case_stats["eng_hin"]["different"],

                "hin_eng_same": case_stats["hin_eng"]["same"],
                "hin_eng_different": case_stats["hin_eng"]["different"],

                "hin_hin_same": case_stats["hin_hin"]["same"],
                "hin_hin_different": case_stats["hin_hin"]["different"],

                **counts
            }

            stats_outfile.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

print(f"\nSaved stats to:\n{OUTPUT_STATS_FILE}")
print(f"Saved rows to:\n{OUTPUT_ROWS_FILE}")