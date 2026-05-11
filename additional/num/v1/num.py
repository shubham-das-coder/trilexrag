import os
import json
import re
from tqdm import tqdm

INPUT_FILE = "combined/nmt/nllb200_3p3b_outputs.jsonl"

OUTPUT_STATS_FILE = "additional/num/nllb200_3p3b_stats.jsonl"

OUTPUT_ROWS_FILE = "additional/num/nllb200_3p3b_rows.jsonl"

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
                    "counts": {perm: 0 for perm in all_permutations},
                    "total_rows": 0,
                    "comparable_rows": 0,
                    "same_numbers": 0,
                    "different_numbers": 0,
                    "case_stats": {
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
                }

            stats = stats_by_file[file_name]

            san_text = row.get("san", "")
            gen_text = row.get("gen", "")

            san_type = detect_number_type(san_text)
            gen_type = detect_number_type(gen_text)

            perm = f"{san_type}_{gen_type}"

            stats["counts"][perm] += 1
            stats["total_rows"] += 1

            extra_fields = {}

            if perm != "no_no":
                extra_fields["number_type_case"] = perm

            if perm in valid_compare_cases:

                stats["comparable_rows"] += 1

                san_numbers = extract_all_numbers(san_text)
                gen_numbers = extract_all_numbers(gen_text)

                if numbers_match(san_text, gen_text):

                    stats["same_numbers"] += 1
                    stats["case_stats"][perm]["same"] += 1

                    extra_fields["same_number"] = True
                    extra_fields["san_numbers"] = san_numbers
                    extra_fields["gen_numbers"] = gen_numbers

                else:

                    stats["different_numbers"] += 1
                    stats["case_stats"][perm]["different"] += 1

                    extra_fields["same_number"] = False
                    extra_fields["san_numbers"] = san_numbers
                    extra_fields["gen_numbers"] = gen_numbers

            if len(extra_fields) > 0:

                updated_row = {
                    **row,
                    **extra_fields
                }

                rows_outfile.write(
                    json.dumps(updated_row, ensure_ascii=False) + "\n"
                )

        except Exception as e:
            print(f"Error processing line: {e}")

    for file_name, stats in stats_by_file.items():

        result = {
            "file": file_name,

            "total_rows": stats["total_rows"],

            "comparable_rows": stats["comparable_rows"],

            "same_numbers": stats["same_numbers"],

            "different_numbers": stats["different_numbers"],

            "eng_eng_same": stats["case_stats"]["eng_eng"]["same"],
            "eng_eng_different": stats["case_stats"]["eng_eng"]["different"],

            "eng_hin_same": stats["case_stats"]["eng_hin"]["same"],
            "eng_hin_different": stats["case_stats"]["eng_hin"]["different"],

            "hin_eng_same": stats["case_stats"]["hin_eng"]["same"],
            "hin_eng_different": stats["case_stats"]["hin_eng"]["different"],

            "hin_hin_same": stats["case_stats"]["hin_hin"]["same"],
            "hin_hin_different": stats["case_stats"]["hin_hin"]["different"],

            **stats["counts"]
        }

        stats_outfile.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

print(f"\nInput file:\n{INPUT_FILE}")
print(f"\nSaved stats to:\n{OUTPUT_STATS_FILE}")
print(f"Saved rows to:\n{OUTPUT_ROWS_FILE}")