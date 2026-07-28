import json
from decimal import Decimal, ROUND_HALF_UP

# Input and output files
INPUT_FILE = "zs/gpt-oss-20b/scores.jsonl"
OUTPUT_FILE = "rnd.txt"


def round_2(x):
    """
    Standard academic rounding (ROUND_HALF_UP)
    commonly preferred in research reporting.
    """
    return Decimal(str(x)).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP
    )


with open(INPUT_FILE, "r", encoding="utf-8") as f, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for line in f:
        data = json.loads(line)

        # Clean dataset name
        dataset = data["file"].replace(".jsonl", "")

        # Convert to percentage
        bleu = data["bleu"] * 100
        meteor = data["meteor"] * 100
        comet = data["wmt22_comet_da"] * 100

        # Proper rounding
        bleu = round_2(bleu)
        meteor = round_2(meteor)
        comet = round_2(comet)

        # Write to txt file
        out.write(f"{dataset}\n")
        out.write(f"{bleu}\n")
        out.write(f"{meteor}\n")
        out.write(f"{comet}\n\n")

print(f"Saved formatted scores to: {OUTPUT_FILE}")