import json

FILE1 = "combined/data/data.jsonl"
FILE2 = "combined/nmt/nllb200_3p3b_outputs.jsonl"

COLUMNS = ["file", "row", "san", "hin"]

def load_jsonl(path):
    data = []

    print(f"Loading: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            obj = json.loads(line)

            missing = [col for col in COLUMNS if col not in obj]

            if missing:
                print(f"{path} | line {line_no} | missing columns: {missing}")
                return None

            data.append({col: obj[col] for col in COLUMNS})

    print(f"Loaded {len(data)} rows from: {path}")

    return data

data1 = load_jsonl(FILE1)
data2 = load_jsonl(FILE2)

if data1 is None or data2 is None:
    exit()

if len(data1) != len(data2):
    print("\nRow count mismatch")
    print(f"{FILE1}: {len(data1)}")
    print(f"{FILE2}: {len(data2)}")
    exit()

found = False

for idx, (r1, r2) in enumerate(zip(data1, data2), start=1):
    if r1 != r2:
        found = True

        print(f"\nMismatch at line {idx}")

        print(f"\nPATH1 : {FILE1}")
        print(f"file  : {r1['file']}")
        print(f"row   : {r1['row']}")
        print(f"san   : {r1['san']}")
        print(f"hin   : {r1['hin']}")

        print(f"\nPATH2 : {FILE2}")
        print(f"file  : {r2['file']}")
        print(f"row   : {r2['row']}")
        print(f"san   : {r2['san']}")
        print(f"hin   : {r2['hin']}")

if not found:
    print("\nYes")
    print(f"{FILE1}")
    print(f"{FILE2}")