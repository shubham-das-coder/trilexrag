import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

folder_path = "additional/script"
output_path = "additional/script/script_percentage.jsonl"

forbidden_keys = {
    "file",
    "total",
    "bengali",
    "latin",
    "urdu_arabic"
}

with open(output_path, "w", encoding="utf-8") as out_f:
    for path in sorted(Path(folder_path).glob("*stats.jsonl")):
        model_name = path.stem.replace("_stats", "")

        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        for row in rows:
            total = Decimal(row["total"])

            bengali = Decimal(row["bengali"])
            latin = Decimal(row["latin"])
            arabic = Decimal(row["urdu_arabic"])

            others = sum(
                Decimal(v)
                for k, v in row.items()
                if k not in forbidden_keys
            )

            def pct(x):
                return float(
                    (x * Decimal(100) / total).quantize(
                        Decimal("0.01"),
                        rounding=ROUND_HALF_UP
                    )
                )

            output_row = {
                "model": model_name,
                "file": row["file"],
                "total": int(total),
                "bengali": pct(bengali),
                "latin": pct(latin),
                "urdu_arabic": pct(arabic),
                "others": pct(others)
            }

            out_f.write(json.dumps(output_row, ensure_ascii=False) + "\n")