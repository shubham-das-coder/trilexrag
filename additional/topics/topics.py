import json
from pathlib import Path
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer

input_files = [
    "combined/zs/qwen3_4b_instruct_outputs.jsonl",
    "combined/nmt/nllb200_1p3b_outputs.jsonl",
    "combined/nmt/nllb200_3p3b_outputs.jsonl"
]

output_dir = "additional/topics"

Path(output_dir).mkdir(
    parents=True,
    exist_ok=True
)

domains = {
    "medical": (
        "अस्पताल डॉक्टर बीमारी दवा रोग उपचार स्वास्थ्य "
        "मरीज संक्रमण वैक्सीन चिकित्सा"
    ),

    "finance": (
        "बैंक पैसा निवेश कर बीमा ऋण अर्थव्यवस्था "
        "व्यापार भुगतान बाजार कंपनी उद्योग"
    ),

    "technology": (
        "विज्ञान अनुसंधान प्रयोगशाला अंतरिक्ष भौतिकी "
        "रसायन जीवविज्ञान कंप्यूटर इंटरनेट सॉफ्टवेयर "
        "तकनीक डेटा एआई रोबोट"
    )
}

threshold = 0.50
batch_size = 1

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU not available."
    )

device = "cuda"

torch.set_grad_enabled(False)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)

model.eval()

domain_names = list(domains.keys())

domain_texts = list(domains.values())

domain_embeddings = model.encode(
    domain_texts,
    batch_size=len(domain_texts),
    convert_to_tensor=True,
    normalize_embeddings=True,
    device=device,
    show_progress_bar=False
)

merged_rows = {}

model_names = []

for input_file in input_files:

    input_path = Path(input_file)

    model_name = input_path.stem.replace(
        "_outputs",
        ""
    )

    model_names.append(model_name)

    print(f"Processing: {model_name}")

    rows = []

    texts = []

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            row = json.loads(line)

            text = row.get(
                "hin",
                ""
            ).strip()

            if not text:
                continue

            rows.append(row)

            texts.append(text)

    text_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
        show_progress_bar=True
    )

    similarity_matrix = (
        text_embeddings @ domain_embeddings.T
    ).cpu()

    for row, similarities in zip(
        rows,
        similarity_matrix
    ):

        similarities = similarities.numpy()

        best_idx = int(
            similarities.argmax()
        )

        best_score = float(
            similarities[best_idx]
        )

        if best_score < threshold:
            continue

        domain = domain_names[best_idx]

        key = (
            row["file"],
            row["row"]
        )

        if key not in merged_rows:

            merged_rows[key] = {
                "file": row["file"],
                "row": row["row"],
                "san": row["san"],
                "hin": row["hin"],
                "domain": domain
            }

        merged_rows[key][model_name] = row.get(
            "gen",
            ""
        )

categorized_rows = defaultdict(list)

for row in merged_rows.values():

    domain = row["domain"]

    ordered_row = {
        "file": row["file"],
        "row": row["row"],
        "san": row["san"],
        "hin": row["hin"],
        "domain": row["domain"]
    }

    for model_name in model_names:

        ordered_row[model_name] = row.get(
            model_name,
            ""
        )

    categorized_rows[domain].append(
        ordered_row
    )

for domain, rows in categorized_rows.items():

    output_file = (
        Path(output_dir) /
        f"{domain}.jsonl"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"Saved: {output_file}")

print("Done.")