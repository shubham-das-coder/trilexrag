import os
import json
import warnings

import torch
import evaluate
from comet import download_model, load_from_checkpoint

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

INPUT_JSONL = "/content/drive/MyDrive/sanskrit/negation/qwen3_4b_instruct_outputs.jsonl"
OUTPUT_JSONL = "/content/drive/MyDrive/sanskrit/negation/qwen3_4b_instruct_scores2.jsonl"

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

sources = []
references = []
predictions = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        if "san" in obj and "hin" in obj and "gen" in obj:
            sources.append(obj["san"].strip())
            references.append(obj["hin"].strip())
            predictions.append(obj["gen"].strip())

min_len = min(len(sources), len(references), len(predictions))

sources = sources[:min_len]
references = references[:min_len]
predictions = predictions[:min_len]

if min_len == 0:
    raise ValueError("No valid samples found.")

bleu_score = bleu.compute(
    predictions=predictions,
    references=[[r] for r in references]
)

meteor_score = meteor.compute(
    predictions=predictions,
    references=references
)

data = [
    {"src": s, "mt": p, "ref": r}
    for s, p, r in zip(sources, predictions, references)
]

comet_output = comet_model.predict(
    data,
    batch_size=1,
    gpus=1
)

comet_score = sum(comet_output["scores"]) / len(comet_output["scores"])

result = {
    "file": os.path.basename(INPUT_JSONL),
    "num_samples": min_len,
    "bleu": bleu_score["bleu"],
    "meteor": meteor_score["meteor"],
    "wmt22_comet_da": comet_score
}

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(json.dumps(result, ensure_ascii=False, indent=2))
print("Saved:", OUTPUT_JSONL)