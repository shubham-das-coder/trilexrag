import os
import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_float32_matmul_precision('high')

import json
import evaluate
from comet import download_model, load_from_checkpoint

INPUT_FOLDER = "/home/shubhamdas-pg/tlr/zs/qwen3-4b"
OUTPUT_SCORES_JSONL = os.path.join(INPUT_FOLDER, "scores.jsonl")

os.makedirs(os.path.dirname(OUTPUT_SCORES_JSONL), exist_ok=True)

if not os.path.exists(OUTPUT_SCORES_JSONL):
    open(OUTPUT_SCORES_JSONL, "w").close()

bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
meteor = evaluate.load("meteor")
bert = evaluate.load("bertscore")

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

for file_name in os.listdir(INPUT_FOLDER):
    if not file_name.endswith(".jsonl"):
        continue

    file_path = os.path.join(INPUT_FOLDER, file_name)

    sources = []
    references = []
    predictions = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "san" in obj and "hin" in obj and "gen" in obj:
                sources.append(obj["san"].strip())
                references.append(obj["hin"].strip())
                predictions.append(obj["gen"].strip())

    min_len = min(len(sources), len(references), len(predictions))

    if min_len == 0:
        continue

    sources = sources[:min_len]
    references = references[:min_len]
    predictions = predictions[:min_len]

    bleu_score = bleu.compute(predictions=predictions, references=references)
    chrf_score = chrf.compute(predictions=predictions, references=references)
    chrfpp_score = chrf.compute(predictions=predictions, references=references, word_order=2)
    meteor_score = meteor.compute(predictions=predictions, references=references)

    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
    comet_output = comet_model.predict(data, batch_size=1, gpus=1)
    comet_score = sum(comet_output["scores"]) / len(comet_output["scores"])

    bert_result = bert.compute(predictions=predictions, references=references, model_type="xlm-roberta-large")
    bert_precision = sum(bert_result["precision"]) / len(bert_result["precision"])
    bert_recall = sum(bert_result["recall"]) / len(bert_result["recall"])
    bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])

    flat_preds = [tok for sent in predictions for tok in sent.split()]
    flat_refs = [tok for sent in references for tok in sent.split()]

    matched = sum(1 for tok in flat_preds if tok in flat_refs)

    token_precision = matched / len(flat_preds) if len(flat_preds) > 0 else 0
    token_recall = matched / len(flat_refs) if len(flat_refs) > 0 else 0
    token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall) if (token_precision + token_recall) > 0 else 0

    result = {
        "file": file_name,
        "num_samples": min_len,
        "bleu": bleu_score["bleu"],
        "chrf": chrf_score["score"],
        "chrf++": chrfpp_score["score"],
        "meteor": meteor_score["meteor"],
        "wmt22_comet_da": comet_score,
        "bert_precision": bert_precision,
        "bert_recall": bert_recall,
        "bert_f1": bert_f1,
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": token_f1
    }

    with open(OUTPUT_SCORES_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(result, ensure_ascii=False))

print("Saved: ", OUTPUT_SCORES_JSONL)