import os
import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_float32_matmul_precision('high')

import json
import evaluate
from comet import download_model, load_from_checkpoint
from collections import defaultdict

INPUT_FILE = "combined/se/qwen3_4b_instruct_outputs.jsonl"
OUTPUT_SCORES_JSONL = "additional/rag/qwen3_4b_instruct_scores.jsonl"

os.makedirs(os.path.dirname(OUTPUT_SCORES_JSONL), exist_ok=True)

bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
meteor = evaluate.load("meteor")
bert = evaluate.load("bertscore")

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

grouped = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if all(k in obj for k in ["file", "san", "hin", "gen", "match_count"]):
            grouped[obj["file"]].append(obj)

with open(OUTPUT_SCORES_JSONL, "w", encoding="utf-8") as out_f:

    for file_name, rows in grouped.items():

        rows = sorted(rows, key=lambda x: x["row"])

        sources = [x["san"].strip() for x in rows]
        references = [x["hin"].strip() for x in rows]
        predictions = [x["gen"].strip() for x in rows]
        match_counts = [x["match_count"] for x in rows]

        min_len = min(len(sources), len(references), len(predictions))

        if min_len == 0:
            continue

        sources = sources[:min_len]
        references = references[:min_len]
        predictions = predictions[:min_len]
        match_counts = match_counts[:min_len]

        bleu_score = bleu.compute(predictions=predictions, references=references)
        chrf_score = chrf.compute(predictions=predictions, references=references)
        chrfpp_score = chrf.compute(predictions=predictions, references=references, word_order=2)
        meteor_score = meteor.compute(predictions=predictions, references=references)

        data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
        comet_output = comet_model.predict(data, batch_size=1, gpus=1)
        comet_score = sum(comet_output["scores"]) / len(comet_output["scores"])

        bert_result = bert.compute(
            predictions=predictions,
            references=references,
            model_type="xlm-roberta-large"
        )

        bert_precision = sum(bert_result["precision"]) / len(bert_result["precision"])
        bert_recall = sum(bert_result["recall"]) / len(bert_result["recall"])
        bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])

        flat_preds = [tok for sent in predictions for tok in sent.split()]
        flat_refs = [tok for sent in references for tok in sent.split()]

        matched = sum(1 for tok in flat_preds if tok in flat_refs)

        token_precision = matched / len(flat_preds) if len(flat_preds) > 0 else 0
        token_recall = matched / len(flat_refs) if len(flat_refs) > 0 else 0

        token_f1 = (
            2 * token_precision * token_recall / (token_precision + token_recall)
            if (token_precision + token_recall) > 0 else 0
        )

        match_count_distribution = dict(sorted(dict(
            (k, match_counts.count(k)) for k in set(match_counts)
        ).items()))

        avg_match_count = sum(match_counts) / len(match_counts)

        result = {
            "file": file_name,
            "num_samples": min_len,
            "avg_match_count": round(avg_match_count, 4),
            "match_count_distribution": match_count_distribution,
            "bleu": round(bleu_score["bleu"], 4),
            "chrf": round(chrf_score["score"], 4),
            "chrf++": round(chrfpp_score["score"], 4),
            "meteor": round(meteor_score["meteor"], 4),
            "wmt22_comet_da": round(comet_score, 4),
            "bert_precision": round(bert_precision, 4),
            "bert_recall": round(bert_recall, 4),
            "bert_f1": round(bert_f1, 4),
            "token_precision": round(token_precision, 4),
            "token_recall": round(token_recall, 4),
            "token_f1": round(token_f1, 4)
        }

        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(json.dumps(result, ensure_ascii=False))

print("Saved:", OUTPUT_SCORES_JSONL)