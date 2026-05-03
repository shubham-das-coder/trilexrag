from datasets import load_dataset
import pandas as pd
import json

ds_san = load_dataset("openlanguagedata/flores_plus", "san_Deva", split="devtest")
ds_hin = load_dataset("openlanguagedata/flores_plus", "hin_Deva", split="devtest")

df_san = ds_san.to_pandas()
df_hin = ds_hin.to_pandas()

assert df_san["id"].tolist() == df_hin["id"].tolist(), "IDs are not aligned!"

output_file = "/home/shubhamdas-pg/tlr/data/flores_plus.jsonl"

with open(output_file, "w", encoding="utf-8") as fout:
    for san_text, hin_text in zip(df_san["text"], df_hin["text"]):
        data = {"san": san_text, "hin": hin_text}
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Saved {output_file}")