import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

STATS_FILE = "/home/shubhamdas-pg/tlr/additional/num/qwen3-4b-instruct/stats.jsonl"

OUTPUT_PNG = "/home/shubhamdas-pg/tlr/additional/num/qwen3-4b-instruct/cm.png"

os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)

with open(STATS_FILE, "r", encoding="utf-8") as f:
    stats_data = [json.loads(line) for line in f]

stats_data = sorted(stats_data, key=lambda x: x["file"])

num_plots = len(stats_data)

cols = 2
rows = int(np.ceil(num_plots / cols))

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(12, 5 * rows)
)

axes = np.array(axes).reshape(-1)

for idx, row in enumerate(stats_data):

    ax = axes[idx]

    file_name = row["file"].replace(".jsonl", "")

    matrix = np.array([
        [
            row["no_no"],
            row["no_eng"],
            row["no_hin"]
        ],
        [
            row["eng_no"],
            row["eng_eng"],
            row["eng_hin"]
        ],
        [
            row["hin_no"],
            row["hin_eng"],
            row["hin_hin"]
        ]
    ], dtype=float)

    row_sums = matrix.sum(axis=1, keepdims=True)

    normalized_matrix = np.divide(
        matrix,
        row_sums,
        where=row_sums != 0
    ) * 100

    df = pd.DataFrame(
        normalized_matrix,
        index=[
            "Source: No",
            "Source: English",
            "Source: Hindi"
        ],
        columns=[
            "Gen: No",
            "Gen: English",
            "Gen: Hindi"
        ]
    )

    sns.heatmap(
        df,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        linewidths=0.5,
        cbar=True,
        vmin=0,
        vmax=100,
        square=True,
        ax=ax,
        cbar_kws={"label": "%"}
    )

    ax.set_title(
        file_name,
        fontsize=13
    )

    ax.set_xlabel(
        "Generated Numeral Type",
        fontsize=11
    )

    ax.set_ylabel(
        "Source Numeral Type",
        fontsize=11
    )

for idx in range(num_plots, len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle(
    "Numeral Transition Confusion Matrices",
    fontsize=18,
    y=1.02
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved combined plot to: {OUTPUT_PNG}")