import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pptx import Presentation
from pptx.util import Inches

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs("diagrams", exist_ok=True)

# -----------------------------
# COMET Scores
# -----------------------------

data = {
    "Flores+":      [41.77, 41.68, 39.90, 39.88, 39.89, 66.14, 68.03],
    "IN22-Conv":    [48.34, 49.05, 45.74, 45.78, 46.05, 67.71, 69.78],
    "IN22-Gen":     [44.18, 43.93, 42.13, 41.98, 42.10, 66.04, 67.52],
    "NIOS":         [51.03, 50.56, 48.39, 48.66, 48.43, 54.06, 56.38]
}

models = [
    "ZS",
    "R (sa-en)",
    "R (sa-en-hi)",
    "R (sa-hi)",
    "R (sa-hi-en)",
    "NLLB 1.3B",
    "NLLB 3.3B"
]

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame(data, index=models)

# -----------------------------
# Plot Heatmap
# -----------------------------

plt.figure(figsize=(12, 7))

ax = sns.heatmap(
    df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar=True,
    annot_kws={
        "size": 20,
        "weight": "normal"   # Normal annotation text
    },
    cbar_kws={
        "shrink": 0.9
    }
)

# -----------------------------
# Font settings
# -----------------------------

# Axis labels
ax.set_xlabel("Datasets", fontsize=20, fontweight='normal')
ax.set_ylabel("Models", fontsize=20, fontweight='normal')

# Tick labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=0,
    fontsize=20,
    fontweight='normal'
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=20,
    fontweight='normal'
)

# Colorbar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)

# Remove title
plt.title("")

# Tight layout
plt.tight_layout()

# -----------------------------
# Save figure
# -----------------------------

png_path = "diagrams/comet.png"
pdf_path = "diagrams/comet.pdf"
pptx_path = "diagrams/comet.pptx"

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')

# Show plot
plt.show()

# -----------------------------
# Save to PowerPoint (.pptx)
# -----------------------------

prs = Presentation()

# Blank slide layout
slide_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(slide_layout)

# Add image to slide
slide.shapes.add_picture(
    png_path,
    Inches(0.5),
    Inches(0.5),
    width=Inches(12.5)
)

# Save PPTX
prs.save(pptx_path)

print("Saved:")
print(f" - {png_path}")
print(f" - {pdf_path}")
print(f" - {pptx_path}")