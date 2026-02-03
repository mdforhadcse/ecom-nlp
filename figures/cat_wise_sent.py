import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# PATH = "../data/processed/dataset_final_selected_processed.csv"
PATH = "../gold/dataset_1k_compound_small_aspects.csv"
df = pd.read_csv(PATH)

cat_col = "Product Category"
sent_col = "absa_overall_sentiment"
top_k = 10

cats = df[cat_col].astype(str).str.strip()
sents = df[sent_col].astype(str).str.strip()

mask = (
    (cats != "")
    & (cats.str.lower() != "nan")
    & (sents != "")
    & (sents.str.lower() != "nan")
)
df2 = pd.DataFrame({cat_col: cats[mask], sent_col: sents[mask]})

 # ---- Build Category × Sentiment table ----
# Keep top categories by overall frequency
cat_counts = df2[cat_col].value_counts()
top_cats = cat_counts.head(top_k).index

sub = df2[df2[cat_col].isin(top_cats)].copy()

# Cross-tab counts
pivot = pd.crosstab(sub[cat_col], sub[sent_col])

# Force a consistent sentiment order (edit if your labels differ)
sent_order = ["Negative", "Neutral", "Positive"]
pivot = pivot.reindex(columns=sent_order, fill_value=0)

# Optional: show percentages instead of raw counts
# options: None, "row" (row-wise %), "overall" (overall %)
normalize = "row"

if normalize == "row":
    data = (pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100).fillna(0)
    value_fmt = "{:.0f}%"
    cbar_label = "Row-wise %"
    title = f"Top {top_k} Categories × Sentiment (Row-wise %)"
elif normalize == "overall":
    total = pivot.to_numpy().sum()
    data = (pivot / total * 100) if total else pivot.astype(float)
    value_fmt = "{:.1f}%"
    cbar_label = "Overall %"
    title = f"Top {top_k} Categories × Sentiment (Overall %)"
else:
    data = pivot
    value_fmt = "{}"
    cbar_label = "Count"
    title = f"Top {top_k} Categories × Sentiment (Counts)"

# ---- Plot heatmap (matplotlib only) ----
fig, ax = plt.subplots(figsize=(12, 6))

im = ax.imshow(data.values, aspect="auto")

# Axis labels
ax.set_title(title)
ax.set_xlabel("Sentiment")
ax.set_ylabel("Product Category")

# Ticks and tick labels
ax.set_xticks(np.arange(data.shape[1]))
ax.set_xticklabels(data.columns, rotation=0)
ax.set_yticks(np.arange(data.shape[0]))
ax.set_yticklabels(data.index)

# Add annotations in each cell
# (Skip text if the cell is 0 to reduce clutter)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data.iat[i, j]
        if val == 0:
            continue
        ax.text(j, i, value_fmt.format(int(val) if value_fmt == "{}" else val),
                ha="center", va="center", fontsize=9)

# Light gridlines between cells (improves readability)
ax.set_xticks(np.arange(-.5, data.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, data.shape[0], 1), minor=True)
ax.grid(which="minor", linestyle="-", linewidth=0.6)
ax.tick_params(which="minor", bottom=False, left=False)

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(cbar_label)

plt.tight_layout()
plt.show()