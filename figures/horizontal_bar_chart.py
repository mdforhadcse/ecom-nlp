import pandas as pd
import matplotlib.pyplot as plt

# PATH = "../data/processed/dataset_final_selected_processed.csv"
PATH = "../gold/dataset_1k_compound_small_aspects.csv"
df = pd.read_csv(PATH)

col = "Product Category"
top_k = 25

s = df[col].astype(str).str.strip()
# remove blanks and NaN-as-text
s = s[(s != "") & (s.str.lower() != "nan")]

# Top-K by count, then convert to % of the whole dataset (after filtering)
counts = s.value_counts()
top = counts.head(top_k)
pct = (top / counts.sum() * 100).sort_values()  # sort for nice horizontal bar order

plt.figure(figsize=(10, 6))
plt.barh(pct.index, pct.values)
plt.title(f"Top {top_k} Product Categories (% of total)")
plt.xlabel("Percentage (%)")
plt.ylabel("Product Category")

# show % values at the end of each bar
for i, v in enumerate(pct.values):
    plt.text(v + 0.2, i, f"{v:.1f}%", va="center")

plt.xlim(0, max(pct.values) * 1.15)
plt.tight_layout()
plt.show()




