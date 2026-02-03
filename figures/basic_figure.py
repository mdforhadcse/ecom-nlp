import json
import ast
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1) Load CSV
# =========================
CSV_PATH = "../gold/test_250.csv"
df = pd.read_csv(CSV_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# =========================
# 2) Parse triplets_json into a long (exploded) table
# =========================
def parse_triplets(cell):
    """Return list[dict] from JSON string or python-literal string; else empty list."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return []

rows = []
for i, r in df.iterrows():
    trip_list = parse_triplets(r.get("absa_triplets_json"))
    for t in trip_list:
        rows.append({
            "row_id": i,
            "review": r.get("Review"),
            "overall_sentiment": r.get("absa_overall_sentiment"),
            "overall_emotion": r.get("absa_overall_emotion"),
            "aspect_category": t.get("aspect_category"),
            "aspect_term": t.get("aspect_term"),
            "opinion_term": t.get("opinion_term"),
            "triplet_sentiment": t.get("sentiment"),
        })

tdf = pd.DataFrame(rows)
print("\nTriplets table shape:", tdf.shape)
print("Unique aspect categories:", tdf["aspect_category"].nunique())
print("Avg triplets/review:", round(len(tdf) / len(df), 2))

# =========================
# 3) Basic counts (for reporting)
# =========================
print("\nOverall sentiment counts:\n", df["absa_overall_sentiment"].value_counts(dropna=False))
print("\nOverall emotion counts:\n", df["absa_overall_emotion"].value_counts(dropna=False))

# =========================
# 4) FIGURE A — Overall sentiment distribution (bar)
# =========================
sent_counts = df["absa_overall_sentiment"].value_counts()
plt.figure()
sent_counts.plot(kind="bar")
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =========================
# 5) FIGURE B — Overall emotion distribution (bar)
# =========================
emo_counts = df["absa_overall_emotion"].value_counts()
plt.figure()
emo_counts.plot(kind="bar")
plt.title("Overall Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =========================
# 6) FIGURE C — Top-N aspect categories (bar)
# =========================
TOP_N = 15
top_aspects = tdf["aspect_category"].value_counts().head(TOP_N)

plt.figure()
top_aspects.sort_values().plot(kind="barh")
plt.title(f"Top {TOP_N} Aspect Categories (Triplet Frequency)")
plt.xlabel("Triplet Count")
plt.ylabel("Aspect Category")
plt.tight_layout()
plt.show()

# =========================
# 7) FIGURE D — Stacked bar: triplet sentiment breakdown for top aspects
# =========================
# Build a pivot: aspect_category x triplet_sentiment
pivot = (
    tdf[tdf["aspect_category"].isin(top_aspects.index)]
    .groupby(["aspect_category", "triplet_sentiment"])
    .size()
    .unstack(fill_value=0)
)

# Make the plot
plt.figure()
pivot.loc[top_aspects.index].plot(kind="bar", stacked=True)
plt.title(f"Triplet Sentiment Breakdown for Top {TOP_N} Aspects")
plt.xlabel("Aspect Category")
plt.ylabel("Triplet Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =========================
# 8) FIGURE E — Heatmap: Overall emotion vs overall sentiment
# (no seaborn; pure matplotlib)
# =========================
ct = pd.crosstab(df["absa_overall_emotion"], df["absa_overall_sentiment"])

plt.figure()
plt.imshow(ct.values, aspect="auto")
plt.title("Overall Emotion vs Overall Sentiment (Counts)")
plt.xlabel("Overall Sentiment")
plt.ylabel("Overall Emotion")
plt.xticks(range(ct.shape[1]), ct.columns, rotation=45, ha="right")
plt.yticks(range(ct.shape[0]), ct.index)

# annotate cells
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        plt.text(j, i, int(ct.values[i, j]), ha="center", va="center")

plt.tight_layout()
plt.show()

# =========================
# 9) Optional: save figures instead of show()
# =========================
# Replace plt.show() with:
# plt.savefig("figure_name.png", dpi=300, bbox_inches="tight")