import pandas as pd
import matplotlib.pyplot as plt
import json

# ====== Load dataset ======
PATH = "../data/processed/dataset_final_selected_processed.csv"
df = pd.read_csv(PATH)

# ====== Choose columns ======
# If you want model output instead, switch to:
emotion_col = "Product Category"
sentiment_col = "absa_overall_sentiment"
# emotion_col = "Emotion"
# sentiment_col = "Sentiment"


def to_list(x):
    """Turn a cell into a list (handles JSON list, comma-separated, or single label)."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []
    # Try JSON (e.g., ["Joy","Sadness"])
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(i).strip() for i in v if str(i).strip()]
    except Exception:
        pass
    # Fallback: comma-separated
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def plot_pie(series: pd.Series, title: str):
    counts = series.value_counts(dropna=True)

    # Group very small slices into "Other" to avoid label overlap
    total = counts.sum()
    pct = (counts / total) * 100
    small = pct[pct < 2.0].index
    # if len(small) > 0 and len(counts) > 1:
    #     other_sum = counts.loc[small].sum()
    #     counts = counts.drop(index=small)
    #     counts.loc["Other"] = other_sum

    def autopct_fmt(p: float) -> str:
        # Hide tiny percentage texts to reduce clutter
        return f"{p:.1f}%" if p >= 1.0 else ""

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=None,  # use legend instead of labels on wedges
        autopct=autopct_fmt,
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        wedgeprops=dict(linewidth=1, edgecolor="white"),
        textprops=dict(fontsize=11),
    )

    ax.set_title(title)

    # Put class labels in a legend outside the pie to prevent overlap
    ax.legend(
        wedges,
        counts.index.tolist(),
        title="Class",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    plt.tight_layout()
    plt.show()

# ====== Emotion pie ======
emo = df[emotion_col].apply(to_list).explode()
emo = emo[emo.notna() & (emo.astype(str).str.strip() != "")]
plot_pie(emo, f"Emotion Distribution ({emotion_col})")

# ====== Sentiment pie ======
sent = df[sentiment_col].apply(to_list).explode()
sent = sent[sent.notna() & (sent.astype(str).str.strip() != "")]
plot_pie(sent, f"Sentiment Distribution ({sentiment_col})")