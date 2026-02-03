import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
gemma2_sentiment = "finetune-gemma2-9b-it/confusion_sentiment.csv"
gemma2_emotions = "finetune-gemma2-9b-it/confusion_emotions.csv"
llama2_sentiment = "finetune-llama-v3p1-8b-instruct/confusion_sentiment.csv"
llama2_emotions = "finetune-llama-v3p1-8b-instruct/confusion_emotions.csv"
mistral_sentiment = "finetune-mistral-7b-instruct-v3/confusion_sentiment.csv"
mistral_emotions = "finetune-mistral-7b-instruct-v3/confusion_emotions.csv"
qwen_sentiment = "finetune-qwen2p5-7b-instruct/confusion_sentiment.csv"
qwen_emotions = "finetune-qwen2p5-7b-instruct/confusion_emotions.csv"

# -----------------------------
# Multi-panel confusion matrices
# -----------------------------


# If your CSVs are saved relative to this script, SEARCH_ROOTS will make paths robust.
script_dir = Path(__file__).resolve().parent

# Try to infer project root. If the script lives in a subfolder like `reports_v2/`,
# the model folders are usually in the parent folder (e.g., `ecom-nlp/`).
# You can also override by setting env var: ECOM_NLP_ROOT=/path/to/ecom-nlp
project_root = Path(os.environ.get("ECOM_NLP_ROOT", ""))
if str(project_root).strip():
    project_root = project_root.expanduser().resolve()
else:
    project_root = script_dir.parent

# Search roots (in order). We include a couple of parents just in case.
SEARCH_ROOTS = [
    script_dir,
    project_root,
    project_root.parent,
    Path.cwd(),
]


# --- Helper: find similar confusion-matrix file if the expected one is missing ---
def _find_similar_cm(csv_path: str) -> Path | None:
    """If the exact file isn't found, try to locate a similar CM file in the same folder.

    Example: gemma2-9b-it/confusion_emotions.csv
      -> search gemma2-9b-it/ for confusion*emotion*.csv
    """
    rel = Path(csv_path)
    rel_dir = rel.parent

    # decide whether caller is looking for sentiment or emotion
    target = "emotion" if "emotion" in rel.name.lower() else ("sentiment" if "sentiment" in rel.name.lower() else "")

    # patterns ordered by specificity
    patterns = []
    if target == "emotion":
        patterns = [
            "confusion*emotion*.csv",
            "*emotion*confusion*.csv",
            "*emotion*.csv",
        ]
    elif target == "sentiment":
        patterns = [
            "confusion*sentiment*.csv",
            "*sentiment*confusion*.csv",
            "*sentiment*.csv",
        ]
    else:
        patterns = ["confusion*.csv", "*.csv"]

    for root in SEARCH_ROOTS:
        base = (root / rel_dir)
        if not base.exists() or not base.is_dir():
            continue
        for pat in patterns:
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]

    return None

MODELS = [
    ("Gemma 2", gemma2_sentiment, gemma2_emotions),
    ("Llama 3.1", llama2_sentiment, llama2_emotions),
    ("Mistral", mistral_sentiment, mistral_emotions),
    ("Qwen 2.5", qwen_sentiment, qwen_emotions),
]

# Optional: if you want a fixed label order, set it here.
# If None, labels will be taken from the CSV header/index.

SENTIMENT_LABELS = None  # edit if your labels differ
EMOTION_LABELS = None  # e.g., ["Happy", "Anger", "Sadness", "Fear", "Love"]


# --- Helper: pretty label formatting for plot axes ---
def _pretty_label(s: str) -> str:
    """Make labels shorter for plots (e.g., 'true_Negative' -> 'Negative')."""
    s = str(s)
    for pref in ("true_", "pred_", "gold_", "label_"):
        if s.startswith(pref):
            s = s[len(pref):]
    s = s.replace("_", " ").strip()
    # Title-case words but keep common abbreviations as-is
    return " ".join(w.capitalize() for w in s.split())


def _pretty_labels(labels: list[str]) -> list[str]:
    return [_pretty_label(x) for x in labels]


def _try_read_cm(csv_path: str) -> pd.DataFrame:
    """Read a confusion matrix CSV robustly.

    Supports:
      - with header + first column as index
      - plain numeric matrix (no header)
    """
    # print(f"Loading CM: {csv_path}")
    # Try resolving against multiple roots
    candidates = []

    # If it's already absolute, try it directly
    raw = Path(csv_path).expanduser()
    if raw.is_absolute():
        candidates.append(raw)

    # Try each search root + the relative path
    for root in SEARCH_ROOTS:
        candidates.append((root / csv_path))

    # Finally, try relative to current working directory (as-is)
    candidates.append(Path(csv_path))

    p = None
    for cand in candidates:
        cand = cand.resolve()
        if cand.exists():
            p = cand
            break

    if p is None:
        # Try to auto-locate a similar file (common naming differences)
        alt = _find_similar_cm(csv_path)
        if alt is not None and alt.exists():
            p = alt
        else:
            tried = "\n".join(str(c.resolve()) for c in candidates)
            raise FileNotFoundError(
                f"Confusion matrix file not found: {csv_path}\nTried:\n{tried}"
            )

    # First try: header + index
    df = pd.read_csv(p, index_col=0)

    # If it looks like a single unnamed numeric column (bad parse), retry without index_col.
    if df.shape[1] == 0:
        df = pd.read_csv(p)

    # If headers are numeric 0..n-1, treat as no-header matrix
    if all(str(c).isdigit() for c in df.columns):
        df = pd.read_csv(p, header=None)

    return df


def _coerce_to_square(df: pd.DataFrame, labels: list[str] | None) -> tuple[np.ndarray, list[str]]:
    """Return (matrix, labels). Keeps provided label order if possible."""
    # If df is numeric-only without labels
    if df.index.dtype == "int64" and all(str(c).isdigit() for c in df.columns):
        mat = df.to_numpy()
        auto_labels = labels if labels is not None else [str(i) for i in range(mat.shape[0])]
        return mat, auto_labels

    # Normalize index/columns to strings
    df2 = df.copy()
    df2.index = df2.index.astype(str)
    df2.columns = df2.columns.astype(str)

    if labels is not None:
        # Reindex rows/cols to requested order (fill missing with 0)
        df2 = df2.reindex(index=labels, columns=labels, fill_value=0)
        return df2.to_numpy(), labels

    # Otherwise, use whatever is in the file
    # If file is not square, try to intersect shared labels
    row_labels = list(df2.index)
    col_labels = list(df2.columns)

    if row_labels != col_labels:
        shared = [x for x in row_labels if x in col_labels]
        if shared:
            df2 = df2.reindex(index=shared, columns=shared, fill_value=0)
            return df2.to_numpy(), shared

    return df2.to_numpy(), row_labels


def plot_cm(
    ax,
    cm: np.ndarray,
    labels: list[str],
    title: str,
    vmax: float | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
):
    """Single confusion-matrix heatmap (matplotlib only)."""
    im = ax.imshow(cm, aspect="equal", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=11)

    pretty = _pretty_labels(labels)

    ax.set_xticks(np.arange(len(pretty)))
    ax.set_yticks(np.arange(len(pretty)))
    ax.set_xticklabels(pretty, rotation=30, ha="right")
    ax.set_yticklabels(pretty)

    ax.tick_params(axis="both", labelsize=10)

    # Only show axis labels where needed (reduces clutter in multi-panel grids)
    if show_xlabel:
        ax.set_xlabel("Predicted")
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("True")
    else:
        ax.set_ylabel("")

    # Annotate each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j,
                i,
                f"{int(val)}",
                ha="center",
                va="center",
                fontsize=15,
                color="white",
                fontweight="bold",
            )

    # Thin gridlines (looks cleaner in papers)
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def main(save_path: str | None = "confusion_matrices_grid.png"):
    # Load all matrices first so we can share a single color scale (optional but nicer)
    mats = []
    meta = []

    for model_name, sent_path, emo_path in MODELS:
        # Sentiment
        try:
            sent_df = _try_read_cm(sent_path)
            sent_cm, sent_labels = _coerce_to_square(sent_df, SENTIMENT_LABELS)
            mats.append(sent_cm)
            meta.append((model_name, "Sentiment", sent_cm, sent_labels, None))
        except FileNotFoundError as e:
            meta.append((model_name, "Sentiment", None, None, str(e)))

        # Emotion
        try:
            emo_df = _try_read_cm(emo_path)
            emo_cm, emo_labels = _coerce_to_square(emo_df, EMOTION_LABELS)
            mats.append(emo_cm)
            meta.append((model_name, "Emotion", emo_cm, emo_labels, None))
        except FileNotFoundError as e:
            meta.append((model_name, "Emotion", None, None, str(e)))

    # Shared color scale across all panels (comment out if you want per-panel scale)
    if mats:
        global_vmax = max(float(np.nanmax(m)) for m in mats if getattr(m, "size", 0))
    else:
        global_vmax = 1.0

    # Layout: 2 rows (Sentiment, Emotion) × N models columns
    n_models = len(MODELS)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_models,
        figsize=(4.0 * n_models, 7.5),
        constrained_layout=False,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.15)
    fig.subplots_adjust(right=0.90)  # leave room for the colorbar on the right

    last_im = None

    # Row 0 = Sentiment, Row 1 = Emotion
    for col, (model_name, sent_path, emo_path) in enumerate(MODELS):
        # Sentiment (meta index col*2)
        m_name, kind, sent_cm, sent_labels, err = meta[col * 2]
        if err is None:
            last_im = plot_cm(
                axes[0, col],
                sent_cm,
                sent_labels,
                title=f"{model_name} — Sentiment",
                vmax=global_vmax,
                show_xlabel=False,
                show_ylabel=(col == 0),
            )
        else:
            axes[0, col].axis("off")
            axes[0, col].set_title(f"{model_name} — Sentiment", fontsize=11)
            axes[0, col].text(0.5, 0.5, "MISSING FILE", ha="center", va="center", fontsize=12)

        # Emotion (meta index col*2+1)
        m_name, kind, emo_cm, emo_labels, err = meta[col * 2 + 1]
        if err is None:
            last_im = plot_cm(
                axes[1, col],
                emo_cm,
                emo_labels,
                title=f"{model_name} — Emotion",
                vmax=global_vmax,
                show_xlabel=True,
                show_ylabel=(col == 0),
            )
        else:
            axes[1, col].axis("off")
            axes[1, col].set_title(f"{model_name} — Emotion", fontsize=11)
            axes[1, col].text(0.5, 0.5, "MISSING FILE", ha="center", va="center", fontsize=12)

    # Single shared colorbar for the whole figure (placed in its own axis so it won't overlap)
    if last_im is not None:
        cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])  # [left, bottom, width, height]
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("Count")

    fig.suptitle("Confusion matrix heatmaps of LLMs", fontsize=14)
    plt.subplots_adjust(top=0.90)

    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"Saved figure: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
