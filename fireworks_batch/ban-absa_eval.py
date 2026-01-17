import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ------------------------------------
# CONFIG: Change these two values only
# ------------------------------------
MODEL_NAME = "accounts/fireworks/models/gemma2-9b-it"

INPUT_PROCESSED_CSV = "ban-absa_dataset_gemma2-9b-it_fireworks_processed_20260117-112358.csv"
# ------------------------------------


# -----------------------------
# Helpers
# -----------------------------
def sanitize_for_folder(name: str) -> str:
    """Convert model name to safe folder string."""
    name = (name or "").strip()
    name = name.replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name.strip("_") or "model"


def norm_label(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip().lower()
    return s if s else None


def normalize_aspect(x):
    x = norm_label(x)
    if x is None:
        return None
    # accept common variants
    if x in ["others", "other"]:
        return "other"
    return x


def normalize_sentiment(x):
    x = norm_label(x)
    if x is None:
        return None
    if x in ["pos", "positive"]:
        return "positive"
    if x in ["neg", "negative"]:
        return "negative"
    if x in ["neu", "neutral"]:
        return "neutral"
    return x


# -----------------------------
# Main Evaluation
# -----------------------------
def main():
    # Create output folder
    safe_model = sanitize_for_folder(MODEL_NAME.split("/")[-1])
    out_dir = Path(f"../ban-absa_report/ban-absa_{safe_model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    metrics_csv_path = out_dir / "eval_metrics.csv"
    aspect_cm_csv_path = out_dir / "aspect_confusion_matrix.csv"
    sentiment_cm_csv_path = out_dir / "sentiment_confusion_matrix.csv"
    diagnostics_csv_path = out_dir / "per_row_diagnostics.csv"

    # Load processed dataset
    df = pd.read_csv(INPUT_PROCESSED_CSV)

    required_cols = ["aspect", "polarity", "pred_aspect", "pred_sentiment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in processed CSV: {missing}")

    # Normalize labels
    df["aspect_true"] = df["aspect"].apply(normalize_aspect)
    df["sent_true"] = df["polarity"].apply(normalize_sentiment)

    df["aspect_pred"] = df["pred_aspect"].apply(normalize_aspect)
    df["sent_pred"] = df["pred_sentiment"].apply(normalize_sentiment)

    # Keep only rows where labels exist
    eval_df = df.dropna(subset=["aspect_true", "sent_true", "aspect_pred", "sent_pred"]).copy()

    print(f"✅ Total rows in file: {len(df)}")
    print(f"✅ Rows used for evaluation: {len(eval_df)}")

    # Label sets (fixed)
    aspect_labels = ["politics", "sports", "religion", "other"]
    sentiment_labels = ["positive", "negative", "neutral"]

    # -----------------------------
    # Metrics
    # -----------------------------
    aspect_acc = accuracy_score(eval_df["aspect_true"], eval_df["aspect_pred"])
    aspect_p, aspect_r, aspect_f1, _ = precision_recall_fscore_support(
        eval_df["aspect_true"], eval_df["aspect_pred"],
        average="macro", zero_division=0
    )

    sent_acc = accuracy_score(eval_df["sent_true"], eval_df["sent_pred"])
    sent_p, sent_r, sent_f1, _ = precision_recall_fscore_support(
        eval_df["sent_true"], eval_df["sent_pred"],
        average="macro", zero_division=0
    )

    # Save metrics CSV
    metrics_df = pd.DataFrame([
        {
            "Model": MODEL_NAME,
            "Task": "Aspect Extraction",
            "Samples": len(eval_df),
            "Accuracy": aspect_acc,
            "Macro_Precision": aspect_p,
            "Macro_Recall": aspect_r,
            "Macro_F1": aspect_f1,
        },
        {
            "Model": MODEL_NAME,
            "Task": "Sentiment Classification",
            "Samples": len(eval_df),
            "Accuracy": sent_acc,
            "Macro_Precision": sent_p,
            "Macro_Recall": sent_r,
            "Macro_F1": sent_f1,
        }
    ])
    metrics_df.to_csv(metrics_csv_path, index=False)

    # -----------------------------
    # Confusion Matrices
    # -----------------------------
    aspect_cm = confusion_matrix(
        eval_df["aspect_true"], eval_df["aspect_pred"], labels=aspect_labels
    )
    aspect_cm_df = pd.DataFrame(
        aspect_cm,
        index=[f"T_{x}" for x in aspect_labels],
        columns=[f"P_{x}" for x in aspect_labels]
    )
    aspect_cm_df.to_csv(aspect_cm_csv_path, index=True)

    sent_cm = confusion_matrix(
        eval_df["sent_true"], eval_df["sent_pred"], labels=sentiment_labels
    )
    sent_cm_df = pd.DataFrame(
        sent_cm,
        index=[f"T_{x}" for x in sentiment_labels],
        columns=[f"P_{x}" for x in sentiment_labels]
    )
    sent_cm_df.to_csv(sentiment_cm_csv_path, index=True)

    # -----------------------------
    # Per-row diagnostics CSV
    # -----------------------------
    diag = eval_df.copy()

    # Add correctness flags
    diag["aspect_correct"] = (diag["aspect_true"] == diag["aspect_pred"]).astype(int)
    diag["sentiment_correct"] = (diag["sent_true"] == diag["sent_pred"]).astype(int)
    diag["both_correct"] = ((diag["aspect_correct"] == 1) & (diag["sentiment_correct"] == 1)).astype(int)

    # Keep important columns (if available)
    keep_cols = []
    for c in ["orig_id", "post", "aspect", "polarity", "pred_aspect", "pred_sentiment",
              "aspect_true", "sent_true", "aspect_pred", "sent_pred",
              "fw_status_code", "fw_raw_text", "fw_error",
              "aspect_correct", "sentiment_correct", "both_correct"]:
        if c in diag.columns:
            keep_cols.append(c)

    diag_out = diag[keep_cols].copy()
    diag_out.to_csv(diagnostics_csv_path, index=False)

    # -----------------------------
    # Print quick summary
    # -----------------------------
    print("\n📘 Aspect Extraction Results")
    print(f"Accuracy     : {aspect_acc:.4f}")
    print(f"Macro_F1     : {aspect_f1:.4f}")

    print("\n📙 Sentiment Classification Results")
    print(f"Accuracy     : {sent_acc:.4f}")
    print(f"Macro_F1     : {sent_f1:.4f}")

    print("\n✅ Saved files inside:", out_dir.resolve())
    print(" -", metrics_csv_path.name)
    print(" -", aspect_cm_csv_path.name)
    print(" -", sentiment_cm_csv_path.name)
    print(" -", diagnostics_csv_path.name)


if __name__ == "__main__":
    main()