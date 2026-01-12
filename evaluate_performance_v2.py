import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ----------------------------
# Helpers
# ----------------------------

def norm_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def safe_json_loads(s: Any) -> Optional[Any]:
    if s is None:
        return None
    if isinstance(s, float) and np.isnan(s):
        return None
    if isinstance(s, (dict, list)):
        return s
    txt = str(s).strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None


def parse_triplets(cell: Any) -> List[Dict[str, Any]]:
    """
    cell is expected to be a JSON string containing either:
      - a list of triplets, OR
      - a dict with key "triplets"
    """
    obj = safe_json_loads(cell)
    if obj is None:
        return []
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)]
    if isinstance(obj, dict):
        ts = obj.get("triplets")
        if isinstance(ts, list):
            return [t for t in ts if isinstance(t, dict)]
    return []


def set_prf(true_set: Set[Any], pred_set: Set[Any]) -> Tuple[int, int, int, float, float, float]:
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if not true_set and not pred_set else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else (1.0 if not true_set and not pred_set else 0.0)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else (1.0 if not true_set and not pred_set else 0.0)
    return tp, fp, fn, prec, rec, f1


# ----------------------------
# Taxonomy coarsening (optional but recommended)
# ----------------------------
GLOBAL_TAXONOMY = {
    "DELIVERY": {"Speed", "Timeliness", "Delivery_Person_Behavior", "Shipping_Cost"},
    "PACKAGING": {"Protection", "Condition", "Aesthetics"},
    "PRICE": {"Value_for_Money", "Discount_Offer", "Extra_Charges"},
    "SERVICE": {"Customer_Support", "Seller_Interaction", "Order_Process", "Return_Refund", "Warranty_After_Sales"},
    "PRODUCT_QUALITY": {"Overall_Quality", "Durability_Longevity", "Performance_Effectiveness"},
    "AUTHENTICITY": {"Originality", "Fake_Counterfeit"},
    "PRODUCT_APPEARANCE_DESIGN": {"Design_Style", "Color_Finish", "Size_Dimensions", "Comfort_Feeling"},
    "USABILITY_SETUP": {"Instructions_Manual", "Compatibility"},
}

# Domain-specific taxonomy (optional): used to coarsen domain categories like CAMERA.Photo_Quality -> CAMERA
# Mirrors taxonomy_json_string -> domain_specific_aspects.*.categories
DOMAIN_TAXONOMY = {
    "ELECTRONICS_AND_GADGETS": {
        "PERFORMANCE": set(),
        "BATTERY": set(),
        "DISPLAY": set(),
        "CAMERA": {"Photo_Quality", "Video_Quality"},
        "BUILD_QUALITY": set(),
        "AUDIO": set(),
        "SOFTWARE": set(),
        "STORAGE": set(),
        "AUTHENTICITY": {"Originality", "Fake"},
    },
    "FASHION_AND_APPAREL": {
        "MATERIAL": set(),
        "FIT_SIZE": set(),
        "DESIGN": set(),
        "DURABILITY": set(),
        "FUNCTIONALITY": set(),
    },
    "BEAUTY_AND_PERSONAL_CARE": {
        "EFFICACY": set(),
        "TEXTURE_SMELL": set(),
        "SUITABILITY": set(),
        "AUTHENTICITY": {"Originality"},
        "PACKAGING": set(),
    },
    "HOME_LIVING_AND_APPLIANCES": {
        "FUNCTIONALITY": {"Performance", "Power_Consumption", "Noise"},
        "BUILD_QUALITY": set(),
        "AESTHETICS": set(),
        "INSTALLATION": set(),
        "DIMENSIONS": set(),
    },
    "FOOD_AND_GROCERY": {
        "TASTE_FLAVOR": set(),
        "FRESHNESS": set(),
        "QUALITY": {"Authenticity", "Purity", "Ingredients"},
        "EXPIRY": set(),
        "PACKAGING": set(),
    },
    "AUTOMOTIVE": {
        "PERFORMANCE": set(),
        "COMPATIBILITY": {"Fitment", "Model_Match"},
        "AUTHENTICITY": {"Originality"},
        "QUALITY": set(),
    },
    "OTHER_PRODUCTS": {
        "FEATURES_FUNCTIONALITY": set(),
        "MATERIAL_BUILD": set(),
        "SIZE_DIMENSIONS": set(),
        "COMFORT": set(),
        "SAFETY": set(),
        "AUTHENTICITY": {"Originality", "Fake_Counterfeit"},
    },
}

_DOMAIN_SUB_TO_PARENT = {
    sub: parent
    for _domain, cats in DOMAIN_TAXONOMY.items()
    for parent, subs in cats.items()
    for sub in subs
}
_DOMAIN_CATEGORY_SET = {cat for _domain, cats in DOMAIN_TAXONOMY.items() for cat in cats.keys()}
_DOMAIN_SET = set(DOMAIN_TAXONOMY.keys())

_SUB_TO_PARENT = {sub: parent for parent, subs in GLOBAL_TAXONOMY.items() for sub in subs}


def coarse_category(cat: Any) -> str:
    """
    Coarsen aspect categories to reduce penalties for taxonomy granularity differences.

    Examples:
      - "PRODUCT_QUALITY.Overall_Quality" -> "PRODUCT_QUALITY"
      - "Overall_Quality" -> "PRODUCT_QUALITY"
      - "ELECTRONICS_AND_GADGETS.CAMERA.Photo_Quality" -> "CAMERA"
      - "Photo_Quality" -> "CAMERA"

    Unknown labels are kept as the first token (best-effort fallback).
    """
    if cat is None:
        return ""
    if isinstance(cat, float) and np.isnan(cat):
        return ""

    s = str(cat).strip()
    if not s:
        return ""

    # unify separators
    s = s.replace("/", ".").replace("|", ".").replace(":", ".")
    parts = [p for p in s.split(".") if p]
    if not parts:
        return s

    # Case 1: explicit global category prefix (e.g., PRODUCT_QUALITY.Overall_Quality)
    if parts[0] in GLOBAL_TAXONOMY:
        return parts[0]

    # Case 2: explicit domain prefix (e.g., ELECTRONICS_AND_GADGETS.CAMERA.Photo_Quality)
    if parts[0] in _DOMAIN_SET:
        if len(parts) >= 2 and parts[1] in DOMAIN_TAXONOMY[parts[0]]:
            return parts[1]
        # if malformed, still try mapping by last token
        last = parts[-1]
        if last in _DOMAIN_SUB_TO_PARENT:
            return _DOMAIN_SUB_TO_PARENT[last]

    # Case 3: category-only label that matches a known domain category (e.g., CAMERA)
    if parts[0] in _DOMAIN_CATEGORY_SET:
        return parts[0]

    # Case 4: leaf/subcategory label only
    last = parts[-1]
    if last in _SUB_TO_PARENT:
        return _SUB_TO_PARENT[last]
    if last in _DOMAIN_SUB_TO_PARENT:
        return _DOMAIN_SUB_TO_PARENT[last]

    # Case 5: full string matches a leaf/subcategory
    if s in _SUB_TO_PARENT:
        return _SUB_TO_PARENT[s]
    if s in _DOMAIN_SUB_TO_PARENT:
        return _DOMAIN_SUB_TO_PARENT[s]

    # Fallback: keep first token
    return parts[0]


def norm_triplet(t: Dict[str, Any], mode: str) -> Optional[Tuple]:
    """
    mode:
      - term_sent: (aspect_term, sentiment)
      - cat_sent_coarse: (coarse_category(aspect_category), sentiment)
      - full: (coarse_category, aspect_term, opinion_term, sentiment)
    """
    if not isinstance(t, dict):
        return None
    term = norm_text(t.get("aspect_term"))
    sent = norm_text(t.get("sentiment"))
    opin = norm_text(t.get("opinion_term"))
    cat = norm_text(coarse_category(t.get("aspect_category")))

    if mode == "term_sent":
        return (term, sent) if term or sent else None
    if mode == "cat_sent_coarse":
        return (cat, sent) if cat or sent else None
    if mode == "full":
        return (cat, term, opin, sent)
    return None


def micro_triplet_scores(df: pd.DataFrame, gold_col: str, pred_col: str, mode: str) -> Dict[str, Any]:
    TP = FP = FN = 0
    for _, row in df.iterrows():
        true_set = set(filter(None, [norm_triplet(t, mode) for t in parse_triplets(row.get(gold_col))]))
        pred_set = set(filter(None, [norm_triplet(t, mode) for t in parse_triplets(row.get(pred_col))]))
        tp, fp, fn, _, _, _ = set_prf(true_set, pred_set)
        TP += tp
        FP += fp
        FN += fn

    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "tp": TP, "fp": FP, "fn": FN,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mode": mode,
    }


# ----------------------------
# Main evaluation
# ----------------------------

def main():

    # ----------------------------
    # Fixed inputs/outputs (no CLI arguments)
    # ----------------------------
    pred_path = Path("fireworks_batch/dataset_gemma2-9b-it_fireworks_processed_20260111-111953.csv")
    out_dir = Path(f"reports/{pred_path.name.split('_')[1]}")
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {pred_path.resolve()}")

    # Single-file mode: this CSV already contains BOTH pseudo-gold (absa_*) and predictions (fw_*)
    df = pd.read_csv(pred_path)
    required = [
        "absa_overall_sentiment", "absa_overall_emotion", "absa_triplets_json",
        "fw_overall_sentiment", "fw_overall_emotion", "fw_triplets_json",
        "fw_status_code", "fw_error"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coverage / validity
    total_rows = len(df)
    status_200 = int((df["fw_status_code"] == 200).sum()) if df["fw_status_code"].dtype != object else int((df["fw_status_code"].astype(str) == "200").sum())
    has_error = int(df["fw_error"].notna().sum())
    has_pred_sent = int(df["fw_overall_sentiment"].notna().sum())
    has_pred_emo = int(df["fw_overall_emotion"].notna().sum())

    # Sentiment & emotion metrics (on rows where both gold and pred exist)
    sent_mask = df["absa_overall_sentiment"].notna() & df["fw_overall_sentiment"].notna()
    emo_mask = df["absa_overall_emotion"].notna() & df["fw_overall_emotion"].notna()

    y_sent_true = df.loc[sent_mask, "absa_overall_sentiment"].astype(str)
    y_sent_pred = df.loc[sent_mask, "fw_overall_sentiment"].astype(str)

    y_emo_true = df.loc[emo_mask, "absa_overall_emotion"].astype(str)
    y_emo_pred = df.loc[emo_mask, "fw_overall_emotion"].astype(str)

    sent_acc = float(accuracy_score(y_sent_true, y_sent_pred)) if len(y_sent_true) else 0.0
    sent_f1 = float(f1_score(y_sent_true, y_sent_pred, average="macro")) if len(y_sent_true) else 0.0
    emo_acc = float(accuracy_score(y_emo_true, y_emo_pred)) if len(y_emo_true) else 0.0
    emo_f1 = float(f1_score(y_emo_true, y_emo_pred, average="macro")) if len(y_emo_true) else 0.0

    # Confusions
    sent_labels = sorted(set(y_sent_true.unique()) | set(y_sent_pred.unique()))
    emo_labels = sorted(set(y_emo_true.unique()) | set(y_emo_pred.unique()))

    cm_sent = confusion_matrix(y_sent_true, y_sent_pred, labels=sent_labels) if len(y_sent_true) else np.zeros((0,0), dtype=int)
    cm_emo = confusion_matrix(y_emo_true, y_emo_pred, labels=emo_labels) if len(y_emo_true) else np.zeros((0,0), dtype=int)

    pd.DataFrame(cm_sent, index=[f"true_{l}" for l in sent_labels], columns=[f"pred_{l}" for l in sent_labels]) \
        .to_csv(out_dir / "confusion_sentiment.csv")
    pd.DataFrame(cm_emo, index=[f"true_{l}" for l in emo_labels], columns=[f"pred_{l}" for l in emo_labels]) \
        .to_csv(out_dir / "confusion_emotion.csv")

    # Triplet scores
    trip_term = micro_triplet_scores(df, "absa_triplets_json", "fw_triplets_json", mode="term_sent")
    trip_cat = micro_triplet_scores(df, "absa_triplets_json", "fw_triplets_json", mode="cat_sent_coarse")
    trip_all = micro_triplet_scores(df, "absa_triplets_json", "fw_triplets_json", mode="full")

    # Per-row diagnostics (useful for error analysis)
    diag_rows = []
    for idx, row in df.iterrows():
        true_trip = set(filter(None, [norm_triplet(t, "term_sent") for t in parse_triplets(row.get("absa_triplets_json"))]))
        pred_trip = set(filter(None, [norm_triplet(t, "term_sent") for t in parse_triplets(row.get("fw_triplets_json"))]))
        _, _, _, p, r, f1 = set_prf(true_trip, pred_trip)

        diag_rows.append({
            "row_idx": idx,
            "absa_overall_sentiment": row.get("absa_overall_sentiment"),
            "fw_overall_sentiment": row.get("fw_overall_sentiment"),
            "absa_overall_emotion": row.get("absa_overall_emotion"),
            "fw_overall_emotion": row.get("fw_overall_emotion"),
            "absa_triplets_n": len(parse_triplets(row.get("absa_triplets_json"))),
            "fw_triplets_n": len(parse_triplets(row.get("fw_triplets_json"))),
            "triplet_term_sent_precision": p,
            "triplet_term_sent_recall": r,
            "triplet_term_sent_f1": f1,
            "fw_status_code": row.get("fw_status_code"),
            "fw_error": row.get("fw_error"),
        })

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out_dir / "per_row_diagnostics.csv", index=False)

    # Summary
    summary = pd.DataFrame([{
        "rows": total_rows,
        "fw_status_200": status_200,
        "fw_errors": has_error,
        "fw_has_sentiment": has_pred_sent,
        "fw_has_emotion": has_pred_emo,
        "sentiment_accuracy": sent_acc,
        "sentiment_macro_f1": sent_f1,
        "emotion_accuracy": emo_acc,
        "emotion_macro_f1": emo_f1,
        "triplet_term_sent_micro_precision": trip_term["precision"],
        "triplet_term_sent_micro_recall": trip_term["recall"],
        "triplet_term_sent_micro_f1": trip_term["f1"],
        "triplet_cat_sent_coarse_micro_precision": trip_cat["precision"],
        "triplet_cat_sent_coarse_micro_recall": trip_cat["recall"],
        "triplet_cat_sent_coarse_micro_f1": trip_cat["f1"],
        "triplet_full_micro_precision": trip_all["precision"],
        "triplet_full_micro_recall": trip_all["recall"],
        "triplet_full_micro_f1": trip_all["f1"],

    }])
    summary.to_csv(out_dir / "performance_summary.csv", index=False)

    print("\nSaved evaluation outputs to:", out_dir.resolve())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()