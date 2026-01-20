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


# Sentiment normalization helper
def norm_sent(x: Any) -> str:
    """Normalize sentiment labels across common variants."""
    s = norm_text(x)
    if not s:
        return ""
    if s in {"pos", "positive", "Positive", "+", "1"}:
        return "positive"
    if s in {"neg", "negative", "Negative", "-", "-1"}:
        return "negative"
    if s in {"neu", "neutral", "Neutral", "0"}:
        return "neutral"
    return s


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
# Soft matching helpers (token overlap / Jaccard / substring)
# ----------------------------

def tokenize(text: Any) -> List[str]:
    """Unicode-aware tokenization for Bangla-English code-mix.

    Important: for taxonomy labels like PRODUCT_QUALITY / Delivery_Person_Behavior,
    we *must* split underscores and other separators, otherwise Jaccard/overlap
    becomes almost always 0 or 1 and changing thresholds won't affect metrics.
    """
    if text is None:
        return []
    if isinstance(text, float) and np.isnan(text):
        return []

    s = str(text).strip()
    if not s:
        return []

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)

    # Split common label separators into spaces so they become separate tokens
    # (underscores were previously treated as part of a token by \w)
    s = re.sub(r"[_.:/|\\-]+", " ", s)

    # Casefold for consistent comparison (handles Bengali + English)
    s = s.casefold()

    # Split on non-word characters (unicode-aware). After separator normalization
    # this yields meaningful multi-token sets for category labels.
    toks = re.split(r"[^\w]+", s)
    return [t for t in toks if t]


def jaccard_sim(a: Any, b: Any) -> float:
    A = set(tokenize(a))
    B = set(tokenize(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def overlap_sim(a: Any, b: Any) -> float:
    """Overlap coefficient: |A∩B| / min(|A|,|B|)"""
    A = set(tokenize(a))
    B = set(tokenize(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    denom = min(len(A), len(B))
    return (len(A & B) / denom) if denom else 0.0


def substring_sim(a: Any, b: Any) -> float:
    """1.0 if either normalized string is a substring of the other."""
    sa = norm_text(a)
    sb = norm_text(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return 1.0 if (sa in sb or sb in sa) else 0.0


def combined_text_sim(a: Any, b: Any) -> float:
    """Robust similarity used for aspect_term/opinion_term.
    Takes the max of substring, Jaccard, and overlap."""
    return max(substring_sim(a, b), jaccard_sim(a, b), overlap_sim(a, b))


def aspect_category_any_sim(a: Any, b: Any) -> float:
    """Similarity for packed category strings like 'TOP=...||SUB=...'.

    A match is accepted if either the TOP or SUB component matches well.
    We take the maximum similarity over all cross-comparisons.
    """
    sa = str(a or "")
    sb = str(b or "")

    def split_parts(s: str) -> Tuple[str, str]:
        top = ""
        sub = ""
        try:
            parts = s.split("||")
            for p in parts:
                pc = str(p).casefold()
                if pc.startswith("top="):
                    # keep value only
                    top = str(p)[4:]
                elif pc.startswith("sub="):
                    sub = str(p)[4:]
        except Exception:
            pass
        return top, sub

    a_top, a_sub = split_parts(sa)
    b_top, b_sub = split_parts(sb)

    return max(
        combined_text_sim(a_top, b_top),
        combined_text_sim(a_sub, b_sub),
        combined_text_sim(a_top, b_sub),
        combined_text_sim(a_sub, b_top),
    )


def greedy_soft_match_counts(true_items: List[str], pred_items: List[str], sim_fn, threshold: float) -> Tuple[int, int, int]:
    """One-to-one greedy matching.

    TP = number of matched pairs (sim>=threshold)
    FP = unmatched predictions
    FN = unmatched gold

    Greedy is simple and stable for small per-row lists.
    """
    true_items = [norm_text(x) for x in true_items if norm_text(x)]
    pred_items = [norm_text(x) for x in pred_items if norm_text(x)]

    if not true_items and not pred_items:
        return 0, 0, 0

    used_true = set()
    tp = 0

    for p in pred_items:
        best_i = None
        best_s = -1.0
        for i, t in enumerate(true_items):
            if i in used_true:
                continue
            s = float(sim_fn(t, p))
            if s > best_s:
                best_s = s
                best_i = i
        if best_i is not None and best_s >= threshold:
            used_true.add(best_i)
            tp += 1

    fp = max(0, len(pred_items) - tp)
    fn = max(0, len(true_items) - tp)
    return tp, fp, fn


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return float(prec), float(rec), float(f1)


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

# ----------------------------
# Field-only evaluation (aspect_category only, aspect_term only, opinion_term only)
# ----------------------------

def extract_field_values(triplets: List[Dict[str, Any]], field: str, category_level: str = "top") -> List[str]:
    """Extract a list of values for a given field from a list of triplet dicts.

    field:
      - "aspect_term"
      - "opinion_term"
      - "aspect_category"
      - "sentiment"  # sentiment inside triplets

    category_level for aspect_category:
      - "any": combines top-level + leaf token (TOP=...||SUB=...)
      - "top": coarsened top-level category (GLOBAL_TAXONOMY parent or DOMAIN parent)
      - "sub": leaf/subcategory token (best-effort)
      - "raw": raw string from data
    """
    out: List[str] = []
    for t in triplets:
        if not isinstance(t, dict):
            continue
        if field == "aspect_term":
            out.append(str(t.get("aspect_term") or "").strip())
        elif field == "opinion_term":
            out.append(str(t.get("opinion_term") or "").strip())
        elif field == "sentiment":
            out.append(norm_sent(t.get("sentiment")))
        elif field == "aspect_category":
            raw = t.get("aspect_category")
            if category_level == "any":
                top = str(coarse_category(raw) or "").strip()
                # best-effort leaf token
                s = "" if raw is None else str(raw).strip()
                s = s.replace("/", ".").replace("|", ".").replace(":", ".")
                parts = [p for p in s.split(".") if p]
                sub = parts[-1] if parts else s
                out.append(f"TOP={top}||SUB={sub}")
            elif category_level == "top":
                out.append(str(coarse_category(raw) or "").strip())
            elif category_level == "sub":
                # best-effort: take the last dotted token
                s = "" if raw is None else str(raw).strip()
                s = s.replace("/", ".").replace("|", ".").replace(":", ".")
                parts = [p for p in s.split(".") if p]
                out.append(parts[-1] if parts else s)
            else:  # raw
                out.append("" if raw is None else str(raw).strip())
    # Normalize empties
    return [x for x in out if norm_text(x)]


def micro_field_scores_soft(
    df: pd.DataFrame,
    gold_col: str,
    pred_col: str,
    field: str,
    *,
    category_level: str = "top",
    sim_fn=combined_text_sim,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Micro precision/recall/F1 for a single extracted field using soft matching."""
    TP = FP = FN = 0

    for _, row in df.iterrows():
        gold_trip = parse_triplets(row.get(gold_col))
        pred_trip = parse_triplets(row.get(pred_col))

        gold_vals = extract_field_values(gold_trip, field, category_level=category_level)
        pred_vals = extract_field_values(pred_trip, field, category_level=category_level)

        tp, fp, fn = greedy_soft_match_counts(gold_vals, pred_vals, sim_fn, threshold)
        TP += tp
        FP += fp
        FN += fn

    prec, rec, f1 = prf_from_counts(TP, FP, FN)
    return {
        "tp": TP, "fp": FP, "fn": FN,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "field": field,
        "category_level": category_level,
        "threshold": float(threshold),
    }



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
# Soft triplet evaluation (supports token overlap / Jaccard / substring)
# ----------------------------

def micro_triplet_scores_soft(
    df: pd.DataFrame,
    gold_col: str,
    pred_col: str,
    mode: str,
    *,
    threshold_term: float = 0.5,
    threshold_opinion: float = 0.5,
    threshold_cat: float = 0.5,
) -> Dict[str, Any]:
    """Soft matching version of triplet scoring.

    mode:
      - "term_sent_soft": aspect_term matched softly + sentiment exact
      - "cat_sent_any_soft": category matched if either top-level OR leaf matches + sentiment exact
      - "full_soft": category (top-level coarse) + aspect_term + opinion_term matched softly + sentiment exact

    Uses one-to-one greedy matching within each row.
    """
    TP = FP = FN = 0

    for _, row in df.iterrows():
        gold_trip = parse_triplets(row.get(gold_col))
        pred_trip = parse_triplets(row.get(pred_col))

        # Build comparable strings per triplet depending on mode.
        gold_items: List[str] = []
        pred_items: List[str] = []

        def pack_term_sent(t: Dict[str, Any]) -> str:
            term = str(t.get("aspect_term") or "").strip()
            sent = str(t.get("sentiment") or "").strip()
            return f"TERM={term}||SENT={sent}"

        def pack_cat_sent_any(t: Dict[str, Any]) -> str:
            raw = t.get("aspect_category")
            top = str(coarse_category(raw) or "").strip()
            s = "" if raw is None else str(raw).strip()
            s = s.replace("/", ".").replace("|", ".").replace(":", ".")
            parts = [p for p in s.split(".") if p]
            sub = parts[-1] if parts else s
            sent = str(t.get("sentiment") or "").strip()
            return f"CAT=TOP={top}||SUB={sub}||SENT={sent}"

        def pack_full(t: Dict[str, Any]) -> Tuple[str, str, str, str]:
            cat = str(coarse_category(t.get("aspect_category")) or "").strip()
            term = str(t.get("aspect_term") or "").strip()
            opin = str(t.get("opinion_term") or "").strip()
            sent = str(t.get("sentiment") or "").strip()
            return cat, term, opin, sent

        if mode == "term_sent_soft":
            gold_items = [pack_term_sent(t) for t in gold_trip if isinstance(t, dict)]
            pred_items = [pack_term_sent(t) for t in pred_trip if isinstance(t, dict)]

            # Similarity: require sentiment exact; term soft.
            def sim_fn(a: str, b: str) -> float:
                # Robust parse (case-insensitive) of packed strings like TERM=...||SENT=...
                def unpack(x: str) -> Tuple[str, str]:
                    s = str(x or "").casefold()
                    term_val = ""
                    sent_val = ""
                    for part in s.split("||"):
                        part = part.strip()
                        if part.startswith("term="):
                            term_val = part[5:]
                        elif part.startswith("sent="):
                            sent_val = part[5:]
                    return term_val, sent_val

                a_term, a_sent = unpack(a)
                b_term, b_sent = unpack(b)
                if a_sent != b_sent:
                    return 0.0
                return combined_text_sim(a_term, b_term)

            tp, fp, fn = greedy_soft_match_counts(gold_items, pred_items, sim_fn, threshold_term)

        elif mode == "cat_sent_any_soft":
            gold_items = [pack_cat_sent_any(t) for t in gold_trip if isinstance(t, dict)]
            pred_items = [pack_cat_sent_any(t) for t in pred_trip if isinstance(t, dict)]

            # Similarity: require sentiment exact; category matches if either TOP or SUB matches.
            def sim_fn(a: str, b: str) -> float:
                # Robust parse (case-insensitive) of packed strings like CAT=TOP=...||SUB=...||SENT=...
                def unpack(x: str) -> Tuple[str, str]:
                    s = str(x or "").casefold()
                    sent_val = ""
                    cat_parts: List[str] = []
                    for part in s.split("||"):
                        part = part.strip()
                        if part.startswith("sent="):
                            sent_val = part[5:]
                        elif part.startswith("cat="):
                            cat_parts.append(part[4:])
                        else:
                            # could be TOP=... or SUB=... if CAT= was stripped earlier
                            cat_parts.append(part)
                    # normalize to the expected packed form TOP=...||SUB=...
                    cat_val = "||".join([p for p in cat_parts if p])
                    return cat_val, sent_val

                a_cat, a_sent = unpack(a)
                b_cat, b_sent = unpack(b)
                if a_sent != b_sent:
                    return 0.0
                return aspect_category_any_sim(a_cat, b_cat)

            tp, fp, fn = greedy_soft_match_counts(gold_items, pred_items, sim_fn, threshold_cat)

        elif mode == "full_soft":
            gold_full = [pack_full(t) for t in gold_trip if isinstance(t, dict)]
            pred_full = [pack_full(t) for t in pred_trip if isinstance(t, dict)]

            # Similarity combines: sentiment exact + category similarity + term/opinion similarity
            def sim_fn(a: Tuple[str, str, str, str], b: Tuple[str, str, str, str]) -> float:
                a_cat, a_term, a_op, a_sent = a
                b_cat, b_term, b_op, b_sent = b
                if norm_text(a_sent) != norm_text(b_sent):
                    return 0.0

                cat_s = combined_text_sim(a_cat, b_cat)
                term_s = combined_text_sim(a_term, b_term)
                op_s = combined_text_sim(a_op, b_op)

                # Weighted blend: category matters but term/opinion dominate extraction quality
                return 0.25 * cat_s + 0.45 * term_s + 0.30 * op_s

            # Greedy matcher for tuples (slightly adapted)
            used_true = set()
            tp = 0
            for p in pred_full:
                best_i = None
                best_s = -1.0
                for i, t in enumerate(gold_full):
                    if i in used_true:
                        continue
                    s = float(sim_fn(t, p))
                    if s > best_s:
                        best_s = s
                        best_i = i
                if best_i is not None and best_s >= max(threshold_cat, threshold_term, threshold_opinion):
                    used_true.add(best_i)
                    tp += 1
            fp = max(0, len(pred_full) - tp)
            fn = max(0, len(gold_full) - tp)

        else:
            raise ValueError(f"Unknown soft triplet scoring mode: {mode}")

        TP += tp
        FP += fp
        FN += fn

    prec, rec, f1 = prf_from_counts(TP, FP, FN)
    return {
        "tp": TP, "fp": FP, "fn": FN,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mode": mode,
        "threshold_term": float(threshold_term),
        "threshold_opinion": float(threshold_opinion),
        "threshold_cat": float(threshold_cat),
    }


# ----------------------------
# Main evaluation
# ----------------------------

def main():

    # ----------------------------
    # Fixed inputs/outputs (no CLI arguments)
    # ----------------------------
    pred_path = Path("fireworks_batch/dataset_qwen2p5-7b-instruct_fireworks_processed_20260120-031010.csv")
    out_dir = Path(f"reports_v2/{pred_path.name.split('_')[1]}")
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

    # Triplet scores (exact)
    trip_term = micro_triplet_scores(df, "absa_triplets_json", "fw_triplets_json", mode="term_sent")
    trip_cat = micro_triplet_scores(df, "absa_triplets_json", "fw_triplets_json", mode="cat_sent_coarse")
    trip_all = micro_triplet_scores(df, "absa_triplets_json", "fw_triplets_json", mode="full")

    # ----------------------------
    # Threshold knobs (change here)
    # ----------------------------
    TERM_THRESH = 0.8
    OPIN_THRESH = 0.8
    CAT_THRESH = 0.8

    # Triplet scores (soft)
    # Tune thresholds if needed; 0.5 is a reasonable starting point for token overlap.
    trip_term_soft = micro_triplet_scores_soft(
        df, "absa_triplets_json", "fw_triplets_json", mode="term_sent_soft",
        threshold_term=TERM_THRESH
    )
    trip_cat_sent_soft = micro_triplet_scores_soft(
        df, "absa_triplets_json", "fw_triplets_json", mode="cat_sent_any_soft",
        threshold_cat=CAT_THRESH
    )
    trip_full_soft = micro_triplet_scores_soft(
        df, "absa_triplets_json", "fw_triplets_json", mode="full_soft",
        threshold_cat=CAT_THRESH, threshold_term=TERM_THRESH, threshold_opinion=OPIN_THRESH
    )

    # Field-only metrics (requested): aspect_category (match if TOP or SUB matches), aspect_term only
    aspect_cat = micro_field_scores_soft(
        df, "absa_triplets_json", "fw_triplets_json", field="aspect_category",
        category_level="any", sim_fn=aspect_category_any_sim, threshold=CAT_THRESH
    )
    # aspect_term only
    aspect_term_only = micro_field_scores_soft(
        df, "absa_triplets_json", "fw_triplets_json", field="aspect_term",
        sim_fn=combined_text_sim, threshold=TERM_THRESH
    )
    # opinion_term only (optional but useful to track)
    opinion_term_only = micro_field_scores_soft(
        df, "absa_triplets_json", "fw_triplets_json", field="opinion_term",
        sim_fn=combined_text_sim, threshold=OPIN_THRESH
    )
    # triplet sentiment only (micro PRF over sentiments inside triplets)
    def sentiment_exact_sim(a: Any, b: Any) -> float:
        return 1.0 if norm_sent(a) == norm_sent(b) else 0.0

    triplet_sentiment_only = micro_field_scores_soft(
        df, "absa_triplets_json", "fw_triplets_json", field="sentiment",
        sim_fn=sentiment_exact_sim, threshold=1.0
    )

    # Per-row diagnostics (useful for error analysis)
    diag_rows = []
    for idx, row in df.iterrows():
        true_trip = set(filter(None, [norm_triplet(t, "term_sent") for t in parse_triplets(row.get("absa_triplets_json"))]))
        pred_trip = set(filter(None, [norm_triplet(t, "term_sent") for t in parse_triplets(row.get("fw_triplets_json"))]))
        _, _, _, p, r, f1 = set_prf(true_trip, pred_trip)

        gold_triplets = parse_triplets(row.get("absa_triplets_json"))
        pred_triplets = parse_triplets(row.get("fw_triplets_json"))

        gold_cat_any = extract_field_values(gold_triplets, "aspect_category", category_level="any")
        pred_cat_any = extract_field_values(pred_triplets, "aspect_category", category_level="any")

        tp_c, fp_c, fn_c = greedy_soft_match_counts(gold_cat_any, pred_cat_any, aspect_category_any_sim, CAT_THRESH)
        cat_p, cat_r, cat_f1 = prf_from_counts(tp_c, fp_c, fn_c)

        gold_term = extract_field_values(gold_triplets, "aspect_term")
        pred_term = extract_field_values(pred_triplets, "aspect_term")
        tp_t, fp_t, fn_t = greedy_soft_match_counts(gold_term, pred_term, combined_text_sim, TERM_THRESH)
        term_p, term_r, term_f1 = prf_from_counts(tp_t, fp_t, fn_t)

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
            "aspect_category_precision_soft": cat_p,
            "aspect_category_recall_soft": cat_r,
            "aspect_category_f1_soft": cat_f1,
            "aspect_term_precision_soft": term_p,
            "aspect_term_recall_soft": term_r,
            "aspect_term_f1_soft": term_f1,
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

        "triplet_term_sent_micro_precision_soft": trip_term_soft["precision"],
        "triplet_term_sent_micro_recall_soft": trip_term_soft["recall"],
        "triplet_term_sent_micro_f1_soft": trip_term_soft["f1"],
        "triplet_cat_sent_micro_precision_soft": trip_cat_sent_soft["precision"],
        "triplet_cat_sent_micro_recall_soft": trip_cat_sent_soft["recall"],
        "triplet_cat_sent_micro_f1_soft": trip_cat_sent_soft["f1"],
        "triplet_full_micro_precision_soft": trip_full_soft["precision"],
        "triplet_full_micro_recall_soft": trip_full_soft["recall"],
        "triplet_full_micro_f1_soft": trip_full_soft["f1"],

        "aspect_category_micro_precision": aspect_cat["precision"],
        "aspect_category_micro_recall": aspect_cat["recall"],
        "aspect_category_micro_f1": aspect_cat["f1"],
        "aspect_term_micro_precision": aspect_term_only["precision"],
        "aspect_term_micro_recall": aspect_term_only["recall"],
        "aspect_term_micro_f1": aspect_term_only["f1"],
        "opinion_term_micro_precision": opinion_term_only["precision"],
        "opinion_term_micro_recall": opinion_term_only["recall"],
        "opinion_term_micro_f1": opinion_term_only["f1"],
        "triplets_sentiment_micro_precision": triplet_sentiment_only["precision"],
        "triplets_sentiment_micro_recall": triplet_sentiment_only["recall"],
        "triplets_sentiment_micro_f1": triplet_sentiment_only["f1"],
        # Add threshold columns for transparency
        "threshold_cat": CAT_THRESH,
        "threshold_term": TERM_THRESH,
        "threshold_opinion": OPIN_THRESH,
    }])
    summary.to_csv(out_dir / "performance_summary.csv", index=False)

    print("\nSaved evaluation outputs to:", out_dir.resolve())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()