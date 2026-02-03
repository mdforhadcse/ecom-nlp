import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from collections import Counter

# ---- 1) Load the file that contains both A and B ----
# If your file is somewhere else, change this path.
PATH = "sample_400_absa_triplets_json_validation.csv"
df = pd.read_csv(PATH)

# ---- 2) Canonicalize triplets so ordering/spaces don't affect comparison ----
def canonical_triplets(triplets_json_str: str) -> str:
    """
    Convert a triplets JSON string into a stable label string.
    - parses JSON list of dicts
    - keeps only (aspect_category, aspect_term, sentiment) if present
    - sorts triplets so order doesn't matter
    - returns a compact JSON string
    """
    if pd.isna(triplets_json_str):
        return "[]"

    s = str(triplets_json_str).strip()
    if s == "":
        return "[]"

    try:
        obj = json.loads(s)
    except Exception:
        # If parsing fails, fall back to raw string
        return s

    # If it's not a list, just dump it stably
    if not isinstance(obj, list):
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    norm = []
    for t in obj:
        if isinstance(t, dict):
            keys = ("aspect_category", "aspect_term", "sentiment")
            if all(k in t for k in keys):
                norm.append({k: t.get(k) for k in keys})
            else:
                norm.append(t)
        else:
            norm.append(t)

    def sort_key(x):
        if isinstance(x, dict):
            return (
                str(x.get("aspect_category", "")),
                str(x.get("aspect_term", "")),
                str(x.get("sentiment", "")),
            )
        return (str(x), "", "")

    norm_sorted = sorted(norm, key=sort_key)
    return json.dumps(norm_sorted, sort_keys=True, separators=(",", ":"))

# ---- 3) Build label vectors ----
A = df["llm_absa_triplets"].apply(canonical_triplets)
B = df["human_absa_triplets"].apply(canonical_triplets)
# print(A.head())
# ---- 4) Cohen's kappa ----
kappa = cohen_kappa_score(A, B)

# ---- 5) (Optional) Also show observed agreement + expected agreement ----
n = len(A)
po = (A.values == B.values).mean()

cA = Counter(A)
cB = Counter(B)
pe = sum((cA[k] / n) * (cB[k] / n) for k in set(cA) | set(cB))
kappa_manual = (po - pe) / (1 - pe) if (1 - pe) != 0 else float("nan")

print(f"Rows: {n}")
# print(f"Observed agreement (Po): {po:.6f}")
# print(f"Expected agreement (Pe): {pe:.6f}")
print(f"Cohen's kappa (using sklearn): {kappa:.6f}")
print(f"Cohen's kappa (manual calculation):  {kappa_manual:.6f}")

# You should see kappa ≈ 0.72 (around 0.719...)