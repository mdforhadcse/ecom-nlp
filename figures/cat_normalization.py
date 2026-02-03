import json
import ast
import pandas as pd

INPUT_PATH = "/mnt/data/dataset_1k_compound_small_aspects.csv"
OUTPUT_PATH = "/mnt/data/dataset_1k_compound_small_aspects_maincat.csv"

df = pd.read_csv(INPUT_PATH)

taxonomy_json_string = """
    {
      "taxonomy_meta": {
        "version": "2.1",
        "description": "Cross-domain + domain-specific aspect taxonomy for Bengali-English code-mixed e-commerce reviews.",
        "instructions": "Map extracted aspects to the most specific 'Sub_Category' available. If no sub-category fits, use the parent 'Category'. If the product domain is unclear, use OTHER_PRODUCTS. Infer implicit aspects when clearly implied."
      },
      "global_aspects": {
        "description": "Aspects applicable to ALL product categories (recommended as primary mapping layer).",
        "categories": {
          "DELIVERY": ["Speed", "Timeliness", "Delivery_Person_Behavior", "Shipping_Cost"],
          "PACKAGING": ["Protection", "Condition", "Aesthetics"],
          "PRICE": ["Value_for_Money", "Discount_Offer", "Extra_Charges"],
          "SERVICE": ["Customer_Support", "Seller_Interaction", "Order_Process", "Return_Refund", "Warranty_After_Sales"],
          "PRODUCT_QUALITY": ["Overall_Quality", "Durability_Longevity", "Performance_Effectiveness"],
          "AUTHENTICITY": ["Originality", "Fake_Counterfeit"],
          "PRODUCT_APPEARANCE_DESIGN": ["Design_Style", "Color_Finish", "Size_Dimensions", "Comfort_Feeling"],
          "USABILITY_SETUP": ["Instructions_Manual", "Compatibility"]
        }
      },
      "domain_specific_aspects": {
        "ELECTRONICS_AND_GADGETS": {
          "categories": {
            "PERFORMANCE": [],
            "BATTERY": [],
            "DISPLAY": [],
            "CAMERA": ["Photo_Quality", "Video_Quality"],
            "BUILD_QUALITY": [],
            "AUDIO": [],
            "SOFTWARE": [],
            "STORAGE": [],
            "AUTHENTICITY": ["Originality", "Fake"]
          }
        },
        "FASHION_AND_APPAREL": {
          "categories": {
            "MATERIAL": [],
            "FIT_SIZE": [],
            "DESIGN": [],
            "DURABILITY": [],
            "FUNCTIONALITY": []
          }
        },
        "BEAUTY_AND_PERSONAL_CARE": {
          "categories": {
            "EFFICACY": [],
            "TEXTURE_SMELL": [],
            "SUITABILITY": [],
            "AUTHENTICITY": ["Originality"],
            "PACKAGING": []
          }
        },
        "HOME_LIVING_AND_APPLIANCES": {
          "categories": {
            "FUNCTIONALITY": ["Performance", "Power_Consumption", "Noise"],
            "BUILD_QUALITY": [],
            "AESTHETICS": [],
            "INSTALLATION": [],
            "DIMENSIONS": []
          }
        },
        "FOOD_AND_GROCERY": {
          "categories": {
            "TASTE_FLAVOR": [],
            "FRESHNESS": [],
            "QUALITY": ["Authenticity", "Purity", "Ingredients"],
            "EXPIRY": [],
            "PACKAGING": []
          }
        },
        "AUTOMOTIVE": {
          "categories": {
            "PERFORMANCE": [],
            "COMPATIBILITY": ["Fitment", "Model_Match"],
            "AUTHENTICITY": ["Originality"],
            "QUALITY": []
          }
        },
        "OTHER_PRODUCTS": {
          "description": "Fallback domain when product type is unclear or not covered above.",
          "categories": {
            "FEATURES_FUNCTIONALITY": [],
            "MATERIAL_BUILD": [],
            "SIZE_DIMENSIONS": [],
            "COMFORT": [],
            "SAFETY": [],
            "AUTHENTICITY": ["Originality", "Fake_Counterfeit"]
          }
        }
      }
    }
    """
taxonomy = json.loads(taxonomy_json_string)

# -----------------------
# Build mapping tables
# -----------------------
main_cats = set()
sub_to_main = {}

def add_categories(categories_dict: dict):
    for main_cat, subs in categories_dict.items():
        main_cats.add(main_cat)
        for sub in subs:
            sub_to_main[sub] = main_cat

add_categories(taxonomy["global_aspects"]["categories"])
for _, dom in taxonomy["domain_specific_aspects"].items():
    add_categories(dom.get("categories", {}))

main_cats_lc = {c.lower(): c for c in main_cats}
sub_to_main_lc = {k.lower(): v for k, v in sub_to_main.items()}

# -----------------------
# Parsers + normalizer
# -----------------------
def safe_parse_triplets(x):
    if pd.isna(x):
        return None
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return None

    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def to_main_category(cat):
    if cat is None or (isinstance(cat, float) and pd.isna(cat)):
        return cat
    s = str(cat).strip()
    if not s:
        return s

    # If "MAIN::SUB" style, take MAIN if it exists
    for sep in ("::", ":", "->", "/", "|"):
        if sep in s:
            main = s.split(sep, 1)[0].strip()
            if main in main_cats:
                return main
            mlc = main.lower()
            if mlc in main_cats_lc:
                return main_cats_lc[mlc]

    # If already main
    if s in main_cats:
        return s
    slc = s.lower()
    if slc in main_cats_lc:
        return main_cats_lc[slc]

    # If it's a known subcategory
    if s in sub_to_main:
        return sub_to_main[s]
    if slc in sub_to_main_lc:
        return sub_to_main_lc[slc]

    # Unknown: keep unchanged
    return s

def normalize_triplets(triplets):
    if not isinstance(triplets, list):
        return triplets
    for t in triplets:
        if isinstance(t, dict) and "aspect_category" in t:
            raw = t.get("aspect_category")
            t["aspect_category_raw"] = raw
            t["aspect_category"] = to_main_category(raw)
    return triplets

# -----------------------
# Apply
# -----------------------
COL = "absa_triplets_json"

parsed = df[COL].apply(safe_parse_triplets)
normalized = parsed.apply(normalize_triplets)

df[COL] = normalized.apply(
    lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
)

# -----------------------
# Quick check (optional)
# -----------------------
mapped = set()
unknown = set()
for v in normalized.dropna():
    if isinstance(v, list):
        for t in v:
            if isinstance(t, dict) and "aspect_category" in t:
                mapped.add(t["aspect_category"])
for x in mapped:
    if x not in main_cats:
        unknown.add(x)

print("Known main categories found:", sorted([x for x in mapped if x in main_cats]))
print("Unknown categories kept as-is (sample):", sorted(list(unknown))[:30])

df.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)