import json
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd


# ----------------------------
# Config (edit as needed)
# ----------------------------
# Input CSV (the one that contains Review + absa_raw_json / absa_raw_text)
INPUT_CSV = Path("../gold/train_700.csv")

# Output JSONL (fine-tuning dataset)
OUTPUT_JSONL = Path("fine_tune_dataset.jsonl")

# Prefer structured JSON assistant output if available
ASSISTANT_COL_PREFERENCE: List[str] = ["absa_raw_json", "absa_raw_text"]

# Optional: include product metadata if your CSV has these columns
INCLUDE_PRODUCT_META = False

# For Gemma-like fine-tunes where `system` role may be unsupported by your trainer,
# set this True to prepend the system instructions into the user content and
# write only (user, assistant) messages.
MERGE_SYSTEM_INTO_USER = True

# ----------------------------
# Core builder
# ----------------------------

def _pick_assistant_value(row: pd.Series, columns: List[str]) -> Optional[Any]:
    for col in columns:
        if col not in row.index:
            continue
        v = row.get(col)
        if v is None:
            continue
        # NaN
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        s = str(v).strip()
        if s:
            return v
    return None


def _normalize_assistant_content(v: Any) -> str:
    # If it's already a dict/list, dump as compact JSON
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)

    s = str(v).strip()
    if not s:
        return ""

    # If it looks like JSON, parse and re-dump for cleanliness
    if s.startswith("{") or s.startswith("["):
        try:
            parsed = json.loads(s)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return s

    return s


def build_finetune_jsonl(
    input_csv: Path,
    output_jsonl: Path,
    system_instruction_text: str,
    include_product_meta: bool,
    assistant_col_preference: List[str],
    merge_system_into_user: bool,
) -> None:
    df = pd.read_csv(input_csv)

    if "Review" not in df.columns:
        raise KeyError("Input CSV must contain a 'Review' column.")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with output_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            review = str(row.get("Review", "")).strip()
            if not review:
                skipped += 1
                continue

            # Build user content (same high-level shape as your batch pipeline)
            if include_product_meta:
                product_name = str(row.get("Product Name", "")).strip()
                product_cat = str(row.get("Product Category", "")).strip()
                user_content = (
                    f"Product: {product_name}\n"
                    f"Category: {product_cat}\n"
                    f"Review: {review}\n"
                    "Annotate the review as instructed."
                )
            else:
                user_content = (
                    f"Review: {review}\n"
                    "Annotate the review as instructed."
                )

            assistant_val = _pick_assistant_value(row, assistant_col_preference)
            if assistant_val is None:
                skipped += 1
                continue

            assistant_content = _normalize_assistant_content(assistant_val)
            if not assistant_content:
                skipped += 1
                continue

            if merge_system_into_user:
                # For trainers/models that don't support system role (e.g., Gemma-style)
                combined_user = f"{system_instruction_text.strip()}\n\n---\n\n{user_content}".strip()
                record = {
                    "messages": [
                        {"role": "user", "content": combined_user},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            else:
                record = {
                    "messages": [
                        {"role": "system", "content": system_instruction_text.strip()},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} lines to {output_jsonl} (skipped {skipped}).")


def main() -> None:

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

    system_prompt = f"""
    You are an expert annotator for Bangla-English code-mixed e-commerce reviews.
    Use the following hierarchical aspect taxonomy when labeling: {taxonomy_json_string}

    Your tasks are:
    1. Identify all (aspect_term, opinion_term, sentiment) triplets in the review.
       - aspect_term: the specific noun or phrase in the text that refers to a product, service or attribute.
       - opinion_term: the phrase expressing sentiment about that aspect.
       - sentiment: Positive, Negative or Neutral, based on the opinion.
       Map aspect_term to the most specific sub-category in the taxonomy; if none fit, use a parent category.

    2. Determine the overall_sentiment of the entire review.  Classify it as one of:
       Positive, Negative, or Neutral.

    3. Determine the overall_emotion expressed in the review.  Use the following emotion classes and their subordinate cues:
       - Love: adoration, affection, fondness, liking, attraction, caring, tenderness, compassion, sentimentality, arousal, desire, lust, passion, infatuation, longing.
       - Happy: amusement, bliss, cheerfulness, gaiety, glee, jolliness, joviality, joy, delight, enjoyment, gladness, jubilation, elation, satisfaction, ecstasy, euphoria, enthusiasm, zeal, zest, excitement, thrill, exhilaration, contentment, pleasure, pride, triumph, eagerness, hope, optimism, enthrallment, rapture, relief.
       - Anger: aggravation, irritation, agitation, annoyance, grouchiness, exasperation, frustration, anger, rage, outrage, fury, wrath, hostility, ferocity, bitterness, hate, loathing, scorn, spite, vengefulness, dislike, resentment, disgust, revulsion, contempt, envy, jealousy, torment.
       - Fear: alarm, shock, fright, horror, terror, panic, hysteria, mortification, anxiety, nervousness, tenseness, uneasiness, apprehension, worry, distress, dread.
       - Sadness: agony, suffering, hurt, anguish, depression, despair, hopelessness, gloom, glumness, unhappiness, grief, sorrow, woe, misery, melancholy, dismay, disappointment, displeasure, guilt, shame, regret, remorse, alienation, isolation, neglect, loneliness, rejection, homesickness, defeat, insecurity, embarrassment, humiliation, insult, pity, sympathy.

    Match the predominant emotional cues in the review to these lists and output one of: Love, Happy, Anger, Fear, or Sadness.

    Output a JSON object with exactly three keys:
    - "overall_sentiment": the overall sentiment label (a string).
    - "overall_emotion": the dominant emotion label (a string).
    - "triplets": a list of objects, each containing the keys "aspect_category", "aspect_term", "opinion_term", and "sentiment".

    Include no extra keys or comments in the output.
    """


    build_finetune_jsonl(
        input_csv=INPUT_CSV,
        output_jsonl=OUTPUT_JSONL,
        system_instruction_text=system_prompt,
        include_product_meta=INCLUDE_PRODUCT_META,
        assistant_col_preference=ASSISTANT_COL_PREFERENCE,
        merge_system_into_user=MERGE_SYSTEM_INTO_USER,
    )


if __name__ == "__main__":
    main()