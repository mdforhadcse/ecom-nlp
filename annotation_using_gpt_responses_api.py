import json, logging, os
from typing import List, Literal

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load API key and instantiate OpenAI client
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment (.env). Set it before running.")
client = OpenAI(api_key=OPENAI_API_KEY)

# Define Pydantic models for structured parsing
class Triplet(BaseModel):
    aspect_category: str
    aspect_term: str
    opinion_term: str
    sentiment: Literal["Positive", "Negative", "Neutral"]

class ReviewAnnotation(BaseModel):
    overall_sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        ..., description="The overall sentiment of the entire review."
    )
    # Only five emotions are considered
    overall_emotion: Literal["Love", "Happy", "Anger", "Fear", "Sadness"] = Field(
        ..., description="The dominant emotion expressed in the review."
    )
    triplets: List[Triplet]

# Hierarchical taxonomy (unchanged from your previous version)
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

# Updated system prompt with overall sentiment/emotion instructions
# Updated system prompt incorporating the emotion hierarchy
system_prompt = f"""
You are an expert annotator for Bangla-English code-mixed e-commerce reviews.
Use the following hierarchical aspect taxonomy when labeling: {taxonomy_json_string}.

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

def build_user_prompt(review_text: str, product_name: str, product_cat: str) -> str:
    return (
        f"Review: {review_text}\n"
        "Annotate the review as instructed."
    )

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

def annotate_review(review_text: str, product_name: str, product_cat: str) -> dict | None:
    msg = build_user_prompt(review_text, product_name, product_cat)
    try:
        parsed_resp = client.responses.parse(
            model=OPENAI_MODEL,
            input=msg,
            instructions=system_prompt,
            text_format=ReviewAnnotation,
        )
        if parsed_resp.output_parsed is not None:
            return parsed_resp.output_parsed.model_dump()

        # Fallback to raw JSON if structured parsing fails
        raw_json = getattr(parsed_resp, "output_text", None)
        if raw_json:
            return json.loads(raw_json)
        out = getattr(parsed_resp, "output", None)
        if out:
            candidate = out[-1]
            content = getattr(candidate, "content", None)
            if content and len(content) > 0:
                raw_json_text = getattr(content[0], "text", None)
                if raw_json_text:
                    return json.loads(raw_json_text)
    except Exception as e:
        logging.error(f"Failed to annotate review: {e}")
    return None

# … your code above remains the same …

if __name__ == "__main__":
    # Load the dataset
    try:
        df = pd.read_csv("data/processed/dataset_final_selected.csv")
    except FileNotFoundError:
        df = pd.read_csv("dataset.csv")

    if "Review" not in df.columns:
        raise KeyError("The dataset does not contain a 'Review' column.")

    # Prepare lists to collect annotation results
    triplet_results, overall_sentiments, overall_emotions = [], [], []

    # Annotate only the first 10 rows for testing
    for _, row in df.head(5).iterrows():
        review_text = row.get("Review", "")
        product_name = row.get("Product Name", "")
        product_cat = row.get("Product Category", "")
        annotation = annotate_review(review_text, product_name, product_cat)
        print(f"\nReview - {_}:", review_text)
        print("Annotation:", annotation)
        if annotation:
            triplet_results.append(annotation.get("triplets", []))
            overall_sentiments.append(annotation.get("overall_sentiment"))
            overall_emotions.append(annotation.get("overall_emotion"))
        else:
            triplet_results.append([])
            overall_sentiments.append(None)
            overall_emotions.append(None)

    # >>> Fix: assign results only to the rows that were processed
    # Fix: assign results only to the rows that were processed, using pd.Series for lists
    n = len(triplet_results)
    df.loc[df.index[:n], "absa_triplets"] = pd.Series(triplet_results, index=df.index[:n])
    df.loc[df.index[:n], "overall_sentiment"] = overall_sentiments
    df.loc[df.index[:n], "overall_emotion"] = overall_emotions

    output_file = f"annotated_dataset_{OPENAI_MODEL}v2.csv"
    df.to_csv(output_file, index=False)
    print(f"Annotated dataset saved to {output_file}")