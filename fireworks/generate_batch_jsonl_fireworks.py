import json
from pathlib import Path
from typing import List, Literal, Type

import pandas as pd
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1. DEFINE PYDANTIC MODELS
# ------------------------------------------------------------------------------
class Triplet(BaseModel):
    """Data class describing a single aspect–opinion–sentiment entry."""
    aspect_category: str
    aspect_term: str
    opinion_term: str
    sentiment: Literal["Positive", "Negative", "Neutral"]


class ReviewAnnotation(BaseModel):
    """Data class describing the full annotation for a review."""
    overall_sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        ..., description="The overall sentiment of the entire review."
    )
    overall_emotion: Literal[
        "Love",
        "Happy",
        "Anger",
        "Fear",
        "Sadness",
    ] = Field(
        ..., description="The dominant emotion expressed in the review."
    )
    triplets: List[Triplet]


# ------------------------------------------------------------------------------
# 2. HELPER: STRICT JSON SCHEMA GENERATOR
# ------------------------------------------------------------------------------
def get_strict_json_schema(model_class: Type[BaseModel]) -> dict:
    """
    Generate a strict JSON schema for a Pydantic model.

    The helper recursively modifies the schema such that all object types
    disallow additional properties and require all defined properties.
    The returned dictionary is ready to be embedded as the `response_format` field
    of a Fireworks/OpenAI Chat Completions or Batch Inference request.

    Parameters
    ----------
    model_class : Type[BaseModel]
        The Pydantic model class to generate a schema for.

    Returns
    -------
    dict
        A dictionary describing the strict JSON response_format object for Chat Completions
        and Batch Inference. The top-level keys are `type` and `json_schema` (containing
        `name`, `schema`, and `strict`).
    """
    # Derive the JSON Schema from the Pydantic model
    schema = model_class.model_json_schema()

    # Recursively enforce strictness on objects
    def make_strict_recursive(node):
        """Recursively enforce strictness across the entire JSON schema tree.

        The previous implementation only recursed into a small, fixed set of keys.
        That misses most nested object schemas in Pydantic output (e.g. properties
        dicts and $defs entries like Triplet), so additionalProperties stays allowed
        in many places.
        """
        if isinstance(node, dict):
            # If this node is an object schema, enforce strict object rules
            if node.get("type") == "object":
                node["additionalProperties"] = False
                props = node.get("properties")
                if isinstance(props, dict):
                    node["required"] = list(props.keys())

            # Recurse through *all* dict values so we don't miss nested schemas
            for value in node.values():
                if isinstance(value, (dict, list)):
                    make_strict_recursive(value)

        elif isinstance(node, list):
            for item in node:
                make_strict_recursive(item)

    make_strict_recursive(schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "review_annotation_schema",
            "schema": schema,
            "strict": True,
        },
    }


# ------------------------------------------------------------------------------
# 3. MAIN FUNCTION TO BUILD BATCH REQUESTS
# ------------------------------------------------------------------------------
def build_batch_requests(
    input_csv: Path,
    output_jsonl: Path,
    system_instruction_text: str,
    model_name: str,
    include_product_meta: bool = False,
) -> None:
    """
    Generate a JSONL file for Fireworks Batch Inference.

    Parameters
    ----------
    input_csv : Path
        Path to the CSV file containing review data.  The file must contain a
        'Review' column and may optionally contain 'Product Name' and
        'Product Category' columns if `include_product_meta` is True.
    output_jsonl : Path
        Path to write the generated JSONL file.  Parent directories will be
        created as needed.
    system_instruction_text : str
        Instructions to guide the model (taxonomy + rules). For models that support the system role, this is sent as a system message.
        For Gemma instruct models (no system role), these instructions are prepended to the user message.
    model_name : str, optional
        Which OpenAI/Fireworks model to use for the batch job. This value is NOT written into
        each JSONL record; the model is set when creating the batch job.
    include_product_meta : bool, optional
        Whether to include product name and category in the user input.  When
        True, `Product Name` and `Product Category` columns from the CSV will
        be appended to the input string.  Default is False.
    temperature : float, optional
        Sampling temperature for the model.  A low value (e.g. 0.0) results in
        deterministic output.
    max_output_tokens : int, optional
        Upper limit on the length of the generated response.  The actual
        tokens consumed will depend on your schema and prompt size.
    """
    # Load data
    df = pd.read_csv(input_csv)
    if "Review" not in df.columns:
        raise KeyError("The input CSV must contain a 'Review' column.")

    # Generate a strict JSON schema from the ReviewAnnotation model
    response_format_obj = get_strict_json_schema(ReviewAnnotation)

    # Ensure the output directory exists
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Build the JSONL lines
    lines_written = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            review_text = str(row.get("Review", "")).strip()
            if not review_text:
                # Skip empty reviews
                continue

            # Construct the user input text; optionally include product metadata
            if include_product_meta:
                product_name = str(row.get("Product Name", "")).strip()
                product_cat = str(row.get("Product Category", "")).strip()
                user_input = (
                    f"Product: {product_name}\n"
                    f"Category: {product_cat}\n"
                    f"Review: {review_text}\n"
                    "Annotate this review."
                )
            else:
                user_input = (
                    f"Review: {review_text}\n"
                    "Annotate this review."
                )

            # Assemble the body for Fireworks Batch Inference (Chat Completions style)
            # NOTE: Do NOT include `model` here; the model is set when creating the batch job.
            # NOTE: Gemma instruct models do NOT support the `system` role.
            model_lc = (model_name or "").lower()
            gemma_like = "gemma" in model_lc

            if gemma_like:
                combined_user = f"{system_instruction_text.strip()}\n\n---\n\n{user_input}".strip()
                messages = [{"role": "user", "content": combined_user}]
            else:
                messages = [
                    {"role": "system", "content": system_instruction_text},
                    {"role": "user", "content": user_input},
                ]

            body = {
                "messages": messages,
                # Enforce structured JSON output
                "response_format": response_format_obj,
            }

            # Build the per‑request object
            request_obj = {
                "custom_id": f"req-{idx}",
                "body": body,
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            lines_written += 1

    print(f"Wrote {lines_written} request lines to {output_jsonl}")


if __name__ == "__main__":
    # Example usage of this script.  For real runs, you might parse
    # command‑line arguments instead of hard‑coding paths and parameters.
    # Define the system instructions (system prompt) for the annotation task
    # This should include your taxonomy and detailed guidelines on triplet
    # extraction, sentiment classification and emotion mapping.

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

    # Paths to the dataset and output
    fireworks_model_name = "accounts/fireworks/models/gemma2-9b-it"
    input_path = Path("../gold/dataset_final_selected_processed_part.csv")
    output_path = Path(f"fireworks_batch_tasks_responses_api_{fireworks_model_name.split('/')[-1]}.jsonl")

    # Generate the batch requests; set include_product_meta=True if you want product metadata in the input
    build_batch_requests(
        input_csv=input_path,
        output_jsonl=output_path,
        system_instruction_text=system_prompt,
        model_name=fireworks_model_name,
        # include_product_meta=True,
    )