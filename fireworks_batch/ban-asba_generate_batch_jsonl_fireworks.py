import json
from pathlib import Path
from typing import List, Literal, Type

import pandas as pd
from pydantic import BaseModel, Field


# # ------------------------------------------------------------------------------
# # 1. DEFINE PYDANTIC MODELS
# # ------------------------------------------------------------------------------
# class Triplet(BaseModel):
#     """Data class describing a single aspect–opinion–sentiment entry."""
#     aspect_category: str
#     aspect_term: str
#     opinion_term: str
#     sentiment: Literal["Positive", "Negative", "Neutral"]
#

class ReviewAnnotation(BaseModel):
    aspect: Literal["politics", "sports", "religion", "other"] = Field(
        ..., description="The dominant aspect category discussed in the review."
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ..., description="The overall sentiment of the entire review."
    )


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
    # Load data
    df = pd.read_csv(input_csv)
    # Generate a strict JSON schema from the ReviewAnnotation model
    response_format_obj = get_strict_json_schema(ReviewAnnotation)

    # Ensure the output directory exists
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Build the JSONL lines
    lines_written = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            review_text = str(row.get("post", "")).strip()
            if not review_text:
                # Skip empty reviews
                continue

            # Construct the user input text; optionally include product metadata
            user_input = (
                    f"Review: {review_text}\n"
                    "Annotate the review as instructed."
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

    system_prompt = """
    You are an expert Bengali Aspect-Based Sentiment Analysis (ABSA) classifier.
    
    Task: Given one Bengali comment, output exactly TWO labels:
    1) aspect
    2) sentiment
    
    You MUST output only valid JSON with exactly these keys:
    {"aspect": "...", "sentiment": "..."}
    
    -----------------------------------
    ASPECT (choose exactly ONE):
    - "politics"  → government, leaders, elections, parties, law, corruption, governance, national issues
    - "sports"    → games, players, teams, matches, scores, tournaments, sports events
    - "religion"  → Allah/Islam/namaz/Quran/prophet, Hindu/Buddhist/Christian faith, rituals, religious groups
    - "other"     → anything else (technology, entertainment, daily life, general talk, unclear topic)
    
    If multiple topics appear, choose the DOMINANT one (main focus of the comment).
    
    -----------------------------------
    SENTIMENT (choose exactly ONE):
    - "positive" → praise, support, happiness, admiration, satisfaction, optimism
    - "negative" → criticism, anger, hate, sadness, mockery, frustration, disappointment
    - "neutral"  → factual statement, question, name-only, unclear, mixed with no clear leaning
    
    Rules:
    - Do NOT output anything except JSON.
    - Do NOT add extra keys or explanations.
    - Detect implicit sentiment and sarcasm (e.g., “বাহ বাহ সরকার!” often means Negative).
    - If sentiment is mixed but clearly leans to one side, choose that side (avoid Neutral unless truly unclear).
    
    Examples:
    Input: "বাংলাদেশের সরকার দুর্নীতিতে ভরা।"
    Output: {"aspect":"politics","sentiment":"negative"}
    
    Input: "মেসির খেলা দেখে মন ভরে গেছে!"
    Output: {"aspect":"sports","sentiment":"positive"}
    
    Input: "আজকের নামাজটা খুব মনোযোগ দিয়ে পড়েছি।"
    Output: {"aspect":"religion","sentiment":"positive"}
    
    Input: "মোবাইলের নেটওয়ার্কটা ভালো না।"
    Output: {"aspect":"other","sentiment":"negative"}
    """

    # Paths to the dataset and output
    fireworks_model_name = "accounts/fireworks/models/gemma2-9b-it"
    input_path = Path("../demo/BAN-ABSA_balanced_1350.csv")
    output_path = Path(f"ban-absa_fireworks_batch_tasks_{fireworks_model_name.split('/')[-1]}.jsonl")

    # Generate the batch requests; set include_product_meta=True if you want product metadata in the input
    build_batch_requests(
        input_csv=input_path,
        output_jsonl=output_path,
        system_instruction_text=system_prompt,
        model_name=fireworks_model_name,
        # include_product_meta=True,
    )