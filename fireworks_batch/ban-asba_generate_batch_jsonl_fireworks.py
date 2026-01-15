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
    aspect: Literal["Politics", "Sports", "Religion", "Others"] = Field(
        ..., description="The dominant emotion expressed in the review."
    )
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
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
    You are an expert language model trained to perform Aspect-Based Sentiment Analysis (ABSA)
    on Bengali text. You must analyze short Bengali comments and classify them into two labels:
    
    1. **Aspect** — The main topic or domain discussed in the comment.
       Choose only one from this fixed list:
       - Politics → Comments about government, leaders, elections, political parties, laws, corruption, or governance.
       - Sports → Comments about games, athletes, matches, scores, tournaments, or sports events.
       - Religion → Comments about faith, religious figures, rituals, beliefs, or religious communities.
       - Others → Any comment not fitting into the above three (e.g., entertainment, technology, general opinion).
    
    2. **Sentiment** — The emotional polarity expressed in the comment toward that aspect.
       Choose only one from this fixed list:
       - Positive → Shows praise, support, optimism, or satisfaction.
       - Negative → Shows criticism, anger, disapproval, or frustration.
       - Neutral → Factual statements, questions, or mixed/unclear opinions without strong emotion.
    
    ---
    
    ### Important Guidelines:
    - The text will be in Bengali. You must fully understand Bengali syntax, idioms, and sentiment.
    - Identify *the dominant* aspect and sentiment even if multiple topics appear.
    - If the sentiment is mixed but leans toward one side, choose that side (avoid Neutral unless truly balanced).
    - Always output your result **strictly** in valid JSON format, conforming exactly to this schema:
    
    ### Examples

    Example 1:
    Comment: "বাংলাদেশের সরকার দুর্নীতিতে ভরা।"
    → Output:
    {"aspect": "Politics", "sentiment": "Negative"}
    
    Example 2:
    Comment: "মেসির খেলা দেখে মন ভরে গেছে!"
    → Output:
    {"aspect": "Sports", "sentiment": "Positive"}
    
    Example 3:
    Comment: "আজকের নামাজটা খুব মনোযোগ দিয়ে পড়েছি।"
    → Output:
    {"aspect": "Religion", "sentiment": "Positive"}
    
    Example 4:
    Comment: "মোবাইলের নেটওয়ার্কটা ভালো না।"
    → Output:
    {"aspect": "Others", "sentiment": "Negative"}

    """

    # Paths to the dataset and output
    fireworks_model_name = "accounts/fireworks/models/gemma2-9b-it"
    input_path = Path("../demo/BAN-ABSA.csv")
    output_path = Path(f"ban-absa_fireworks_batch_tasks_{fireworks_model_name.split('/')[-1]}.jsonl")

    # Generate the batch requests; set include_product_meta=True if you want product metadata in the input
    build_batch_requests(
        input_csv=input_path,
        output_jsonl=output_path,
        system_instruction_text=system_prompt,
        model_name=fireworks_model_name,
        # include_product_meta=True,
    )