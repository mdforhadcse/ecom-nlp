import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# Config
# ----------------------------
INPUT_JSONL = "batch_tasks_responses_api.jsonl"   # your file path
OUTPUT_JSONL = "batch_output.jsonl"
ERROR_JSONL = "batch_errors.jsonl"

# Original dataset (the same one used by jsonl.py to build the JSONL)
INPUT_CSV = "data/processed/dataset_final_selected.csv"

# Final processed dataset
OUTPUT_CSV = "data/processed/dataset_final_selected_processed.csv"

# Optional: keep the raw parsed JSON in a column as well
SAVE_RAW_JSON = True

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment (.env). Set it before running.")
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Helper: download file content by file_id
# ----------------------------
def download_file(file_id: str, out_path: str) -> None:
    file_response = client.files.content(file_id)
    # In the OpenAI SDK, .content() returns a response-like object with .text
    content_text = file_response.text
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content_text)

# ----------------------------
# Step A: Upload JSONL for Batch
# ----------------------------
with open(INPUT_JSONL, "rb") as f:
    uploaded = client.files.create(file=f, purpose="batch")  # purpose must be "batch"
print("Uploaded file_id:", uploaded.id)

# ----------------------------
# Step B: Create Batch (Responses endpoint)
# ----------------------------
batch = client.batches.create(
    input_file_id=uploaded.id,
    endpoint="/v1/responses",
    completion_window="24h",
    metadata={"job": "absa_annotation"}
)
print("Batch id:", batch.id)
print("Batch status:", batch.status)

# ----------------------------
# Step C: Poll until done
# ----------------------------
while True:
    batch = client.batches.retrieve(batch.id)
    print("Status:", batch.status)

    if batch.status in ("completed", "failed", "expired", "cancelled"):
        break

    time.sleep(10)

# ----------------------------
# Step D: Download outputs (and errors)
# ----------------------------
if batch.status == "completed":
    if batch.output_file_id:
        download_file(batch.output_file_id, OUTPUT_JSONL)
        print("Saved output to:", OUTPUT_JSONL)
    else:
        print("No output_file_id found (unexpected). Check batch object.")

    if batch.error_file_id:
        download_file(batch.error_file_id, ERROR_JSONL)
        print("Saved errors to:", ERROR_JSONL)

elif batch.status in ("failed", "expired"):
    print("Batch ended with status:", batch.status)
    if batch.error_file_id:
        download_file(batch.error_file_id, ERROR_JSONL)
        print("Saved errors to:", ERROR_JSONL)
else:
    print("Batch ended with status:", batch.status)

# ----------------------------
# Step E (optional): Parse output lines and map by custom_id
# ----------------------------

def extract_assistant_output_text(response_body: dict) -> str | None:
    """Extract the assistant's output text from a Responses API response body."""
    if not isinstance(response_body, dict):
        return None

    for item in response_body.get("output", []) or []:
        if item.get("type") == "message" and item.get("role") == "assistant":
            for part in item.get("content", []) or []:
                if part.get("type") == "output_text" and part.get("text") is not None:
                    return part.get("text")
    return None


def safe_json_loads(s: str) -> dict | None:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


results_by_custom_id = {}

if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            custom_id = rec.get("custom_id")

            # Batch output line format includes `response` and `error`
            resp_wrapper = rec.get("response") or {}
            status_code = resp_wrapper.get("status_code")
            body = resp_wrapper.get("body")
            error = rec.get("error")

            results_by_custom_id[custom_id] = {
                "status_code": status_code,
                "body": body,
                "error": error,
            }

    print("Parsed results for custom_ids:", list(results_by_custom_id.keys()))
else:
    print(f"Output JSONL not found at {OUTPUT_JSONL}. Skipping parsing.")


# ----------------------------
# Step F: Merge batch outputs back into the original CSV and write a processed CSV
# ----------------------------

def custom_id_to_row_idx(custom_id: str) -> int | None:
    """Convert jsonl.py custom_id format (e.g., 'req-12') to row index 12."""
    if not custom_id or not isinstance(custom_id, str):
        return None
    if not custom_id.startswith("req-"):
        return None
    try:
        return int(custom_id.split("-", 1)[1])
    except Exception:
        return None


if results_by_custom_id:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"INPUT_CSV not found: {INPUT_CSV}. Make sure it matches the CSV used by jsonl.py"
        )

    df = pd.read_csv(INPUT_CSV)
    # Ensure row positions match jsonl.py enumerate() order
    df = df.reset_index(drop=True)

    # Ensure output columns exist
    for col in [
        "absa_overall_sentiment",
        "absa_overall_emotion",
        "absa_triplets_json",
        "absa_raw_text",
        "absa_status_code",
        "absa_error",
    ]:
        if col not in df.columns:
            df[col] = None

    filled = 0
    parse_fail = 0

    for custom_id, payload in results_by_custom_id.items():
        row_idx = custom_id_to_row_idx(custom_id)
        if row_idx is None or row_idx < 0 or row_idx >= len(df):
            continue

        status_code = payload.get("status_code")
        body = payload.get("body")
        error = payload.get("error")

        df.at[row_idx, "absa_status_code"] = status_code
        df.at[row_idx, "absa_error"] = json.dumps(error, ensure_ascii=False) if error else None

        # Extract the model's output text
        out_text = extract_assistant_output_text(body)
        df.at[row_idx, "absa_raw_text"] = out_text

        if not out_text:
            continue

        parsed_obj = safe_json_loads(out_text)
        if parsed_obj is None:
            parse_fail += 1
            continue

        # Your Pydantic schema (jsonl.py) emits: overall_sentiment, overall_emotion, triplets
        df.at[row_idx, "absa_overall_sentiment"] = parsed_obj.get("overall_sentiment")
        df.at[row_idx, "absa_overall_emotion"] = parsed_obj.get("overall_emotion")
        df.at[row_idx, "absa_triplets_json"] = json.dumps(
            parsed_obj.get("triplets", []), ensure_ascii=False
        )

        if SAVE_RAW_JSON:
            # Optionally keep the full parsed object, including top-level keys
            df.at[row_idx, "absa_raw_json"] = json.dumps(parsed_obj, ensure_ascii=False)

        filled += 1

    # If SAVE_RAW_JSON is enabled, ensure the column exists
    if SAVE_RAW_JSON and "absa_raw_json" not in df.columns:
        df["absa_raw_json"] = None

    # Write processed dataset
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved processed CSV to: {OUTPUT_CSV}")
    print(f"Rows filled: {filled} / {len(df)}")
    if parse_fail:
        print(f"Warning: {parse_fail} rows had non-JSON output_text and were not parsed.")