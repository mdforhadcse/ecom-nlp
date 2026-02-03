
# Helper: normalize model id for batch jobs
def normalize_model_id_for_batch(model_id: str) -> str:
    """BatchInferenceJobs expects accounts/<account-id>/models/<model-id>.

    The deployment UI often shows accounts/<account-id>/deployedModels/<deployment-id>.
    If a deployedModels path is provided, convert it to the models path.
    """
    if not isinstance(model_id, str):
        return model_id
    s = model_id.strip()
    # Convert deployedModels -> models for batch jobs
    if "/deployedModels/" in s:
        return s.replace("/deployedModels/", "/models/", 1)
    return s

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# ----------------------------
# Config
# ----------------------------

# Path to the JSONL file generated for Fireworks batch inference.  Each
# line should contain a ``custom_id`` and a ``body`` field matching the
# Fireworks API requirements.
INPUT_JSONL = "fireworks_batch_tasks_deepseek-r1-0528-distill-qwen3-8b.jsonl"
INPUT_CSV = "../gold/test_300.csv"
MODEL_ID = "accounts/fireworks/models/deepseek-r1-0528-distill-qwen3-8b"

# Names for the Fireworks datasets and job. These must be unique within
# your account. If you rerun with the same IDs, Fireworks will reject the
# job because the output dataset already exists.
#
# Tip: set FW_RUN_TAG in your .env to keep runs reproducible, otherwise a
# timestamp-based tag is used.
RUN_TAG = time.strftime("%Y%m%d-%H%M%S")

INPUT_DATASET_ID = f"input-dataset-{RUN_TAG}"
OUTPUT_DATASET_ID = f"output-dataset-{RUN_TAG}"
BATCH_JOB_ID = f"batch-job-{RUN_TAG}"



# Inference parameters: adjust as needed.  ``max_tokens`` limits the
# length of the generated output; ``temperature`` controls randomness.
INFERENCE_PARAMS = {
    "maxTokens": 256,
    "temperature": 0.0,
    "topP": 1.0,
    "n": 1
}
# Final processed dataset path. The script writes the merged CSV with
# additional columns for Fireworks outputs.
OUTPUT_CSV = f"dataset_{normalize_model_id_for_batch(MODEL_ID).split('/')[-1]}_fireworks_processed_{RUN_TAG}.csv"

# Optional: keep the raw parsed JSON in a column as well
SAVE_RAW_JSON = True


# Poll interval (seconds) when checking batch job status
POLL_INTERVAL = 10

# Max number of error records to print when a batch job fails
MAX_ERROR_LINES_TO_PRINT = 20

# Terminal job states Fireworks may return
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "EXPIRED",
    "CANCELLED",
    "JOB_STATE_COMPLETED",
    "JOB_STATE_FAILED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_CANCELLED",
}

# ----------------------------
# Environment & API setup
# ----------------------------

load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_ACCOUNT_ID = os.getenv("FIREWORKS_ACCOUNT_ID")

if not FIREWORKS_API_KEY:
    raise RuntimeError(
        "Missing FIREWORKS_API_KEY in environment (.env). Set it before running."
    )
if not FIREWORKS_ACCOUNT_ID:
    raise RuntimeError(
        "Missing FIREWORKS_ACCOUNT_ID in environment (.env). Set it before running."
    )

# Base URL for Fireworks API
BASE_URL = "https://api.fireworks.ai/v1"

# Convenience header for authenticated requests
AUTH_HEADERS = {
    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    "Content-Type": "application/json",
}


def create_dataset(dataset_id: str) -> None:
    """Create a new dataset entry on Fireworks.

    If the dataset already exists, Fireworks returns OK and this call is
    idempotent.  Only ``userUploaded`` needs to be present for datasets
    created from client data.
    """
    url = f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/datasets"
    payload = {
        "datasetId": dataset_id,
        "dataset": {"userUploaded": {}},
    }
    resp = requests.post(url, headers=AUTH_HEADERS, data=json.dumps(payload))
    if not resp.ok:
        raise RuntimeError(
            f"Failed to create dataset {dataset_id}: {resp.status_code} {resp.text}"
        )
    print(f"Created dataset '{dataset_id}' (or already exists)")


def upload_dataset_file(dataset_id: str, jsonl_path: Path) -> None:
    """Upload a JSONL file to an existing dataset via multipart upload.

    Fireworks allows uploading files up to 150MB via the simple upload
    endpoint.  Larger files require using ``:getUploadEndpoint`` and
    presigned URLs, but typical batch workloads will fit within the limit.
    """
    url = (
        f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/datasets/{dataset_id}:upload"
    )
    with jsonl_path.open("rb") as f:
        files = {"file": (jsonl_path.name, f, "application/jsonl")}
        headers = {"Authorization": AUTH_HEADERS["Authorization"]}
        resp = requests.post(url, headers=headers, files=files)
    if not resp.ok:
        raise RuntimeError(
            f"Failed to upload dataset file {jsonl_path} to {dataset_id}: {resp.status_code} {resp.text}"
        )
    print(f"Uploaded {jsonl_path} to dataset '{dataset_id}' ({resp.json().get('bytes')} bytes)")


def create_batch_job(
    job_id: str,
    model_id: str,
    input_dataset_id: str,
    output_dataset_id: str,
    inference_params: Dict[str, Any],
) -> None:
    """Launch a batch inference job on Fireworks.

    The job will process all requests in the input dataset asynchronously and
    write the outputs to a newly-created output dataset (it must not already exist).  ``job_id`` must be unique
    within your account.  If a job with the same ID already exists, the
    call will fail.  See Fireworks docs for details on inference parameters.
    """
    url = (
        f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/batchInferenceJobs"
        f"?batchInferenceJobId={job_id}"
    )
    payload = {
        "model": normalize_model_id_for_batch(model_id),
        "inputDatasetId": f"accounts/{FIREWORKS_ACCOUNT_ID}/datasets/{input_dataset_id}",
        "outputDatasetId": f"accounts/{FIREWORKS_ACCOUNT_ID}/datasets/{output_dataset_id}",
        "inferenceParameters": inference_params,
    }
    resolved_model = payload["model"]
    if resolved_model != model_id:
        print(f"Note: normalized MODEL_ID for batch job: {model_id} -> {resolved_model}")
    resp = requests.post(url, headers=AUTH_HEADERS, data=json.dumps(payload))
    if not resp.ok:
        raise RuntimeError(
            f"Failed to create batch job {job_id}: {resp.status_code} {resp.text}"
        )
    print(f"Created batch job '{job_id}' for model '{model_id}'")



def get_batch_job(job_id: str) -> Dict[str, Any]:
    """Retrieve the full batch inference job object."""
    url = f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/batchInferenceJobs/{job_id}"
    resp = requests.get(url, headers=AUTH_HEADERS)
    if not resp.ok:
        raise RuntimeError(
            f"Failed to get batch job: {resp.status_code} {resp.text}"
        )
    return resp.json()


def get_batch_job_status(job_id: str) -> str:
    """Retrieve the current state/status of the batch inference job."""
    data = get_batch_job(job_id)
    # Fireworks returns either a `state` or `status` field depending on API version.
    return data.get("state") or data.get("status") or "UNKNOWN"


def download_output_dataset(dataset_id: str, output_dir: Path) -> Dict[str, Path]:
    """Download all files from an output dataset using signed URLs.

    Returns a mapping from the object name to the local file path.
    """
    url = (
        f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/datasets/{dataset_id}:getDownloadEndpoint"
    )
    resp = requests.get(url, headers=AUTH_HEADERS)
    if not resp.ok:
        raise RuntimeError(
            f"Failed to get download endpoints for dataset {dataset_id}: {resp.status_code} {resp.text}"
        )
    data = resp.json()
    signed_urls: Dict[str, str] = data.get("filenameToSignedUrls", {})
    if not signed_urls:
        raise RuntimeError(
            f"No signed URLs returned for dataset {dataset_id}. Did the job produce any output?"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    local_paths: Dict[str, Path] = {}
    for object_path, signed_url in signed_urls.items():
        filename = os.path.basename(object_path)
        local_path = output_dir / filename
        print(f"Downloading {filename}…")
        with requests.get(signed_url, stream=True) as r:
            r.raise_for_status()
            with local_path.open("wb") as out:
                for chunk in r.iter_content(chunk_size=8192):
                    out.write(chunk)
        local_paths[filename] = local_path
    return local_paths


def extract_output_text(response_body: Dict[str, Any]) -> Optional[str]:
    """Extract the assistant's output text from a Responses API response body.

    Fireworks responses may include either an ``output`` field (similar to
    OpenAI) or a ``choices`` field (OpenAI‑style) depending on the model.
    This helper attempts to locate the generated JSON string in either
    structure.  Returns ``None`` if nothing was found.
    """
    if not isinstance(response_body, dict):
        return None
    # Case 1: Fireworks structured output (matches OpenAI responses.parse)
    if "output" in response_body:
        for item in response_body.get("output", []) or []:
            if item.get("type") == "message" and item.get("role") == "assistant":
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text" and part.get("text"):
                        return part.get("text")
    # Case 2: OpenAI‑style chat completions format
    choices = response_body.get("choices")
    if choices and isinstance(choices, list):
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        if message and isinstance(message, dict):
            return message.get("content")
    return None


def safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    """Safely parse a JSON string into a dict.  Returns None if parsing fails."""
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def main() -> None:
    # Ensure input files exist
    input_jsonl_path = Path(INPUT_JSONL)
    if not input_jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {INPUT_JSONL}")
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    # Step 1: Create the input dataset.
    # NOTE: For Fireworks batch inference, the `outputDatasetId` must NOT already exist
    # when creating the batch job. The batch job creation call will create the output
    # dataset for you.
    create_dataset(INPUT_DATASET_ID)

    # Step 2: Upload input JSONL to the input dataset
    upload_dataset_file(INPUT_DATASET_ID, input_jsonl_path)

    # Step 3: Launch the batch inference job
    create_batch_job(
        BATCH_JOB_ID,
        MODEL_ID,
        INPUT_DATASET_ID,
        OUTPUT_DATASET_ID,
        INFERENCE_PARAMS,
    )

    # Step 4: Poll job status until completed or failed
    print(f"Polling job '{BATCH_JOB_ID}' status…")
    while True:
        state = get_batch_job_status(BATCH_JOB_ID)
        print(f"Current job state: {state}")
        if state in TERMINAL_STATES:
            break
        time.sleep(POLL_INTERVAL)

    if state not in ("COMPLETED", "JOB_STATE_COMPLETED"):
        # Pull full job details for debugging
        job = get_batch_job(BATCH_JOB_ID)
        print("\nBatch job did not complete successfully. Job details:")
        print(json.dumps(job, ensure_ascii=False, indent=2))

        # Try to download and print a few error records from the output dataset (if present)
        try:
            download_dir = Path(f"fireworks_batch_outputs_{normalize_model_id_for_batch(MODEL_ID).split('/')[-1]}")
            result_files = download_output_dataset(OUTPUT_DATASET_ID, download_dir)
            # Prefer files with 'error' in the filename
            error_candidates = [
                p for name, p in result_files.items()
                if name.lower().endswith(".jsonl") and "error" in name.lower()
            ]
            if error_candidates:
                err_path = error_candidates[0]
                print(f"\nSample errors from: {err_path}")
                with err_path.open("r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= MAX_ERROR_LINES_TO_PRINT:
                            break
                        if line.strip():
                            print(line.rstrip())
            else:
                print("\nNo error JSONL file found in output dataset download.")
        except Exception as e:
            print(f"\nCould not download/print error file: {e}")

        raise RuntimeError(
            f"Batch job ended with state '{state}'. See job details and sample errors above."
        )

    print(f"Batch job '{BATCH_JOB_ID}' completed successfully.")

    # Step 5: Download results from the output dataset
    download_dir = Path("fireworks_batch_outputs")
    result_files = download_output_dataset(OUTPUT_DATASET_ID, download_dir)
    print(f"Downloaded files: {list(result_files.keys())}")

    # Identify results and errors files using Fireworks conventions.
    results_path: Optional[Path] = None
    errors_path: Optional[Path] = None

    for name, path in result_files.items():
        lower = name.lower()

        # Fireworks commonly returns `BIJOutputSet.jsonl` for successes
        if lower.endswith(".jsonl") and "bijoutputset" in lower:
            results_path = path
            continue

        # Fireworks commonly returns `error-data` (no .jsonl extension) for errors
        if lower == "error-data" or lower.endswith("error-data"):
            errors_path = path
            continue

        # Fallback heuristics
        if lower.endswith(".jsonl"):
            if "error" in lower or "failure" in lower:
                errors_path = path
            else:
                if results_path is None:
                    results_path = path

    if not results_path:
        raise RuntimeError("Could not find a results JSONL file in the downloaded dataset.")

    # Step 6: Parse results and map them back to the original CSV
    results_by_custom_id: Dict[str, Dict[str, Any]] = {}

    # ---- Parse success results (BIJOutputSet.jsonl) ----
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            custom_id = rec.get("custom_id") or rec.get("request_id") or rec.get("customId")
            if custom_id is None:
                continue
            custom_id = str(custom_id).strip()

            # Fireworks BIJOutputSet records typically store the chat completion response in `response`
            body = rec.get("response")
            if body is None:
                body = rec.get("body")

            results_by_custom_id[custom_id] = {
                "status_code": 200,
                "body": body,
                "error": None,
            }

    # ---- Parse errors (error-data) ----
    if errors_path and errors_path.exists():
        with errors_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                custom_id = rec.get("custom_id") or rec.get("request_id") or rec.get("customId")
                if custom_id is None:
                    continue
                custom_id = str(custom_id).strip()

                err = rec.get("error") or {}
                code = err.get("code")

                results_by_custom_id[custom_id] = {
                    "status_code": code,
                    "body": None,
                    "error": err,
                }

    print(f"Parsed {len(results_by_custom_id)} results from batch output.")

    # Step 7: Load the original CSV and merge the results
    df = pd.read_csv(INPUT_CSV).reset_index(drop=True)
    # Ensure result columns exist
    for col in [
        "fw_overall_sentiment",
        "fw_overall_emotion",
        "fw_triplets_json",
        "fw_raw_text",
        "fw_status_code",
        "fw_error",
    ]:
        if col not in df.columns:
            df[col] = None

    filled = 0
    parse_fail = 0
    for custom_id, payload in results_by_custom_id.items():
        # custom_id is expected to be of the form 'req-<row_index>'
        if not isinstance(custom_id, str) or not custom_id.startswith("req-"):
            continue
        try:
            row_idx = int(custom_id.split("-", 1)[1])
        except Exception:
            continue
        if row_idx < 0 or row_idx >= len(df):
            continue
        status_code = payload.get("status_code")
        body = payload.get("body")
        error = payload.get("error")
        df.at[row_idx, "fw_status_code"] = status_code
        df.at[row_idx, "fw_error"] = json.dumps(error, ensure_ascii=False) if error else None

        out_text = extract_output_text(body)
        df.at[row_idx, "fw_raw_text"] = out_text
        if not out_text:
            continue
        parsed_obj = safe_json_parse(out_text)
        if parsed_obj is None:
            parse_fail += 1
            continue
        df.at[row_idx, "fw_overall_sentiment"] = parsed_obj.get("overall_sentiment")
        df.at[row_idx, "fw_overall_emotion"] = parsed_obj.get("overall_emotion")
        df.at[row_idx, "fw_triplets_json"] = json.dumps(
            parsed_obj.get("triplets", []), ensure_ascii=False
        )
        if SAVE_RAW_JSON:
            if "fw_raw_json" not in df.columns:
                df["fw_raw_json"] = None
            df.at[row_idx, "fw_raw_json"] = json.dumps(parsed_obj, ensure_ascii=False)
        filled += 1

    # Save processed CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(
        f"Saved processed dataset with Fireworks results to: {OUTPUT_CSV}\n"
        f"Rows filled: {filled} / {len(df)}; parse failures: {parse_fail}"
    )


if __name__ == "__main__":
    main()