import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# ----------------------------
# Config (edit these)
# ----------------------------

# This MUST be a JSONL file where each line is: {"custom_id": "req-0", "body": {...}}
# (i.e., Fireworks Batch input JSONL format)
INPUT_JSONL = "fireworks_batch/fireworks_batch_tasks_llama-v3p1-8b-instruct.jsonl"

# Optional: also merge Batch output JSONL back into a source CSV
# (Set to your real path if you want CSV output)
INPUT_CSV = "../gold/dataset_1k_test_200.csv"

# These are the filenames commonly produced by Fireworks batch output datasets.
# If your output uses different names, update these.
BIJ_OUTPUT_FILENAME = "BIJOutputSet.jsonl"
ERROR_OUTPUT_FILENAME = "error-data"  # optional if exists

# Output CSV name (will be saved inside the per-run output folder)
OUTPUT_CSV_NAME = "dataset_ft-llama-v3p1-8b.csv"

# Batch API base URL (NOT the realtime inference base)
BASE_URL = "https://api.fireworks.ai/v1"

# You can paste either a deployed model or a model resource.
# - Deployed (works for realtime chat, NOT directly for batch):
#   accounts/<account-id>/deployedModels/<deploy-id>
# - Model resource (what batch wants):
#   accounts/<account-id>/models/<model-id>
MODEL_ID = "accounts/forhad1822-iw05l8ae4/deployedModels/ft-llama-v3p1-8b-instruct-bmrqxd5r"

# Batch inference parameters
INFERENCE_PARAMS = {
    "maxTokens": 512,
    "temperature": 0.0,
    "topP": 1.0,
    "n": 1,
}

# Unique ids per run
RUN_TAG = time.strftime("%Y%m%d-%H%M%S")
INPUT_DATASET_ID = f"input-dataset-{RUN_TAG}"
OUTPUT_DATASET_ID = f"output-dataset-{RUN_TAG}"
BATCH_JOB_ID = f"batch-job-{RUN_TAG}"

POLL_INTERVAL = 10

# ----------------------------
# Env
# ----------------------------

load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "").strip()
FIREWORKS_ACCOUNT_ID = os.getenv("FIREWORKS_ACCOUNT_ID", "").strip()

if not FIREWORKS_API_KEY:
    print("Error: FIREWORKS_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)
if not FIREWORKS_ACCOUNT_ID:
    print("Error: FIREWORKS_ACCOUNT_ID environment variable is not set.", file=sys.stderr)
    sys.exit(1)

AUTH_HEADERS = {
    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    "Content-Type": "application/json",
}



# ----------------------------
# Helpers
# ----------------------------

def extract_output_text(resp: dict) -> Optional[str]:
    """Extract assistant text from common Fireworks/OpenAI-like response shapes."""
    if isinstance(resp, dict) and resp.get("choices"):
        return resp["choices"][0].get("message", {}).get("content")

    # Some batch outputs store structured output under `output`.
    if isinstance(resp, dict) and "output" in resp:
        for item in resp.get("output", []) or []:
            if item.get("type") == "message" and item.get("role") == "assistant":
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text":
                        return part.get("text")

    return None


def merge_fw_jsonl_into_input_csv(
    input_csv: Path,
    bij_jsonl: Path,
    error_jsonl: Optional[Path],
    output_csv: Path,
) -> None:
    """Merge Fireworks BatchInference output JSONL into the original input CSV.

    Assumptions:
    - Each output line contains `custom_id` like `req-0`, `req-1`, ...
    - The numeric suffix corresponds to the row index (after reset_index).
    - The assistant output text is JSON that contains keys:
        - overall_sentiment
        - overall_emotion
        - triplets (list)

    The function is defensive: it won't crash if parsing fails.
    """

    if not input_csv.exists():
        raise FileNotFoundError(f"INPUT_CSV not found: {input_csv}")
    if not bij_jsonl.exists():
        raise FileNotFoundError(f"BIJ output JSONL not found: {bij_jsonl}")

    df = pd.read_csv(input_csv).reset_index(drop=True)

    # Ensure columns
    for col in [
        "fw_status_code",
        "fw_error",
        "fw_raw_text",
        "fw_overall_sentiment",
        "fw_overall_emotion",
        "fw_triplets_json",
        "fw_raw_json",
    ]:
        if col not in df.columns:
            df[col] = None

    # Parse BIJOutputSet (successes)
    with bij_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            cid = str(rec.get("custom_id", "")).strip()
            if not cid.startswith("req-"):
                continue

            try:
                idx = int(cid.split("-", 1)[1])
            except Exception:
                continue

            if idx < 0 or idx >= len(df):
                continue

            resp = rec.get("response") or {}
            text = extract_output_text(resp)

            df.at[idx, "fw_status_code"] = 200
            df.at[idx, "fw_raw_text"] = text

            if text:
                try:
                    parsed = json.loads(text)
                    df.at[idx, "fw_overall_sentiment"] = parsed.get("overall_sentiment")
                    df.at[idx, "fw_overall_emotion"] = parsed.get("overall_emotion")
                    df.at[idx, "fw_triplets_json"] = json.dumps(
                        parsed.get("triplets", []), ensure_ascii=False
                    )
                    df.at[idx, "fw_raw_json"] = json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    # If assistant output isn't valid JSON, keep raw text only.
                    pass

    # Parse errors if present
    if error_jsonl and error_jsonl.exists():
        with error_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                cid = str(rec.get("custom_id", "")).strip()
                if not cid.startswith("req-"):
                    continue

                try:
                    idx = int(cid.split("-", 1)[1])
                except Exception:
                    continue

                if idx < 0 or idx >= len(df):
                    continue

                err = rec.get("error") or {}
                df.at[idx, "fw_status_code"] = err.get("code")
                df.at[idx, "fw_error"] = json.dumps(err, ensure_ascii=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved merged CSV: {output_csv} | rows={len(df)}")

def create_dataset(dataset_id: str) -> None:
    url = f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/datasets"
    payload = {"datasetId": dataset_id, "dataset": {"userUploaded": {}}}
    resp = requests.post(url, headers=AUTH_HEADERS, data=json.dumps(payload))
    if not resp.ok:
        raise RuntimeError(f"Failed to create dataset {dataset_id}: {resp.status_code} {resp.text}")
    print(f"Created dataset '{dataset_id}' (or already exists)")


def upload_dataset_file(dataset_id: str, jsonl_path: Path) -> None:
    """Upload a dataset file via the supported :upload endpoint."""
    url = f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/datasets/{dataset_id}:upload"
    with jsonl_path.open("rb") as f:
        files = {"file": (jsonl_path.name, f)}
        headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}"}
        resp = requests.post(url, headers=headers, files=files)
    if not resp.ok:
        raise RuntimeError(
            f"Failed to upload dataset file {jsonl_path} to {dataset_id}: {resp.status_code} {resp.text}"
        )
    meta = resp.json() if resp.content else {}
    print(f"Uploaded {jsonl_path} to dataset '{dataset_id}' ({meta.get('bytes')} bytes)")


def resolve_model_id_for_batch(model_id: str) -> str:
    """BatchInferenceJobs expects accounts/<account-id>/models/<model-id>.

    If you paste a deployed model resource (accounts/.../deployedModels/...),
    resolve it via GET deployedModels and return the `model` field.
    """
    if not isinstance(model_id, str):
        return model_id

    s = model_id.strip()
    if "/deployedModels/" not in s:
        return s

    deployed_id = s.split("/deployedModels/", 1)[1]
    url = f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/deployedModels/{deployed_id}"
    resp = requests.get(url, headers=AUTH_HEADERS)
    if not resp.ok:
        raise RuntimeError(f"Failed to resolve deployed model '{s}': {resp.status_code} {resp.text}")

    data = resp.json() if resp.content else {}
    base_model = data.get("model")
    if not base_model and isinstance(data.get("deployedModel"), dict):
        base_model = data["deployedModel"].get("model")

    if not base_model:
        raise RuntimeError(
            "Could not find 'model' in deployedModels response. "
            "Copy the fine-tune Output Model (accounts/.../models/...) and set MODEL_ID to that."
        )

    return str(base_model).strip()


def create_batch_job(
    job_id: str,
    model_id: str,
    input_dataset_id: str,
    output_dataset_id: str,
    inference_params: Dict[str, Any],
) -> Dict[str, Any]:
    url = (
        f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/batchInferenceJobs"
        f"?batchInferenceJobId={job_id}"
    )

    payload = {
        "model": model_id,
        "inputDatasetId": f"accounts/{FIREWORKS_ACCOUNT_ID}/datasets/{input_dataset_id}",
        "outputDatasetId": f"accounts/{FIREWORKS_ACCOUNT_ID}/datasets/{output_dataset_id}",
        "inferenceParameters": inference_params,
    }

    resp = requests.post(url, headers=AUTH_HEADERS, data=json.dumps(payload))
    if not resp.ok:
        raise RuntimeError(f"Failed to create batch job {job_id}: {resp.status_code} {resp.text}")

    return resp.json() if resp.content else {}


def get_batch_job(job_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/batchInferenceJobs/{job_id}"
    resp = requests.get(url, headers=AUTH_HEADERS)
    if not resp.ok:
        raise RuntimeError(f"Failed to get batch job {job_id}: {resp.status_code} {resp.text}")
    return resp.json()


def get_batch_job_state(job_id: str) -> str:
    data = get_batch_job(job_id)
    return data.get("state") or data.get("status") or "UNKNOWN"


def download_output_dataset(dataset_id: str, output_dir: Path) -> Dict[str, Path]:
    url = f"{BASE_URL}/accounts/{FIREWORKS_ACCOUNT_ID}/datasets/{dataset_id}:getDownloadEndpoint"
    resp = requests.get(url, headers=AUTH_HEADERS)
    if not resp.ok:
        raise RuntimeError(
            f"Failed to get download endpoints for dataset {dataset_id}: {resp.status_code} {resp.text}"
        )

    data = resp.json() if resp.content else {}
    signed_urls: Dict[str, str] = data.get("filenameToSignedUrls", {})
    if not signed_urls:
        raise RuntimeError(f"No signed URLs returned for dataset {dataset_id}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    local_paths: Dict[str, Path] = {}

    for object_path, signed_url in signed_urls.items():
        filename = os.path.basename(object_path)
        local_path = output_dir / filename
        print(f"Downloading {filename}...")
        with requests.get(signed_url, stream=True) as r:
            r.raise_for_status()
            with local_path.open("wb") as out:
                for chunk in r.iter_content(chunk_size=8192):
                    out.write(chunk)
        local_paths[filename] = local_path

    return local_paths


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    jsonl_path = Path(INPUT_JSONL)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"INPUT_JSONL not found: {INPUT_JSONL}")

    # 1) Create + upload input dataset
    create_dataset(INPUT_DATASET_ID)
    upload_dataset_file(INPUT_DATASET_ID, jsonl_path)

    # 2) Resolve model id for batch
    resolved_model_id = resolve_model_id_for_batch(MODEL_ID)
    print(f"Using model for batch: {resolved_model_id}")

    # 3) Create output dataset id is referenced in the job payload (must not exist yet)
    job = create_batch_job(
        BATCH_JOB_ID,
        resolved_model_id,
        INPUT_DATASET_ID,
        OUTPUT_DATASET_ID,
        INFERENCE_PARAMS,
    )
    print("Created batch job:")
    print(json.dumps(job, ensure_ascii=False, indent=2))

    # 4) Poll
    while True:
        state = get_batch_job_state(BATCH_JOB_ID)
        print(f"Job state: {state}")
        if state in ("COMPLETED", "JOB_STATE_COMPLETED", "FAILED", "JOB_STATE_FAILED", "EXPIRED", "JOB_STATE_EXPIRED", "CANCELLED", "JOB_STATE_CANCELLED"):
            break
        time.sleep(POLL_INTERVAL)

    if state not in ("COMPLETED", "JOB_STATE_COMPLETED"):
        details = get_batch_job(BATCH_JOB_ID)
        print("\nJob did not complete successfully:")
        print(json.dumps(details, ensure_ascii=False, indent=2))
        raise RuntimeError(f"Batch job ended with state: {state}")

    # 5) Download output dataset files
    out_dir = Path("fireworks_ft_batch_outputs") / RUN_TAG
    files = download_output_dataset(OUTPUT_DATASET_ID, out_dir)
    print("Downloaded:")
    for k, v in files.items():
        print(f"- {k}: {v}")

    # 6) Optionally merge outputs back into an input CSV
    input_csv_path = Path(INPUT_CSV)
    if input_csv_path.exists():
        bij_path = files.get(BIJ_OUTPUT_FILENAME)
        if not bij_path:
            # Fallback: pick the first .jsonl file if the expected name isn't found
            jsonl_files = [p for name, p in files.items() if str(name).lower().endswith(".jsonl")]
            bij_path = jsonl_files[0] if jsonl_files else None

        if not bij_path:
            print(
                "Warning: Could not find BIJ output JSONL in downloaded files; skipping CSV merge. "
                f"Expected '{BIJ_OUTPUT_FILENAME}'."
            )
            return

        err_path = files.get(ERROR_OUTPUT_FILENAME) or (out_dir / ERROR_OUTPUT_FILENAME)
        if err_path and not err_path.exists():
            err_path = None

        merged_csv_path = out_dir / OUTPUT_CSV_NAME
        merge_fw_jsonl_into_input_csv(
            input_csv=input_csv_path,
            bij_jsonl=bij_path,
            error_jsonl=err_path,
            output_csv=merged_csv_path,
        )
    else:
        print(
            f"Note: INPUT_CSV not found at '{input_csv_path}'. "
            "Skipping CSV merge (JSONL outputs are still downloaded)."
        )


if __name__ == "__main__":
    main()