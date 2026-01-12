# merge_fw_jsonl_into_input_csv.py
import json
from pathlib import Path
import pandas as pd

INPUT_CSV = "../../gold/dataset_1k_test_200.csv"   # change to your real path
BIJ_JSONL  = "BIJOutputSet.jsonl"
ERROR_JSONL = "error-data"                      # optional if exists
OUTPUT_CSV = "dataset_ft-llama-v3p1-8b.csv"

def extract_output_text(resp: dict):
    if isinstance(resp, dict) and resp.get("choices"):
        return resp["choices"][0].get("message", {}).get("content")
    if isinstance(resp, dict) and "output" in resp:
        for item in resp.get("output", []) or []:
            if item.get("type") == "message" and item.get("role") == "assistant":
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text":
                        return part.get("text")
    return None

# Load input
df = pd.read_csv(INPUT_CSV).reset_index(drop=True)

# Ensure columns
for col in ["fw_status_code","fw_error","fw_raw_text","fw_overall_sentiment","fw_overall_emotion","fw_triplets_json","fw_raw_json"]:
    if col not in df.columns:
        df[col] = None

# Parse BIJOutputSet (successes)
with Path(BIJ_JSONL).open("r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cid = str(rec.get("custom_id","")).strip()
        if not cid.startswith("req-"):
            continue
        idx = int(cid.split("-",1)[1])

        resp = rec.get("response") or {}
        text = extract_output_text(resp)

        df.at[idx, "fw_status_code"] = 200
        df.at[idx, "fw_raw_text"] = text

        if text:
            try:
                parsed = json.loads(text)
                df.at[idx, "fw_overall_sentiment"] = parsed.get("overall_sentiment")
                df.at[idx, "fw_overall_emotion"] = parsed.get("overall_emotion")
                df.at[idx, "fw_triplets_json"] = json.dumps(parsed.get("triplets", []), ensure_ascii=False)
                df.at[idx, "fw_raw_json"] = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass

# Parse errors if present
err_path = Path(ERROR_JSONL)
if err_path.exists():
    with err_path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec.get("custom_id","")).strip()
            if not cid.startswith("req-"):
                continue
            idx = int(cid.split("-",1)[1])
            err = rec.get("error") or {}
            df.at[idx, "fw_status_code"] = err.get("code")
            df.at[idx, "fw_error"] = json.dumps(err, ensure_ascii=False)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV} | rows={len(df)}")