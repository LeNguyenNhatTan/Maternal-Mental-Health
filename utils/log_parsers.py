# utils/log_parsers.py
import json, os, glob
from typing import Optional, Tuple
from utils.ppd_labels import risk_from_profile_name, extract_risk_after_risklevel

def load_chatlog(filepath: str) -> Optional[dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def pair_truth_pred_from_log(filepath: str) -> Optional[Tuple[str, str, str]]:
    """
    return (conversation_id, y_true_risk, y_pred_risk) với y* là 1 trong 4 chuỗi VALID_RISK.
    - y_true lấy từ metadata.patient_profile -> map qua PROFILE_TO_RISK
    - y_pred lấy CHỈ từ phần sau 'Risk Level:' trong diagnosis
    """
    data = load_chatlog(filepath)
    if not data:
        return None

    meta = data.get("metadata", {})
    profile = meta.get("patient_profile")
    y_true = risk_from_profile_name(profile)

    diagnosis = data.get("diagnosis") or ""
    y_pred = extract_risk_after_risklevel(diagnosis)

    conv_id = os.path.splitext(os.path.basename(filepath))[0]
    if y_true is None or y_pred is None:
        return None
    return (conv_id, y_true, y_pred)

def find_log_by_conv_id(logs_root: str, conv_id: str) -> Optional[str]:
    pattern = os.path.join(logs_root, "**", f"{conv_id}.json")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None
