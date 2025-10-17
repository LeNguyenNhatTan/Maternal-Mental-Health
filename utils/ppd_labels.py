# utils/ppd_labels.py
import re
from typing import Optional

VALID_RISK = [
    "Probable depression, urgent referral",
    "High possibility of depression",
    "Depression possible",
    "Depression not likely",
]
VALID_RISK_SET = set(VALID_RISK)
# Map lower-case -> canonical string (để so khớp case-insensitive, trả về đúng bản chuẩn)
_CANON_BY_LOWER = {v.lower(): v for v in VALID_RISK}

PROFILE_TO_RISK = {
    "ppd_low": "Depression not likely",
    "ppd_possible": "Depression possible",
    "ppd_high": "High possibility of depression",
    "ppd_probable": "Probable depression, urgent referral",
    "ppd_proable": "Probable depression, urgent referral",
    "ppd_urgent": "Probable depression, urgent referral",
}

# Bắt đúng phần sau "Risk Level:" đến hết dòng (non-greedy)
RISK_LINE_RE = re.compile(r"risk\s*level\s*:\s*(.+?)(?:\r?\n|$)", re.IGNORECASE)

def risk_from_profile_name(profile: str) -> Optional[str]:
    if not profile:
        return None
    return PROFILE_TO_RISK.get(profile.strip().lower())

def extract_risk_after_risklevel(diagnosis_field: str) -> Optional[str]:
    if not diagnosis_field:
        return None
    m = RISK_LINE_RE.search(diagnosis_field)
    if not m:
        return None

    raw = m.group(1).strip()
    # chuẩn hoá khoảng trắng
    risk_text = " ".join(raw.split())

    # cắt phần chú thích/suffix sau nhãn (ngoặc, dấu gạch, pipe, thẻ markup)
    # ví dụ: "Depression not likely (concerns ...)" -> "Depression not likely"
    for sep in [" (", " —", " –", " - ", " |", "<", "["]:
        idx = risk_text.find(sep)
        if idx != -1:
            risk_text = risk_text[:idx].rstrip()

    # bỏ dấu câu cuối nếu còn
    while risk_text and risk_text[-1] in ".;:,":
        risk_text = risk_text[:-1].rstrip()

    # so khớp case-insensitive và trả canonical
    return _CANON_BY_LOWER.get(risk_text.lower())
