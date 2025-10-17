# scripts/render_ppd_visuals.py
# -*- coding: utf-8 -*-
"""
Đọc chat_logs/_metrics/<model>/metrics_summary.json
→ Vẽ Confusion Matrix (PNG) + Render Classification Report (HTML + CSV)
Lưu ngay cùng thư mục với metrics_summary.json
"""

import os, sys, json
from typing import List, Dict

# --- path patch: cho phép "python scripts/render_ppd_visuals.py" chạy từ repo root ---
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_ORDER = [
    "Depression not likely",
    "Depression possible",
    "High possibility of depression",
    "Probable depression, urgent referral",
]


def load_metrics_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_png: str):
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = plt.gca()

    # Dùng colormap "Blues" để có tông xanh dịu
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    # Ghi số vào từng ô
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, int(cm[i, j]),
                ha="center", va="center", color="black"
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # thêm thanh màu bên phải
    fig.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)



def report_dict_to_dataframe(rep: Dict, label_order: List[str]) -> pd.DataFrame:
    """
    rep là output_dict của sklearn.classification_report (đã được lưu trong metrics_summary.json)
    Trả về DataFrame gọn:
      index = [label_order..., 'macro avg', 'weighted avg', 'accuracy']
    """
    rows = []
    # lấy theo đúng thứ tự label
    for lbl in label_order:
        d = rep.get(lbl, {}) if isinstance(rep, dict) else {}
        rows.append({
            "class": lbl,
            "precision": d.get("precision", 0),
            "recall": d.get("recall", 0),
            "f1-score": d.get("f1-score", 0),
            "support": d.get("support", 0),
        })

    # thêm macro/weighted
    for special in ["macro avg", "weighted avg"]:
        d = rep.get(special, {}) if isinstance(rep, dict) else {}
        rows.append({
            "class": special,
            "precision": d.get("precision", 0),
            "recall": d.get("recall", 0),
            "f1-score": d.get("f1-score", 0),
            "support": d.get("support", 0),
        })

    # accuracy (sklearn đặt ở key 'accuracy' là 1 số; ta đưa vào hàng riêng)
    acc = rep.get("accuracy", None) if isinstance(rep, dict) else None
    rows.append({
        "class": "accuracy",
        "precision": "",
        "recall": "",
        "f1-score": acc if acc is not None else "",
        "support": "",
    })

    df = pd.DataFrame(rows)
    return df


def render_one_model(model_metrics_dir: str):
    """
    Đầu vào: chat_logs/_metrics/<model_name>/
      - metrics_summary.json
    Đầu ra:
      - cm.png               (confusion matrix)
      - report.csv           (classification report dạng bảng)
      - report.html          (bảng HTML dễ nhìn)
      - overview.txt         (tóm tắt nhanh overall accuracy/F1)
    """
    metrics_json = os.path.join(model_metrics_dir, "metrics_summary.json")
    if not os.path.exists(metrics_json):
        print(f"⚠️  metrics_summary.json not found in {model_metrics_dir}, skip.")
        return

    data = load_metrics_json(metrics_json)
    model_name = data.get("model") or os.path.basename(model_metrics_dir)
    cm = data.get("confusion_matrix")
    per_class = data.get("per_class", {})
    overall = data.get("overall", {})
    label_order = data.get("label_order", LABEL_ORDER)

    # 1) Confusion Matrix
    if cm is not None:
        cm_arr = np.array(cm, dtype=int)
        cm_png = os.path.join(model_metrics_dir, "cm.png")
        plot_confusion_matrix(
            cm_arr, label_order,
            title=f"Confusion Matrix — {model_name}",
            out_png=cm_png
        )
        print(f"   • Saved CM: {cm_png}")
    else:
        print(f"   • No confusion_matrix for {model_name}")

    # 2) Classification Report → CSV + HTML
    df = report_dict_to_dataframe(per_class, label_order)
    rep_csv = os.path.join(model_metrics_dir, "report.csv")
    rep_html = os.path.join(model_metrics_dir, "report.html")

    # Làm tròn đẹp số liệu
    def _fmt(x):
        try:
            return f"{float(x):.3f}"
        except Exception:
            return x

    df_fmt = df.copy()
    for col in ["precision", "recall", "f1-score"]:
        df_fmt[col] = df_fmt[col].apply(_fmt)

    df_fmt.to_csv(rep_csv, index=False, encoding="utf-8-sig")
    # HTML có style đơn giản
    html = [
        "<html><head><meta charset='utf-8'><title>Classification Report</title>",
        "<style>",
        "table {border-collapse: collapse; font-family: Arial, sans-serif;}",
        "th, td {border: 1px solid #ddd; padding: 6px 10px; font-size: 13px;}",
        "th {background: #f2f2f2;}",
        "tr:nth-child(even) {background: #fafafa;}",
        "caption {text-align:left; font-weight:600; margin: 8px 0;}",
        "</style></head><body>",
        f"<h2>Classification Report — {model_name}</h2>",
        df_fmt.to_html(index=False, escape=False),
        "<hr/>",
        "<p><em>Note:</em> 'accuracy' nằm ở cột f1-score do format từ sklearn.</p>",
        "</body></html>",
    ]
    with open(rep_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"   • Saved report: {rep_csv}")
    print(f"   • Saved report: {rep_html}")

    # 3) Ghi tóm tắt nhanh
    overview_txt = os.path.join(model_metrics_dir, "overview.txt")
    with open(overview_txt, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluated pairs: {data.get('evaluated_pairs', 0)} / {data.get('total_rows_in_batch', 0)}\n")
        f.write(f"Accuracy: {overall.get('accuracy', 0):.3f}\n")
        f.write(f"F1_macro: {overall.get('f1_macro', 0):.3f}\n")
        f.write(f"F1_micro: {overall.get('f1_micro', 0):.3f}\n")
    print(f"   • Saved overview: {overview_txt}")


def main():
    # Root chat_logs nằm ở gốc project (đã thống nhất trước đó với bạn)
    chat_logs_root = os.path.join(_REPO_ROOT, "chat_logs")
    metrics_root = os.path.join(chat_logs_root, "_metrics")

    if len(sys.argv) > 1:
        # render cho 1 model cụ thể
        model_name = sys.argv[1]
        model_dir = os.path.join(metrics_root, model_name)
        if not os.path.isdir(model_dir):
            print(f"❌ Not found: {model_dir}")
            sys.exit(1)
        print(f"\n=== Render visuals for: {model_name} ===")
        render_one_model(model_dir)
    else:
        # render cho tất cả model trong _metrics/
        if not os.path.isdir(metrics_root):
            print(f"❌ Not found metrics root: {metrics_root}")
            sys.exit(1)
        subdirs = [
            d for d in os.listdir(metrics_root)
            if os.path.isdir(os.path.join(metrics_root, d)) and not d.startswith(".")
        ]
        if not subdirs:
            print(f"⚠️ No model subfolders under {metrics_root}")
            sys.exit(0)
        for name in subdirs:
            print(f"\n=== Render visuals for: {name} ===")
            render_one_model(os.path.join(metrics_root, name))


if __name__ == "__main__":
    main()
