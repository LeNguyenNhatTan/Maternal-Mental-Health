# scripts/eval_ppd_metrics.py

import os, sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)   
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import os, csv, json, sys
from typing import List
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from utils.log_parsers import find_log_by_conv_id, pair_truth_pred_from_log

LABEL_ORDER = [
    "Depression not likely",
    "Depression possible",
    "High possibility of depression",
    "Probable depression, urgent referral",
]
IDX = {lbl:i for i,lbl in enumerate(LABEL_ORDER)}

def evaluate_model(model_dir: str, out_root: str):
    print(f"\n=== Evaluating model: {os.path.basename(model_dir)} ===")
    batch_csv = os.path.join(model_dir, "batch_summary.csv")
    logs_root = model_dir
    model_name = os.path.basename(model_dir)

    if not os.path.exists(batch_csv):
        print(f"⚠️  batch_summary.csv not found in {model_dir}, skipping.")
        return None

    out_dir = os.path.join(out_root, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load conversation IDs
    conv_ids: List[str] = []
    with open(batch_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            conv_id = (r.get("Conversation ID") or r.get("conversation_id") or "").strip()
            if conv_id:
                conv_ids.append(conv_id)

    # 2) Parse logs
    y_true, y_pred = [], []
    per_item = []

    for conv_id in conv_ids:
        fp = find_log_by_conv_id(logs_root, conv_id)
        if not fp:
            per_item.append({
                "conversation_id": conv_id, "status": "missing_log",
                "y_true": None, "y_pred": None, "log_path": None
            })
            continue

        pair = pair_truth_pred_from_log(fp)
        if pair is None:
            per_item.append({
                "conversation_id": conv_id, "status": "parse_failed",
                "y_true": None, "y_pred": None, "log_path": fp
            })
            continue

        _, tru, pred = pair
        y_true.append(tru)
        y_pred.append(pred)
        per_item.append({
            "conversation_id": conv_id, "status": "ok",
            "y_true": tru, "y_pred": pred, "log_path": fp
        })

    # 3) Compute metrics
    summary = {
        "model": model_name,
        "total_rows_in_batch": len(conv_ids),
        "evaluated_pairs": len(y_true),
        "skipped": len(conv_ids) - len(y_true),
        "label_order": LABEL_ORDER,
        "overall": {},
        "per_class": {},
        "confusion_matrix": None,
        "support_per_class": None,
    }

    if y_true:
        y_true_idx = [IDX[s] for s in y_true]
        y_pred_idx = [IDX[s] for s in y_pred]

        labels_idx = list(range(len(LABEL_ORDER)))

        acc = accuracy_score(y_true_idx, y_pred_idx)
        f1_macro = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
        f1_micro = f1_score(y_true_idx, y_pred_idx, average="micro", zero_division=0)

        clf_rep = classification_report(
            y_true_idx,
            y_pred_idx,
            labels=labels_idx,                  
            target_names=LABEL_ORDER,          
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_true_idx,
            y_pred_idx,
            labels=labels_idx                   
        )

        # support mỗi lớp (đếm theo y_true)
        from collections import Counter
        cnt = Counter(y_true_idx)
        support = [cnt.get(i, 0) for i in labels_idx]

        summary["overall"] = {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}
        summary["per_class"] = clf_rep
        summary["confusion_matrix"] = cm.tolist()
        summary["support_per_class"] = dict(zip(LABEL_ORDER, support))

    # 4) Save results
    per_item_csv = os.path.join(out_dir, "pairwise_predictions.csv")
    with open(per_item_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["conversation_id","status","y_true","y_pred","log_path"])
        writer.writeheader()
        for r in per_item:
            writer.writerow(r)

    summary_json = os.path.join(out_dir, "metrics_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[✓] {model_name}: {summary['evaluated_pairs']} / {summary['total_rows_in_batch']} evaluated")
    print(f"    Accuracy={summary['overall'].get('accuracy',0):.3f} | F1_macro={summary['overall'].get('f1_macro',0):.3f}")
    print(f"    Saved to {out_dir}")
    return summary


def main():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(scripts_dir)
    chat_logs_root = os.path.join(repo_root, "chat_logs")
    out_root = os.path.join(chat_logs_root, "_metrics")
    os.makedirs(out_root, exist_ok=True)

    # Nếu có argument model_name thì chỉ đánh giá model đó
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        model_dir = os.path.join(chat_logs_root, model_name)
        if not os.path.exists(model_dir):
            print(f"Model folder '{model_name}' not found in {chat_logs_root}")
            sys.exit(1)
        evaluate_model(model_dir, out_root)
    else:
        # chạy tất cả model con
        subfolders = [
            os.path.join(chat_logs_root, d)
            for d in os.listdir(chat_logs_root)
            if os.path.isdir(os.path.join(chat_logs_root, d)) and not d.startswith("_")
        ]
        for subdir in subfolders:
            evaluate_model(subdir, out_root)

if __name__ == "__main__":
    main()
