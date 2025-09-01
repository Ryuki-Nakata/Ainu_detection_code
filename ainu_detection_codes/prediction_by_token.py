# classify_lines_by_tokens.py
import os
import argparse
import json
from typing import List, Dict, Tuple
import pandas as pd
from joblib import load

def load_models(model_dir: str):
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    clf_path = os.path.join(model_dir, "random_forest_classifier.joblib")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vectorizer not found: {vec_path}")
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Classifier not found: {clf_path}")
    vectorizer = load(vec_path)
    clf = load(clf_path)
    return vectorizer, clf

def tokenize_by_space(line: str, lowercase: bool = False) -> List[str]:
    """スペースで分割のみ。句読点等の処理はしない。"""
    if lowercase:
        line = line.lower()
    return [t for t in line.strip().split() if t]

def predict_tokens(tokens: List[str], vectorizer, clf, cache: Dict[str, str]) -> List[str]:
    """同じトークンはキャッシュして高速化。"""
    preds = []
    to_infer = []
    idx_map = []
    for i, tok in enumerate(tokens):
        if tok in cache:
            preds.append(cache[tok])
        else:#キャッシュとしてtokenが存在していない→分類機通す必要あり
            to_infer.append(tok)
            idx_map.append(i)
            preds.append(None)  # 後で埋める

    if to_infer:
        X = vectorizer.transform(to_infer)
        batch_preds = clf.predict(X)
        for i, label in zip(idx_map, batch_preds):
            cache[tokens[i]] = label
            preds[i] = label

    return preds

def classify_file(
    input_path: str,
    model_dir: str = ".",
    target_label: str = "ainu",
    lowercase_tokens: bool = False,
    output_prefix: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    行ごとにスペース分割→各トークンを言語判定→全トークンが target_label なら行は Ainu 文。
    出力:
      - {prefix}_classified.csv     : 行単位サマリ
      - {prefix}_classified.jsonl   : 行＋トークン詳細
      - {prefix}_ainu_only.txt      : Ainuと判定された行のみ
      - {prefix}_not_ainu.txt       : Ainu以外トークンを含む行のみ
    """
    vectorizer, clf = load_models(model_dir)
    cache: Dict[str, str] = {}

    if output_prefix is None:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_prefix = f"{stem}"

    rows = []
    jsonl_path = f"{output_prefix}_classified.jsonl"
    ainu_only_path = f"{output_prefix}_ainu_only.txt"
    not_ainu_path = f"{output_prefix}_not_ainu.txt"

    total_lines = 0
    ainu_only_count = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(jsonl_path, "w", encoding="utf-8") as f_jsonl, \
         open(ainu_only_path, "w", encoding="utf-8") as f_ainu, \
         open(not_ainu_path, "w", encoding="utf-8") as f_non:

        for line_no, line in enumerate(f_in, start=1):
            text = line.rstrip("\n")
            if not text.strip():
                # 空行はスキップ（必要なら保持してもOK）
                continue

            total_lines += 1
            tokens = tokenize_by_space(text, lowercase=lowercase_tokens)
            if len(tokens) == 0:
                continue

            token_labels = predict_tokens(tokens, vectorizer, clf, cache)
            is_all_target = all(lbl == target_label for lbl in token_labels)

            if is_all_target:
                ainu_only_count += 1
                f_ainu.write(text + "\n")
            else:
                f_non.write(text + "\n")

            # 詳細は JSONL に1行ずつ保存（後で柔軟に分析しやすい）
            detail = {
                "line_no": line_no,
                "text": text,
                "tokens": tokens,
                "pred_labels": token_labels,
                "is_ainu_line": bool(is_all_target)
            }
            f_jsonl.write(json.dumps(detail, ensure_ascii=False) + "\n")

            # CSV 用のサマリ行
            non_target_tokens = [t for t, lab in zip(tokens, token_labels) if lab != target_label]
            non_target_labels = [lab for lab in token_labels if lab != target_label]
            rows.append({
                "line_no": line_no,
                "is_ainu_line": bool(is_all_target),
                "num_tokens": len(tokens),
                "num_pred_target": sum(1 for lab in token_labels if lab == target_label),
                "num_pred_non_target": len(non_target_tokens),
                "non_target_labels_set": "|".join(sorted(set(non_target_labels))) if non_target_labels else "",
                # 後で見返すために最初の数個だけサンプル保存（長過ぎる展開を避ける）
                "non_target_token_samples": " || ".join(non_target_tokens[:5]),
                "text": text
            })

    df = pd.DataFrame(rows)
    csv_path = f"{output_prefix}_classified.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    summary = {
        "input_path": input_path,
        "model_dir": model_dir,
        "target_label": target_label,
        "total_lines": total_lines,
        "ainu_only_lines": ainu_only_count,
        "not_ainu_lines": total_lines - ainu_only_count,
        "csv_path": csv_path,
        "jsonl_path": jsonl_path,
        "ainu_only_txt": ainu_only_path,
        "not_ainu_txt": not_ainu_path
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Line-level Ainu detection via token-wise RF predictions."
    )
    parser.add_argument(
        "--input", required=True,
        help="入力テキストファイル（1行=1文）"
    )
    parser.add_argument(
        "--model-dir", default=".",
        help="joblib（tfidf_vectorizer / random_forest_classifier）のあるディレクトリ"
    )
    parser.add_argument(
        "--target", default="ainu",
        help="全トークンがこのラベルならAinu文とみなす（デフォルト: ainu）"
    )
    parser.add_argument(
        "--lowercase", action="store_true",
        help="トークンを小文字化して判定する"
    )
    parser.add_argument(
        "--output-prefix", default=None,
        help="出力ファイル接頭辞（未指定なら入力ファイル名のstem）"
    )
    args = parser.parse_args()

    df, summary = classify_file(
        input_path=args.input,
        model_dir=args.model_dir,
        target_label=args.target,
        lowercase_tokens=args.lowercase,
        output_prefix=args.output_prefix
    )
    print("=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
