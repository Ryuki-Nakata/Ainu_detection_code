import os
import argparse
import json
from typing import List, Dict, Tuple
import pandas as pd
from joblib import load

# ---------------- モデル読み込み ----------------
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

# ---------------- 前処理 ----------------
def tokenize_by_space(line: str, lowercase: bool = False) -> List[str]:
    """スペースで分割のみ。句読点等の処理はしない。"""
    if lowercase:
        line = line.lower()
    return [t for t in line.strip().split() if t]

def build_token_ngrams(tokens: List[str], n: int, join_with: str = " ") -> List[str]:
    """
    連続する n トークンで n-gram を作る。
    tokens が n 未満の場合は、その行全体（トークン全部）を 1 つの“短い n-gram”として返す。
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return [join_with.join(tokens)] if tokens else []
    return [join_with.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# ---------------- 予測（キャッシュ付き） ----------------
def predict_strings(strings: List[str], vectorizer, clf, cache: Dict[str, str]) -> List[str]:
    """同じ n-gram はキャッシュして高速化。"""
    preds = []
    to_infer = []
    idx_map = []
    for i, s in enumerate(strings):
        if s in cache:
            preds.append(cache[s])
        else:
            to_infer.append(s)
            idx_map.append(i)
            preds.append(None)
    if to_infer:
        X = vectorizer.transform(to_infer)
        batch = clf.predict(X)
        for i, label in zip(idx_map, batch):
            cache[strings[i]] = label
            preds[i] = label
    return preds

# ---------------- メイン処理 ----------------
def classify_file_by_ngrams(
    input_path: str,
    model_dir: str = ".",
    target_label: str = "ainu",
    lowercase_tokens: bool = False,
    output_prefix: str = None,
    ngram_size: int = 2
) -> Tuple[pd.DataFrame, Dict]:
    """
    行をスペース分割 → 連続 n トークンの n-gram を作成 → 各 n-gram をRFで言語判定。
    文中の n-gram のうち 1 つでも target_label 以外が出たら、その文は純Ainuではないと判定。
    出力:
      - {prefix}_classified.csv
      - {prefix}_classified.jsonl
      - {prefix}_ainu_only.txt
      - {prefix}_not_ainu.txt
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
                continue

            total_lines += 1
            tokens = tokenize_by_space(text, lowercase=lowercase_tokens)
            if len(tokens) == 0:
                continue

            ngrams = build_token_ngrams(tokens, n=ngram_size, join_with=" ")
            if not ngrams:
                # トークンが空でなければ最低1つは返す設計だが、保険でスキップ
                continue

            ngram_labels = predict_strings(ngrams, vectorizer, clf, cache)
            # 1つでも target_label 以外があれば、その行は純Ainuではない
            is_all_target = all(lbl == target_label for lbl in ngram_labels)

            if is_all_target:
                ainu_only_count += 1
                f_ainu.write(text + "\n")
            else:
                f_non.write(text + "\n")

            # JSONL 詳細
            detail = {
                "line_no": line_no,
                "text": text,
                "tokens": tokens,
                "ngrams": ngrams,
                "ngram_size": ngram_size,
                "pred_labels": ngram_labels,
                "is_ainu_line": bool(is_all_target)
            }
            f_jsonl.write(json.dumps(detail, ensure_ascii=False) + "\n")

            # CSV 用サマリ
            non_target_ngrams = [g for g, lab in zip(ngrams, ngram_labels) if lab != target_label]
            non_target_labels = [lab for lab in ngram_labels if lab != target_label]
            rows.append({
                "line_no": line_no,
                "is_ainu_line": bool(is_all_target),
                "num_tokens": len(tokens),
                "ngram_size": ngram_size,
                "num_ngrams": len(ngrams),
                "num_pred_target_ngrams": sum(1 for lab in ngram_labels if lab == target_label),
                "num_pred_non_target_ngrams": len(non_target_ngrams),
                "non_target_labels_set": "|".join(sorted(set(non_target_labels))) if non_target_labels else "",
                "non_target_ngram_samples": " || ".join(non_target_ngrams[:5]),
                "text": text
            })

    df = pd.DataFrame(rows)
    csv_path = f"{output_prefix}_classified.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    summary = {
        "input_path": input_path,
        "model_dir": model_dir,
        "target_label": target_label,
        "ngram_size": ngram_size,
        "total_lines": total_lines,
        "ainu_only_lines": ainu_only_count,
        "not_ainu_lines": total_lines - ainu_only_count,
        "csv_path": csv_path,
        "jsonl_path": jsonl_path,
        "ainu_only_txt": ainu_only_path,
        "not_ainu_txt": not_ainu_path
    }
    return df, summary

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Line-level Ainu detection via token n-grams with RF."
    )
    parser.add_argument("--input", required=True,
                        help="入力テキストファイル（1行=1文）")
    parser.add_argument("--model-dir", default=".",
                        help="joblib（tfidf_vectorizer / random_forest_classifier）のディレクトリ")
    parser.add_argument("--target", default="ainu",
                        help="n-gram がこのラベルなら Ainu とみなす（デフォルト: ainu）")
    parser.add_argument("--lowercase", action="store_true",
                        help="トークンを小文字化して判定する")
    parser.add_argument("--output-prefix", default=None,
                        help="出力ファイル接頭辞（未指定なら入力ファイル名のstem）")
    parser.add_argument("--ngram", type=int, default=2,
                        help="トークン n-gram の n（デフォルト: 2）")
    args = parser.parse_args()

    if args.ngram < 1:
        raise ValueError("--ngram は 1 以上にしてください")

    df, summary = classify_file_by_ngrams(
        input_path=args.input,
        model_dir=args.model_dir,
        target_label=args.target,
        lowercase_tokens=args.lowercase,
        output_prefix=args.output_prefix,
        ngram_size=args.ngram
    )

    print("=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
