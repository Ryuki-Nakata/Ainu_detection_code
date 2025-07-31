import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

# --- ユーティリティ関数 -------------------------------------------------------

def load_data(file_path: str, label: str) -> pd.DataFrame:
    """
    テキストファイルを読み込み、
    各行を 'text' カラム、与えられたラベルを 'label' カラムに格納した DataFrame を返す。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    df = pd.DataFrame({'text': lines, 'label': label})
    print(f"Loaded {len(df)} sentences from {file_path} as [{label}]")
    return df

def evaluate_classifier(clf, X_test, y_test):
    """
    混同行列と分類レポートを出力する。
    """
    y_pred = clf.predict(X_test)
    # confusion matrix を取得（labels で順番を固定）
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    # 行単位で割合に変換（＝各言語の実際の件数で割る）
    cm_ratio = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 可視化（割合表示）
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm_ratio,
                annot=True,
                fmt='.4f',
                xticklabels=clf.classes_,
                yticklabels=clf.classes_,
                cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Proportions)')
    plt.tight_layout()
    plt.show()

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))


# --- メイン処理 ---------------------------------------------------------------

def main():
    # 1) データ読み込み
    ja_df   = load_data('ja_story_3k_romaji.txt',   'ja')
    hi_df   = load_data('hi_story_3k_romaji.txt',   'hi')
    ainu_df = load_data('ainu_story_3k.txt', 'ainu')
    haw_df = load_data('haw_story_3k.txt', 'haw')

    # 2) データ統合
    data = pd.concat([ja_df, hi_df, ainu_df, haw_df], ignore_index=True)
    X_text = data['text']
    y       = data['label']

    # 3) 特徴量抽出：文字レベル n-gram TF-IDF
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X = vectorizer.fit_transform(X_text)
    print(f"Vectorized {X.shape[0]} texts → {X.shape[1]} features")

    # 4) 訓練／テスト分割 (層化抽出)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # 5) モデル訓練
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("RandomForest trained.")

    # 6) 評価
    evaluate_classifier(clf, X_test, y_test)

    # 7) モデル & ベクトライザを保存
    dump(vectorizer, 'tfidf_vectorizer.joblib')
    dump(clf,        'random_forest_classifier.joblib')
    print("Saved: tfidf_vectorizer.joblib, random_forest_classifier.joblib")

if __name__ == '__main__':
    main()
