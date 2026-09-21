import re
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# ---- CHANGE THESE TO MATCH YOUR CSV ----
CSV_PATH = "data/spam.csv"
TEXT_COL = "text"
LABEL_COL = "label"
# ----------------------------------------

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))
STEM = PorterStemmer()

def clean(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " emailaddr ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(STEM.stem(w) for w in text.split() if w not in STOP)

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    df = df[[TEXT_COL, LABEL_COL]].dropna()

    # Convert labels to 0/1 (handles "spam"/"ham" or 1/0)
    if df[LABEL_COL].dtype == object:
        df[LABEL_COL] = df[LABEL_COL].str.lower().map({"spam": 1, "ham": 0})
    df = df.dropna()

    print("Class counts:\n", df[LABEL_COL].value_counts())

    df["clean"] = df[TEXT_COL].apply(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df[LABEL_COL], test_size=0.2,
        random_state=42, stratify=df[LABEL_COL]
    )

    candidates = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC(),
    }

    best_name, best_acc, best_pipe = None, 0, None
    for name, clf in candidates.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"\n=== {name} | accuracy: {acc:.4f} ===")
        print(classification_report(y_test, pred, target_names=["ham", "spam"]))
        if acc > best_acc:
            best_name, best_acc, best_pipe = name, acc, pipe

    print(f"\nBest model: {best_name} ({best_acc:.4f})")
    joblib.dump(best_pipe, "models/spam_model.pkl")
    print("Saved to models/spam_model.pkl")