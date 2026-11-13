"""Train the spam classifier and save it to model.joblib."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/sms.csv").dropna()
df["y"] = (df["label"] == "spam").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
)


def make(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("clf", clf),
    ])


nb = make(MultinomialNB(alpha=0.1))
lr = make(LogisticRegression(max_iter=1000))

for name, model in [("Naive Bayes", nb), ("Logistic Regression", lr)]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))

# Save a confusion matrix picture for the report
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ConfusionMatrixDisplay.from_estimator(nb, X_test, y_test, display_labels=["ham", "spam"])
    plt.title("Naive Bayes confusion matrix")
    plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("\nSaved confusion_matrix.png")
except ImportError:
    print("\n(matplotlib not installed, skipped confusion matrix image)")

# The app uses Naive Bayes (classic, fast and easy to explain in a viva)
joblib.dump(nb, "model.joblib")
print("Saved model.joblib")
