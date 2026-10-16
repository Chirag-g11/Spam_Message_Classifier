
"""
Spam Message Classifier
Training + Evaluation + Interactive Prediction
"""

from pathlib import Path
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sms_spam.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# 1. Text preprocessing
def clean_text(text):
    """Convert text to lowercase and clean URLs and punctuation."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. Load and validate dataset
def load_data():
    """Load the labelled spam-message dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Create data/sms_spam.csv with label and message columns."
        )

    df = pd.read_csv(DATA_PATH)

    required = {"label", "message"}

    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain 'label' and 'message' columns."
        )

    df = df[["label", "message"]].dropna()

    df["label"] = (
        df["label"].astype(str).str.lower().str.strip()
    )

    df["label"] = df["label"].replace({
        "0": "ham",
        "1": "spam"
    })

    df = df[df["label"].isin(["ham", "spam"])]

    df["message"] = df["message"].map(clean_text)

    df = df[df["message"].str.len() > 0]
    df = df.drop_duplicates()

    if df["label"].nunique() < 2:
        raise ValueError(
            "Dataset must contain both ham and spam messages."
        )

    return df


# 3. Train and evaluate models
def train_model():

    print("\nLoading dataset...")

    df = load_data()

    X = df["message"]
    y = df["label"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Define machine-learning pipelines
    candidates = {
        "Multinomial Naive Bayes": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2)
                )
            ),
            ("model", MultinomialNB())
        ]),

        "Logistic Regression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2)
                )
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ])
    }

    results = []
    trained_models = {}

    # Train and compare both models
    for name, pipeline in candidates.items():

        print(f"\nTraining {name}...")

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        results.append({
            "model": name,
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "precision_spam": precision_score(
                y_test,
                predictions,
                pos_label="spam",
                zero_division=0
            ),
            "recall_spam": recall_score(
                y_test,
                predictions,
                pos_label="spam",
                zero_division=0
            ),
            "f1_spam": f1_score(
                y_test,
                predictions,
                pos_label="spam",
                zero_division=0
            )
        })

        trained_models[name] = pipeline

    # Rank models by spam precision, then F1-score
    scores = pd.DataFrame(results).sort_values(
        ["precision_spam", "f1_spam"],
        ascending=False
    )

    print("\n===== MODEL COMPARISON =====")
    print(
        scores.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}"
        )
    )

    scores.to_csv(
        REPORT_DIR / "metrics.csv",
        index=False
    )

    # Select and save the best model
    best_name = scores.iloc[0]["model"]
    best_model = trained_models[best_name]

    joblib.dump(best_model, MODEL_PATH)

    with open(
        MODEL_DIR / "model_name.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(best_name)

    # Evaluate the selected model
    final_predictions = best_model.predict(X_test)

    print(f"\nSelected model: {best_name}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            final_predictions,
            zero_division=0
        )
    )

    # Generate confusion matrix
    matrix = confusion_matrix(
        y_test,
        final_predictions,
        labels=["ham", "spam"]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["Predicted ham", "Predicted spam"],
        yticklabels=["Actual ham", "Actual spam"]
    )

    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=160
    )

    plt.close()

    print("\nModel and reports saved successfully.")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {REPORT_DIR / 'metrics.csv'}")
    print(
        f"Confusion matrix: "
        f"{REPORT_DIR / 'confusion_matrix.png'}"
    )

    return best_model


# 4. Interactive prediction
def predict_messages(model):

    print("\n===== SPAM MESSAGE CLASSIFIER =====")
    print("Enter a message to classify.")
    print("Type 'exit' to stop.\n")

    while True:

        message = input("Enter message: ").strip()

        if message.lower() == "exit":
            print("Program closed.")
            break

        if not message:
            print("Please enter a message.\n")
            continue

        cleaned_message = clean_text(message)

        prediction = model.predict([cleaned_message])[0]

        if prediction == "spam":
            print(
                "Prediction: SPAM "
                "- Treat this message cautiously.\n"
            )
        else:
            print(
                "Prediction: HAM "
                "- Likely a legitimate message.\n"
            )


# 5. Run the complete project
def main():

    model = train_model()

    predict_messages(model)


if __name__ == "__main__":
    main()
