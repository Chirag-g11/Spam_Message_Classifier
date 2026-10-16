# Spam Message Classifier

A beginner-friendly machine-learning project that classifies SMS text as `spam` or `ham`.

## Important dataset note
The included `data/sms_spam.csv` is a **small demonstration dataset** so you can run the project immediately. It is not large enough to make reliable real-world claims. For your final submission, use the full SMS Spam Collection dataset and evaluate it again.

## 1. Install Python
Install Python 3.10 or newer from https://www.python.org/downloads/ and tick **Add Python to PATH** during installation.

## 2. Open the project folder in Windows
Open Command Prompt or the VS Code terminal, then use `cd` to move into this folder.

## 3. Create a virtual environment (recommended)
```bat
python -m venv .venv
.venv\Scripts\activate
```

## 4. Install dependencies
```bat
python -m pip install -r requirements.txt
```

## 5. Train, evaluate & Test your own messages
```bat
python spam_classifier.py
```
This splits the dataset into 80% training and 20% testing, compares Multinomial Naive Bayes with Logistic Regression, prints metrics, and saves the best model, a CSV metrics report, and a confusion-matrix image.

Type a message and press Enter. Type `exit` to stop.

## Use a full dataset for final submission
Search for the **UCI SMS Spam Collection** dataset and download its `SMSSpamCollection` file. Convert it to `data/sms_spam.csv` with headers `label,message`, where each row contains a label (`ham` or `spam`) and its message text. Keep the dataset's license/usage terms and cite the dataset source in your report. Do not use the test split to tune the model.

## Project structure
```text
spam_message_classifier/
├── data/
│   └── sms_spam.csv
├── models/                 # created after training
├── reports/                # created after training
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## What the metrics mean
- **Precision (spam):** Of messages predicted as spam, how many really were spam. Higher precision reduces false alarms.
- **Recall (spam):** Of all real spam messages, how many the model found.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Shows correct and incorrect ham/spam predictions.
- **Accuracy:** Fraction of all test messages classified correctly.

A high score on the small demo dataset does not prove real-world performance. Report scores from the full held-out dataset in your submission.
