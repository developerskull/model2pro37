# 🛡️ SMS Spam Shield AI

An explainable NLP application that classifies SMS messages as **Spam** or **Legitimate (Ham)** using TF-IDF n-grams and Multinomial Naive Bayes, packaged in an interactive Streamlit dashboard.

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-generate data or train model
python generate_data.py   # creates data/sms.csv
python train.py           # trains models, saves model.joblib & confusion_matrix.png

# 3. Launch the modern UI
streamlit run app.py      # opens at http://localhost:8501
```

---

## ✨ Features & UI Upgrades

- **🔍 Live Inspector**:
  - **Quick Preset Chips**: Test common scams (Prize Scams, Bank KYC, Job Offers, Casual Chat, Meetings, Delivery alerts) with a single click.
  - **Real-time SMS Character & Word Counter**: Estimates standard SMS segments (160 characters per part).
  - **Vibrant Risk Cards & Gauge**: Displays confidence metrics and visual status banners (Critical/Moderate risk).
  - **Explainable AI (XAI)**: Highlights the specific token n-grams swaying the Naive Bayes log-odds ratio towards Spam or Ham.
  - **Heuristic Threat Detection**: Detects phone harvesting sequences, external URLs, high-urgency keywords, and financial incentives.

- **📁 Batch SMS Scanner**:
  - Classify multiple messages at once by pasting line-by-line or uploading a CSV file.
  - Export annotated classification results directly as CSV.

- **⚙️ Adjustable Sensitivity Threshold**:
  - Control the spam probability cutoff slider (10% to 90%) in real time to tune between conservative and aggressive filtering.

- **🧠 Model Diagnostics & Viva Cheat Sheet**:
  - Displays confusion matrix, feature size, and explanations for Viva/technical evaluation.

- **📜 Session Scan History**:
  - Track messages evaluated during your session with one-click CSV export.

---

## 🛠️ How It Works

1. **Dataset**: `data/sms.csv` containing labeled SMS samples (`spam` / `ham`).
2. **Feature Engineering**: `TfidfVectorizer` with sublinear TF scaling, English stop-word removal, and unigrams + bigrams `(1, 2)`.
3. **Classification**: `MultinomialNB(alpha=0.1)` with Laplace smoothing.
4. **Evaluation**: Stratified train/test split reporting Precision, Recall, F1 score, and Confusion Matrix.

---

## 🎓 Viva & Technical Highlights

- **TF-IDF**: Penalizes common corpus words while giving high weight to discriminative words like *claim*, *urgent*, or *free*.
- **Naive Bayes**: Uses Bayes' Theorem under the conditional independence assumption. Despite being "naive", it trains rapidly and excels on sparse NLP vector spaces.
- **Precision vs. Recall**: In spam filtering, False Positives (flagging legitimate messages as spam) are typically more disruptive than False Negatives, making precision tuning essential.
