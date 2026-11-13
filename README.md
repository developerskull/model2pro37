# SMS Spam Detector

A simple NLP mini project: classify an SMS as spam or not spam using TF-IDF and Naive Bayes, with a Streamlit web app.

## Run it

```bash
pip install -r requirements.txt
python generate_data.py   # creates data/sms.csv (skip if you use your own CSV)
python train.py           # trains the models and saves model.joblib
streamlit run app.py      # opens the web app at http://localhost:8501
```

## How it works

1. **Data**: `data/sms.csv` with columns `label` (spam/ham) and `text`. 2000 messages, 25% spam.
2. **Text to numbers**: `TfidfVectorizer` with English stop words removed and unigrams plus bigrams.
3. **Models**: Multinomial Naive Bayes (used in the app) compared with Logistic Regression.
4. **Evaluation**: 80/20 stratified split, reporting precision, recall, F1 and a confusion matrix (`confusion_matrix.png`).

## IMPORTANT: use real data before you present or submit

The included data is generated from templates by `generate_data.py`, so both models score about 100%. That number is NOT a real result. Real SMS spam gets about 97-98% accuracy with this method.

To get real numbers:
1. Download the "SMS Spam Collection" dataset from Kaggle or the UCI repository (about 5,500 messages).
2. Save it as `data/sms.csv` with the columns `label` and `text` (rename them if needed).
3. Run `python train.py` again and put your real metrics in this README.

## Viva points

- **TF-IDF**: gives a word a high score if it is frequent in a message but rare across all messages, so "free" and "claim" matter more than "the".
- **Naive Bayes**: uses Bayes' theorem and assumes words are independent given the class. The assumption is "naive" but it works very well for text and trains in milliseconds.
- **Why not just accuracy**: spam is the minority class, so precision (not flagging real messages) and recall (catching spam) matter too.
- **False positive vs false negative**: marking a real message as spam is usually worse than letting some spam through.

## Ideas to extend

- Add a "top spam words" chart from the model's learned weights
- Try an SVM or a small BERT model and compare
- Add a threshold slider in the app
- Deploy on Streamlit Community Cloud
