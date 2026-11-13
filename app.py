"""Streamlit web app: paste an SMS, find out if it is spam."""
import joblib
import streamlit as st

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("📩 SMS Spam Detector")
st.caption("TF-IDF + Naive Bayes classifier")


@st.cache_resource
def load():
    return joblib.load("model.joblib")


model = load()

EXAMPLES = {
    "Pick an example...": "",
    "Prize scam": "Congratulations! You have won a free iPhone. Call 09876543210 now to claim.",
    "Fake KYC": "Your KYC is expired. Update immediately at http://sbi-verify.co.in or your account will be suspended",
    "Normal message": "Hey, are you coming to college tomorrow? Please bring the notes.",
}
choice = st.selectbox("Try an example", list(EXAMPLES))
text = st.text_area("Message", value=EXAMPLES[choice], height=140, placeholder="Paste an SMS here...")

if st.button("Check message", type="primary"):
    if not text.strip():
        st.warning("Please enter a message first.")
    else:
        p_spam = float(model.predict_proba([text])[0][1])
        if p_spam >= 0.5:
            st.error(f"🚨 SPAM  (confidence {p_spam:.0%})")
        else:
            st.success(f"✅ Not spam  (confidence {1 - p_spam:.0%})")
        st.progress(p_spam, text=f"Spam probability: {p_spam:.0%}")
        st.caption("Trained on generated sample data, so treat results as a demo.")
