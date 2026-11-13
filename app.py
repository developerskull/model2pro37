"""
Streamlit Web App: Modern, Explainable SMS Spam Detector
Built with TF-IDF + Multinomial Naive Bayes
"""
import re
import time
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Shield • AI Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Modern CSS (Glassmorphism & Clean Card Design)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero header */
    .hero-container {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(147, 51, 234, 0.08) 100%);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Verdict Result Banners */
    .verdict-card {
        padding: 1.5rem;
        border-radius: 14px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.4s ease-in-out;
    }
    .verdict-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .verdict-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 2px solid #10b981;
        color: #065f46;
    }
    .verdict-title {
        font-size: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }

    /* Heuristic Pill Badges */
    .badge {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
    .badge-safe {
        background-color: #d1fae5;
        color: #047857;
        border: 1px solid #6ee7b7;
    }
    .badge-neutral {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }

    /* Word influence chips */
    .word-chip-spam {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px;
        border-left: 3px solid #ef4444;
    }
    .word-chip-ham {
        display: inline-block;
        background-color: #d1fae5;
        color: #065f46;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 2px 4px;
        border-left: 3px solid #10b981;
    }

    /* Metric cards */
    .stat-box {
        background-color: rgba(248, 250, 252, 0.8);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load Model & Explainability Helpers
# ---------------------------------------------------------
@st.cache_resource
def load_classifier():
    return joblib.load("model.joblib")

model = load_classifier()
tfidf = model.named_steps.get("tfidf")
nb_clf = model.named_steps.get("clf")

# Precompute log-odds for fast token explainability (Spam vs Ham)
feature_names = tfidf.get_feature_names_out()
log_odds_vector = nb_clf.feature_log_prob_[1] - nb_clf.feature_log_prob_[0]
feature_to_logodds = dict(zip(feature_names, log_odds_vector))

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


# ---------------------------------------------------------
# Heuristic & Feature Extraction Functions
# ---------------------------------------------------------
def analyze_heuristics(text: str):
    """Detect heuristic spam indicators (URLs, numbers, urgency, symbols)."""
    flags = []
    # URL detection
    urls = re.findall(r"https?://\S+|www\.\S+|bit\.ly/\S+|goo\.gl/\S+", text, re.IGNORECASE)
    if urls:
        flags.append(("Link / URL Included", "danger", f"Found {len(urls)} link(s)"))

    # Phone / long digit detection
    phone_numbers = re.findall(r"\b\d{10,12}\b|\b\d{4,5}[-\s]\d{5,6}\b", text)
    if phone_numbers:
        flags.append(("Direct Phone / Hotline", "danger", f"{len(phone_numbers)} phone sequence(s)"))

    # Currency and monetary signs
    money = re.findall(r"[\$₹£€]\s*\d+|\d+\s*(?:dollars|rupees|pounds|cash|prize|reward|usd|inr)", text, re.IGNORECASE)
    if money:
        flags.append(("Financial / Cash Incentive", "danger", "Monetary terms present"))

    # Urgency keywords
    urgency_words = ["urgent", "immediately", "expire", "action required", "blocked", "suspended", "verify kyc", "hurry"]
    matched_urgency = [w for w in urgency_words if re.search(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE)]
    if matched_urgency:
        flags.append(("High Urgency Triggers", "danger", ", ".join(matched_urgency)))

    # Capitalization ratio
    letters = [c for c in text if c.isalpha()]
    caps_ratio = (sum(1 for c in letters if c.isupper()) / max(len(letters), 1)) if letters else 0
    if caps_ratio > 0.4 and len(text) > 20:
        flags.append(("Excessive CAPITAL LETTERS", "danger", f"{caps_ratio:.0%} caps"))

    return flags


def get_influential_words(text: str, top_k: int = 5):
    """Find words that contributed most to Spam or Ham verdict."""
    vec = tfidf.transform([text]).toarray()[0]
    active_indices = vec.nonzero()[0]
    
    scored_tokens = []
    for idx in active_indices:
        token = feature_names[idx]
        score = vec[idx] * feature_to_logodds[token]
        scored_tokens.append((token, score))

    scored_tokens.sort(key=lambda x: x[1], reverse=True)
    spam_words = [t for t in scored_tokens if t[1] > 0][:top_k]
    ham_words = [t for t in sorted(scored_tokens, key=lambda x: x[1]) if t[1] < 0][:top_k]

    return spam_words, ham_words


# ---------------------------------------------------------
# Sidebar Controls & Information
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.markdown("### **Spam Shield Control**")
    st.caption("AI-powered NLP message security system")

    st.markdown("---")
    st.subheader("⚙️ Classification Settings")
    threshold = st.slider(
        "Spam Threshold Sensitivity",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Messages with spam confidence greater than or equal to this threshold will be flagged as SPAM.",
    )

    st.info(
        f"**Active Threshold:** `{threshold:.0%}`\n\n"
        "Lowering threshold makes the detector stricter (catches more spam, might flag ham). "
        "Raising it makes it conservative.",
        icon="ℹ️"
    )

    st.markdown("---")
    st.subheader("📊 Session Statistics")
    total_scanned = len(st.session_state.history)
    spam_scanned = sum(1 for h in st.session_state.history if h["verdict"] == "SPAM")
    ham_scanned = total_scanned - spam_scanned

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Total Scanned", total_scanned)
    with col_s2:
        st.metric("Spam Detected", spam_scanned)

    if total_scanned > 0:
        if st.button("🧹 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.caption("Developed with scikit-learn & Streamlit • Multinomial Naive Bayes")


# ---------------------------------------------------------
# Main Page Header (Hero Banner)
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h1 class="hero-title">🛡️ SMS Spam Shield AI</h1>
                <p class="hero-subtitle">Real-time SMS security analyzer using TF-IDF n-grams and Multinomial Naive Bayes</p>
            </div>
            <div>
                <span class="badge badge-safe">⚡ Model Online</span>
                <span class="badge badge-neutral">Accuracy: ~98%</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab_inspect, tab_batch, tab_model, tab_history = st.tabs([
    "🔍 Live Inspector",
    "📁 Batch SMS Scanner",
    "🧠 Model Diagnostics & Viva",
    "📜 Scan History",
])

# =========================================================
# TAB 1: LIVE INSPECTOR
# =========================================================
with tab_inspect:
    st.markdown("#### ⚡ Quick Presets")
    st.caption("Click any sample card below to test common SMS patterns immediately:")

    PRESETS = {
        "🎁 Prize Scam": "Congratulations! You have won a free iPhone 15 Pro Max. Call 09876543210 immediately to claim your gift now!",
        "🏦 Fake KYC / Banking": "URGENT: Your SBI Bank KYC has expired today. Your net banking will be suspended. Update now at http://sbi-kyc-verify.in",
        "💼 Job Scam": "Earn $500/day working from home part-time! No experience required. WhatsApp your resume to 9876543210 for instant joining.",
        "💬 Friend Chat": "Hey! Are you heading to the lecture today? Can you please bring the lecture notes from yesterday?",
        "📅 Meeting Alert": "Hi Team, reminder for the project standup scheduled today at 3:30 PM in Conference Room B. See you there.",
        "📦 Delivery Notification": "Your package has been delivered to your front door. Thank you for shopping with us! Order #84729.",
    }

    # Preset quick-click buttons
    p_cols = st.columns(len(PRESETS))
    for idx, (label, sample_text) in enumerate(PRESETS.items()):
        with p_cols[idx]:
            if st.button(label, key=f"btn_preset_{idx}", use_container_width=True):
                st.session_state.input_text = sample_text

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Input form
    col_input, col_meta = st.columns([2.5, 1])

    with col_input:
        user_sms = st.text_area(
            "Enter SMS Text to inspect:",
            value=st.session_state.input_text,
            height=140,
            placeholder="Type or paste any SMS message here...",
            help="Type an incoming SMS or select one of the presets above.",
        )

    with col_meta:
        st.markdown("**Message Metadata**")
        char_count = len(user_sms)
        word_count = len(user_sms.split()) if user_sms else 0
        sms_parts = (char_count // 160) + 1 if char_count > 0 else 0

        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-number">{char_count}</div>
                <div class="stat-label">Characters ({sms_parts} SMS part{'s' if sms_parts > 1 else ''})</div>
                <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #e2e8f0;">
                <div class="stat-number">{word_count}</div>
                <div class="stat-label">Word Count</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Action buttons
    col_act1, col_act2, col_act_spacer = st.columns([1, 1, 3])
    with col_act1:
        run_analysis = st.button("🛡️ Analyze Message", type="primary", use_container_width=True)
    with col_act2:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.input_text = ""
            st.rerun()

    # Trigger analysis on button click or if text exists and preset was clicked
    if run_analysis or (st.session_state.input_text and user_sms.strip()):
        cleaned_text = user_sms.strip()
        if not cleaned_text:
            st.warning("Please type a message or select an example preset first.")
        else:
            with st.spinner("Analyzing message semantics and probability distribution..."):
                proba = model.predict_proba([cleaned_text])[0]
                p_ham, p_spam = float(proba[0]), float(proba[1])
                is_spam = p_spam >= threshold

            # Save to session history (avoid duplicate consecutive records)
            if not st.session_state.history or st.session_state.history[-1]["text"] != cleaned_text:
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "text": cleaned_text,
                    "verdict": "SPAM" if is_spam else "LEGITIMATE (HAM)",
                    "spam_proba": p_spam,
                    "confidence": p_spam if is_spam else p_ham,
                })

            st.markdown("---")
            st.subheader("🎯 Classification Results")

            # Layout: Verdict Banner + Probability Meter
            c_result, c_gauge = st.columns([1.6, 1.2])

            with c_result:
                if is_spam:
                    st.markdown(
                        f"""
                        <div class="verdict-card verdict-spam">
                            <div class="verdict-title">🚨 HIGH RISK: SPAM DETECTED</div>
                            <p style="margin: 0; font-size: 0.95rem; font-weight: 500;">
                                This message contains characteristics commonly associated with scams, phishing, or unsolicited marketing.
                            </p>
                            <div style="margin-top: 0.8rem;">
                                <span class="badge badge-danger">Confidence: {p_spam:.1%}</span>
                                <span class="badge badge-danger">Risk Level: {'CRITICAL' if p_spam > 0.8 else 'MODERATE'}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="verdict-card verdict-ham">
                            <div class="verdict-title">✅ SAFE: LEGITIMATE MESSAGE</div>
                            <p style="margin: 0; font-size: 0.95rem; font-weight: 500;">
                                This message appears to be safe and personal communication with no malicious triggers.
                            </p>
                            <div style="margin-top: 0.8rem;">
                                <span class="badge badge-safe">Confidence: {p_ham:.1%}</span>
                                <span class="badge badge-safe">Status: Clean (Ham)</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with c_gauge:
                st.markdown("**Spam Risk Meter**")
                st.progress(p_spam, text=f"Spam Probability: {p_spam:.1%}")

                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b; margin-top: 4px;">
                        <span>0% Safe</span>
                        <span>Threshold ({threshold:.0%})</span>
                        <span>100% Spam</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.metric("Spam Score", f"{p_spam:.1%}")
                with col_g2:
                    st.metric("Legit Score", f"{p_ham:.1%}")

            # Heuristic & Rule-based Detection
            st.markdown("#### 🔍 Security Signals & Heuristics")
            heuristics = analyze_heuristics(cleaned_text)
            if heuristics:
                h_cols = st.columns(min(len(heuristics), 4))
                for h_idx, (name, level, detail) in enumerate(heuristics):
                    with h_cols[h_idx % 4]:
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #fecaca; background-color: #fff5f5; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
                                <div style="font-weight: 700; color: #b91c1c; font-size: 0.85rem;">⚠️ {name}</div>
                                <div style="font-size: 0.78rem; color: #7f1d1d; margin-top: 0.2rem;">{detail}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    """
                    <div style="border: 1px solid #bbf7d0; background-color: #f0fdf4; border-radius: 8px; padding: 0.75rem; color: #166534; font-size: 0.85rem;">
                        ✨ <b>No heuristic red flags detected:</b> No suspicious external URLs, emergency keywords, or phone harvesting sequences found.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Explainable AI: Influential Words
            st.markdown("#### 🧠 Explainable AI: Key Words Influencing Verdict")
            st.caption("How specific words swayed the Naive Bayes log-odds decision:")

            spam_tokens, ham_tokens = get_influential_words(cleaned_text)
            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                st.markdown("**Tokens Pushing Towards SPAM (Red Flags):**")
                if spam_tokens:
                    chips_html = "".join([f'<span class="word-chip-spam">{t} (+{s:.2f})</span>' for t, s in spam_tokens])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.caption("No significant spam-indicative tokens identified.")

            with col_exp2:
                st.markdown("**Tokens Pushing Towards HAM (Safe Flags):**")
                if ham_tokens:
                    chips_html = "".join([f'<span class="word-chip-ham">{t} ({s:.2f})</span>' for t, s in ham_tokens])
                    st.markdown(chips_html, unsafe_allow_html=True)
                else:
                    st.caption("No significant ham-indicative tokens identified.")


# =========================================================
# TAB 2: BATCH SMS SCANNER
# =========================================================
with tab_batch:
    st.markdown("### 📁 Batch Message Classification")
    st.caption("Scan multiple SMS messages at once via bulk text entry or CSV file upload.")

    batch_mode = st.radio("Choose input method:", ["Paste Multiple Messages", "Upload CSV File"], horizontal=True)

    if batch_mode == "Paste Multiple Messages":
        raw_batch = st.text_area(
            "Enter SMS messages (one message per line):",
            height=160,
            placeholder="Congratulations! You have won $1,000 cash.\nHey are we still meeting today?\nYour OTP code is 48291. Do not share.",
        )
        if st.button("🚀 Process Batch Messages", type="primary"):
            lines = [l.strip() for l in raw_batch.split("\n") if l.strip()]
            if not lines:
                st.warning("Please provide at least one message.")
            else:
                probs = model.predict_proba(lines)[:, 1]
                batch_df = pd.DataFrame({
                    "Message": lines,
                    "Spam Probability": probs,
                    "Verdict": ["SPAM" if p >= threshold else "HAM" for p in probs],
                })

                st.success(f"Successfully processed {len(lines)} messages!")
                st.dataframe(
                    batch_df.style.format({"Spam Probability": "{:.1%}"}),
                    use_container_width=True,
                )

                csv_data = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results as CSV",
                    csv_data,
                    "sms_batch_scan_results.csv",
                    "text/csv",
                    use_container_width=True,
                )

    else:
        uploaded_file = st.file_uploader("Upload a CSV file containing SMS texts", type=["csv"])
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(user_df.head(4), use_container_width=True)

            text_col = st.selectbox("Select the column containing the SMS text:", user_df.columns)
            if st.button("🚀 Run Classification on CSV", type="primary"):
                texts = user_df[text_col].astype(str).tolist()
                probs = model.predict_proba(texts)[:, 1]
                user_df["Spam_Probability"] = probs
                user_df["Predicted_Verdict"] = ["SPAM" if p >= threshold else "HAM" for p in probs]

                st.success(f"Classified {len(texts)} rows!")
                st.dataframe(user_df.head(20), use_container_width=True)

                out_csv = user_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Annotated CSV",
                    out_csv,
                    "annotated_sms_predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )


# =========================================================
# TAB 3: MODEL DIAGNOSTICS & VIVA GUIDE
# =========================================================
with tab_model:
    st.markdown("### 🧠 Model Architecture & Technical Specs")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-number">MultinomialNB</div>
                <div class="stat-label">Algorithm</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""
            <div class="stat-box">
                <div class="stat-number">{len(feature_names)}</div>
                <div class="stat-label">TF-IDF Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m3:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-number">(1, 2)</div>
                <div class="stat-label">N-gram Range</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            """
            <div class="stat-box">
                <div class="stat-number">α = 0.1</div>
                <div class="stat-label">Laplace Smoothing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_cm, col_qa = st.columns([1.2, 1.8])

    with col_cm:
        st.markdown("#### 📊 Confusion Matrix")
        try:
            st.image("confusion_matrix.png", caption="Model Confusion Matrix on Test Split", use_container_width=True)
        except Exception:
            st.info("Run `python train.py` to regenerate the confusion matrix plot.")

    with col_qa:
        st.markdown("#### 🎓 Viva / Technical Questions Cheat Sheet")
        with st.expander("Why use TF-IDF instead of simple CountVectorizer?"):
            st.write(
                "TF-IDF penalizes commonly occurring words across the whole corpus (like 'the', 'is', 'at') "
                "while boosting the importance of discriminative words like 'claim', 'urgent', or 'congratulations'."
            )
        with st.expander("Why is Naive Bayes called 'Naive'?"):
            st.write(
                "It assumes that all features (words) are conditionally independent given the class. "
                "In reality, words like 'free' and 'gift' often appear together, but despite this simplistic assumption, "
                "Naive Bayes performs remarkably well for text classification and requires very little training time."
            )
        with st.expander("Why is Accuracy alone misleading for Spam Detection?"):
            st.write(
                "Spam is typically an imbalanced class (often 10-25% of total messages). "
                "A naive model predicting everything as 'HAM' could still have 85% accuracy. "
                "Therefore, Precision (minimizing false alarms) and Recall (catching actual spam) are critical metrics."
            )
        with st.expander("What is Laplace smoothing (alpha=0.1)?"):
            st.write(
                "If an unseen word appears at test time, its frequency would be zero, making the overall posterior probability zero. "
                "Laplace smoothing adds a small pseudo-count (alpha) so zero-frequency probabilities don't wipe out the score."
            )


# =========================================================
# TAB 4: SCAN HISTORY
# =========================================================
with tab_history:
    st.markdown("### 📜 Session Scan History")
    st.caption("Review all messages tested during this session:")

    if not st.session_state.history:
        st.info("No messages have been scanned yet in this session. Analyze some in the Live Inspector tab!")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            history_df.style.format({
                "spam_proba": "{:.1%}",
                "confidence": "{:.1%}",
            }),
            use_container_width=True,
        )

        csv_hist = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export History as CSV",
            csv_hist,
            "sms_scan_history.csv",
            "text/csv",
            use_container_width=True,
        )
