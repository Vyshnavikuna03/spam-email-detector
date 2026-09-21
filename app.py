import streamlit as st
import joblib
from train import clean

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

@st.cache_resource
def load_model():
    return joblib.load("models/spam_model.pkl")

model = load_model()

st.title("📧 Spam Email Detector")
st.caption("SVM + TF-IDF, trained on ~83k emails (TREC + Enron)")

text = st.text_area("Paste the email text here", height=220)

if st.button("Check"):
    if text.strip():
        if model.predict([clean(text)])[0] == 1:
            st.error("🚨 This looks like SPAM")
        else:
            st.success("✅ This looks legitimate")
    else:
        st.warning("Please paste some text first")

st.markdown("---")
st.caption("Note: the model was trained on older email data, so it may miss newer spam styles.")