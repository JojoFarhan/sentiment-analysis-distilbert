# streamlit_app.py
import streamlit as st
from model_infer import predict_sentiment

st.set_page_config(page_title="Social Sentiment (DistilBERT)", layout="centered")
st.title("Sentiment Analysis on Social Media")
st.caption("Paste a Twitter or Facebook post. We'll classify it using DistilBERT.")

text = st.text_area("Post text", height=150)
if st.button("Analyze"):
    res = predict_sentiment(text)
    st.metric("Prediction", res["label"].capitalize(), f"{res['score']*100:.2f}% confidence")
    st.write("### Class Probabilities")
    probs = {r['label'].capitalize(): r['score'] for r in res['raw']}
    st.bar_chart(probs)