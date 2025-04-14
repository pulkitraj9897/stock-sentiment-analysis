import streamlit as st
import pandas as pd
from transformers import pipeline
import os

# Prevent tokenizer multithreading warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Title
st.title("📊 Sentiment Analysis Dashboard")

# Cache the sentiment analyzer
@st.cache_resource
def get_analyzer():
    return pipeline("sentiment-analysis", device=-1)


analyzer = get_analyzer()

# Load Kaggle dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/stock_sentiment.csv")  # Ensure this path is valid

data = load_data()

# Show raw data (optional)
if st.checkbox("Show raw data"):
    st.write(data)

# Separator
st.markdown("---")

# Sentiment analysis input
st.subheader("🔍 Analyze Sentiment")
text_input = st.text_area("Or paste your own text here:")

if text_input:
    try:
        result = analyzer(text_input)[0]
        emoji = "😊" if result["label"] == "POSITIVE" else "😠" if result["label"] == "NEGATIVE" else "😐"
        st.write(f"**Sentiment:** {emoji} {result['label']} (Confidence: {result['score']:.2f})")
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")

# Separator
st.markdown("---")

# Display sample sentiments from dataset
st.subheader("📜 Sample Sentiments from Dataset")
st.write(data.head(10))  # Show first 10 entries

# Add a refresh button
if st.button("🔄 Refresh Data"):
    data = load_data()
    st.experimental_rerun()
