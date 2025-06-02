import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load data from JSONL
@st.cache_data
def load_data():
    with open("sanskrit_advanced_nlp.jsonl", "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f]
    return pd.DataFrame(lines)

df = load_data()

# Vectorization
vectorizer = TfidfVectorizer(max_features=300)
lsa = TruncatedSVD(n_components=5)

corpus = df["lemmatized"].tolist()
tfidf_matrix = vectorizer.fit_transform(corpus)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# UI
st.title("🕉️ Sanskrit AI Rigveda Verse Finder")
query = st.text_input("🔍 Enter a Sanskrit word or root (e.g., 'अग्नि', 'सोम'):")

if query:
    query_vec = lsa.transform(vectorizer.transform([query]))
    scores = lsa_matrix.dot(query_vec.T).flatten()
    df["score"] = scores
    results = df.sort_values("score", ascending=False).head(5)

    st.write("### 🔎 Top Matches:")
    for i, row in results.iterrows():
        st.markdown(f"**📜 Sanskrit:** {row['prompt']}")
        st.markdown(f"**🌐 Translation:** {row['completion']}")
        st.markdown("---")

