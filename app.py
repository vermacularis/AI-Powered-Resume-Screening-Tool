import streamlit as st
import os
import pandas as pd
import spacy
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def score_resumes(jd, resumes):
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform([jd] + resumes)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return scores

st.title("AI Resume Screening Tool")

jd_text = st.text_area("Paste Job Description", height=200)

uploaded_files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True, type=['pdf'])

if st.button("Rank Candidates"):
    resume_texts = [clean_text(extract_text(f)) for f in uploaded_files]
    scores = score_resumes(jd_text, resume_texts)

    results = pd.DataFrame({
        "Candidate": [f.name for f in uploaded_files],
        "Score": scores
    }).sort_values(by="Score", ascending=False)

    st.write("### Ranked Candidates")
    st.dataframe(results)

    st.download_button("Download Results as CSV", results.to_csv(index=False), "ranked_candidates.csv")
