import streamlit as st
import pdfplumber
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- BERT MODEL ----------------
vectorizer = TfidfVectorizer()
uploaded_file = st.file_uploader(...)
resume_text = extract_text(uploaded_file)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume_text])

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text

# ---------------- EXTRACT PDF TEXT ----------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# ---------------- SKILL DICTIONARY ----------------
job_roles = {
    "Data Scientist": [
        "python","machine learning","deep learning","nlp","statistics",
        "data analysis","pandas","numpy","scikit learn","sql",
        "data visualization","matplotlib","seaborn","tableau",
        "power bi","linear algebra"
    ],

    "Web Developer": [
        "html","css","javascript","react","node","express",
        "mongodb","mysql","api","frontend","backend"
    ],

    "AI Engineer": [
        "deep learning","neural networks","cnn","rnn","transformers",
        "tensorflow","pytorch","nlp","computer vision"
    ],

    "Java Developer": [
        "java","spring","spring boot","hibernate","jdbc",
        "multithreading","collections","oops"
    ],

    "ML Engineer": [
        "machine learning","model deployment","mlops","docker",
        "kubernetes","feature engineering","pipeline","scikit learn"
    ]
}

# ---------------- UI ----------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("AI Resume Analyzer 🔍")
st.write("Smart Resume Screening + Hiring Decision System")

role = st.selectbox("Select Job Role", list(job_roles.keys()))
cutoff = st.slider("Select Cutoff (%)", 0, 100, 30)

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# ---------------- MAIN LOGIC ----------------
if uploaded_file:

    st.success("Resume uploaded successfully!")

    raw_text = extract_text(uploaded_file)
    text = clean_text(raw_text)

    job_desc = " ".join(job_roles[role])

    # ---------- TF-IDF ----------
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    vectors = vectorizer.fit_transform([text, job_desc])
    tfidf_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # ---------- BERT ----------
    emb1 = bert_model.encode(text)
    emb2 = bert_model.encode(job_desc)
    bert_score = cosine_similarity([emb1], [emb2])[0][0]

    # ---------- FINAL SCORE ----------
    similarity = max(tfidf_score, bert_score)
    score = round(similarity * 100, 2)

    # ---------- SKILL MATCH ----------
    skills = job_roles[role]
    matched = [s for s in skills if s in text]
    missing = [s for s in skills if s not in text]

    # ---------- RESULT ----------
    st.subheader("Match Score")
    st.progress(int(score))
    st.write(f"{score} %")

    if score >= cutoff:
        st.success("Selected ✅")
        show_questions = True
    elif score >= (cutoff - 10):
        st.warning("Near selection (Can be selected for Intership)")
        show_questions = True
  

    else:
        st.error("Not Selected ❌")
        show_questions = False

    # ---------- SKILLS ----------
    st.subheader("Matched Skills")
    st.write(", ".join(matched))

    st.subheader("Missing Skills")
    st.write(", ".join(missing))

    # ---------- QUESTIONS ----------
    if show_questions:

        st.subheader("Suggested Interview Questions 📚")

        if role == "Data Scientist":
            questions = [
                "Explain data science lifecycle",
                "What is EDA?",
                "How do you handle missing data?",
                "Explain feature selection",
                "What is hypothesis testing?",
                "Difference between classification and regression",
                "Explain precision and recall",
                "What is overfitting?"
            ]

        elif role == "Web Developer":
            questions = [
                "Explain HTML, CSS, JS",
                "What is DOM?",
                "What is REST API?",
                "Frontend vs Backend",
                "Explain React",
                "What is responsive design?"
            ]

        elif role == "AI Engineer":
            questions = [
                "What is deep learning?",
                "Explain neural networks",
                "What is CNN?",
                "What is RNN?",
                "Explain transformers"
            ]

        elif role == "Java Developer":
            questions = [
                "Explain OOP concepts",
                "What is JVM?",
                "What is Spring Boot?",
                "Explain multithreading"
            ]

        elif role == "ML Engineer":
            questions = [
                "What is MLOps?",
                "What is model deployment?",
                "Explain pipelines",
                "What is feature engineering"
            ]

        for q in questions:
            st.write(f"- {q}")

    # ---------- EXTRACTED TEXT ----------
    st.subheader("Extracted Text")
    st.write(raw_text)
