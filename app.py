import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("AI Resume Analyzer 🔍")
st.write("Smart Resume Screening + Hiring Decision System")

# ---------------------- CLEAN TEXT ----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text

# ---------------------- EXTRACT TEXT ----------------------
def extract_text(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except:
        return ""
    return text

# ---------------------- JOB ROLES ----------------------
job_roles = {
    "Data Scientist": [
        "python","machine learning","deep learning","nlp","statistics",
        "data analysis","pandas","numpy","scikit learn","sql",
        "matplotlib","seaborn","tableau","power bi","linear algebra"
    ],
    "Web Developer": [
        "html","css","javascript","react","node","express",
        "mongodb","mysql","django","flask"
    ],
    "Android Developer": [
        "java","kotlin","android","xml","firebase","api"
    ]
}

# ---------------------- UI ----------------------
role = st.selectbox("Select Job Role", list(job_roles.keys()))
cutoff = st.slider("Select Cutoff (%)", 0, 100, 50)

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

# ---------------------- MAIN LOGIC ----------------------
if uploaded_file is not None:

    st.info("Processing resume...")

    resume_text = extract_text(uploaded_file)

    if resume_text.strip() == "":
        st.error("❌ Could not extract text. Try another PDF.")
    else:
        resume_text = clean_text(resume_text)

        # Convert job skills into text
        job_text = " ".join(job_roles[role])

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, job_text])

        # Cosine Similarity
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        score = round(similarity * 100, 2)

        st.subheader("📊 Match Score")
        st.progress(int(score))
        st.write(f"**Score: {score}%**")

        # ---------------------- DECISION ----------------------
        if score >= cutoff:
            st.success("✅ Selected for Interview")
        elif score >= cutoff - 10:
            st.warning("⚠ Near Selection (Consider Internship)")
        else:
            st.error("❌ Rejected")

        # ---------------------- SKILL MATCH ----------------------
        st.subheader("🧠 Skill Analysis")

        matched_skills = []
        for skill in job_roles[role]:
            if skill in resume_text:
                matched_skills.append(skill)

        st.write("**Matched Skills:**", matched_skills if matched_skills else "None")

        missing_skills = list(set(job_roles[role]) - set(matched_skills))
        st.write("**Missing Skills:**", missing_skills if missing_skills else "None")
