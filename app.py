
import os
import tempfile

import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama


# -----------------------------
# Streamlit configuration
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer & Job Recommendation")
st.write(
    "Upload your resume to analyze your skills "
    "and discover suitable job roles."
)


# -----------------------------
# Local LLM
# -----------------------------
@st.cache_resource
def load_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0.2
    )


# -----------------------------
# Sample job database
# -----------------------------
jobs = [
    {
        "title": "Python Developer",
        "skills": [
            "python", "django", "flask", "sql",
            "git", "rest api"
        ],
        "experience": "Entry-level"
    },
    {
        "title": "Data Analyst",
        "skills": [
            "python", "sql", "pandas", "excel",
            "power bi", "statistics"
        ],
        "experience": "Entry-level"
    },
    {
        "title": "Full Stack Developer",
        "skills": [
            "python", "javascript", "react",
            "node.js", "mongodb", "html", "css"
        ],
        "experience": "Entry-level"
    },
    {
        "title": "Machine Learning Engineer",
        "skills": [
            "python", "pandas", "numpy",
            "scikit-learn", "machine learning",
            "statistics"
        ],
        "experience": "Entry-level"
    },
    {
        "title": "Generative AI Developer",
        "skills": [
            "python", "langchain", "llm",
            "rag", "embeddings", "vector database"
        ],
        "experience": "Entry-level"
    }
]


# -----------------------------
# Extract resume text
# -----------------------------
def extract_resume(uploaded_file):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        pages = PyPDFLoader(temp_path).load()

        text = "\n".join(
            page.page_content for page in pages
        )

        return text

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# -----------------------------
# Analyze resume with LLM
# -----------------------------
def analyze_resume(resume_text):
    llm = load_llm()

    prompt = f"""
    You are an expert resume reviewer and career advisor.

    Analyze the resume below.

    Provide these sections:
    1. Professional summary
    2. Technical skills
    3. Education
    4. Projects and experience
    5. Strengths
    6. Missing or weak areas
    7. Suggestions to improve the resume

    Use only information supported by the resume.
    If information is unavailable, say "Not specified".
    Do not invent qualifications or experience.

    RESUME:
    {resume_text[:12000]}
    """

    response = llm.invoke(prompt)
    return response.content


# -----------------------------
# Recommend matching jobs
# -----------------------------
def recommend_jobs(resume_text):
    resume_lower = resume_text.lower()
    recommendations = []

    for job in jobs:
        matched = [
            skill for skill in job["skills"]
            if skill.lower() in resume_lower
        ]

        missing = [
            skill for skill in job["skills"]
            if skill.lower() not in resume_lower
        ]

        score = round(
            len(matched) / len(job["skills"]) * 100
        )

        recommendations.append({
            "Job Role": job["title"],
            "Match Score (%)": score,
            "Matched Skills": ", ".join(matched) or "None detected",
            "Missing Skills": ", ".join(missing) or "None detected",
            "Experience": job["experience"]
        })

    return pd.DataFrame(recommendations).sort_values(
        by="Match Score (%)",
        ascending=False
    )


# -----------------------------
# User interface
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

if uploaded_file:
    if st.button("Analyze Resume", type="primary"):
        try:
            with st.spinner("Reading your resume..."):
                resume_text = extract_resume(uploaded_file)

            if not resume_text.strip():
                st.error(
                    "No readable text was found. "
                    "This PDF may be scanned; OCR is needed."
                )
                st.stop()

            st.session_state["resume_text"] = resume_text

            with st.spinner("Analyzing resume with Llama 3.2..."):
                analysis = analyze_resume(resume_text)

            st.session_state["analysis"] = analysis
            st.session_state["recommendations"] = (
                recommend_jobs(resume_text)
            )

        except Exception as error:
            st.error(
                "Could not analyze the resume. "
                "Check that Ollama is running and the "
                "llama3.2 model is installed."
            )
            st.caption(str(error))


# -----------------------------
# Display results
# -----------------------------
if "analysis" in st.session_state:
    tab1, tab2 = st.tabs([
        "AI Resume Analysis",
        "Job Recommendations"
    ])

    with tab1:
        st.subheader("Your Resume Review")
        st.markdown(st.session_state["analysis"])

    with tab2:
        st.subheader("Recommended Job Roles")

        result_df = st.session_state["recommendations"]

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Match scores are keyword-based estimates, "
            "not guaranteed hiring probabilities."
        )

        if not result_df.empty:
            selected_job = st.selectbox(
                "Choose a job role",
                result_df["Job Role"].tolist()
            )

            job_row = result_df[
                result_df["Job Role"] == selected_job
            ].iloc[0]

            st.metric(
                "Estimated Skills Match",
                f'{job_row["Match Score (%)"]}%'
            )

            st.write("**Skills already found:**")
            st.write(job_row["Matched Skills"])

            st.write("**Skills to learn or demonstrate:**")
            st.write(job_row["Missing Skills"])
