# AI Resume Analyzer and Job Recommendation System

## Project Overview
An AI-powered application that analyzes resumes and recommends suitable job roles using Python, Streamlit, LangChain, and Ollama.

## Features
- Upload resumes in PDF format.
- Extract text from resumes.
- Analyze skills, education, projects, and experience using an LLM.
- Identify missing skills and suggest resume improvements.
- Recommend job roles based on skill matching.
- Display job match scores.

## Technologies Used
- Python
- Streamlit
- LangChain
- Ollama (Llama 3.2)
- PyPDF
- Pandas

## Installation

1. Clone the repository:
   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Navigate to the project folder:
   ```bash
   cd resume-analyzer-job-recommendation
   ```

3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

4. Install and download the Llama 3.2 model using Ollama:
   ```bash
   ollama pull llama3.2
   ```

5. Run the application:
   ```bash
   python -m streamlit run app.py
   ```

## Future Improvements
- RAG-based semantic job matching.
- Integration with real job listings.
- Resume report downloads.
- Resume-to-job comparison dashboard.

## Author
Add your name and GitHub profile here.
