import streamlit as st
from sentence_transformers import SentenceTransformer
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Load Deep Learning Model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🤖 AI Resume Analyzer & Builder")

st.write(
    "Analyze your resume using Deep Learning and "
    "compare it with a Job Description."
)

st.divider()

# -----------------------------
# Personal Information
# -----------------------------
st.header("👤 Personal Information")

name = st.text_input("Full Name")
email = st.text_input("Email")
phone = st.text_input("Phone Number")
location = st.text_input("Location")

# -----------------------------
# Education
# -----------------------------
st.header("🎓 Education")

education = st.text_area(
    "Education",
    placeholder="Example: B.Tech in Artificial Intelligence and Machine Learning"
)

# -----------------------------
# Skills
# -----------------------------
st.header("💻 Skills")

skills = st.text_area(
    "Skills",
    placeholder="Example: Python, Machine Learning, SQL, Deep Learning, Java"
)

# -----------------------------
# Projects
# -----------------------------
st.header("📁 Projects")

projects = st.text_area(
    "Projects",
    placeholder="Example: AI Personal Finance Tracker"
)

# -----------------------------
# Job Description
# -----------------------------
st.header("💼 Job Description")

job_description = st.text_area(
    "Paste the Job Description here",
    height=180
)

# -----------------------------
# Analyze Resume
# -----------------------------
if st.button("🔍 Analyze Resume", type="primary"):

    if not name or not skills or not job_description:

        st.warning(
            "Please enter your Name, Skills and Job Description."
        )

    else:

        # Combine resume information
        resume_text = f"""
        Name: {name}
        Education: {education}
        Skills: {skills}
        Projects: {projects}
        """

        # -----------------------------
        # Deep Learning Embeddings
        # -----------------------------
        resume_embedding = model.encode(resume_text)
        job_embedding = model.encode(job_description)

        # Calculate semantic similarity
        similarity = model.similarity(
            resume_embedding,
            job_embedding
        )

        score = float(similarity[0][0]) * 100

        # Keep score between 0 and 100
        score = max(0, min(score, 100))

        # -----------------------------
        # Display Score
        # -----------------------------
        st.divider()

        st.header("📊 Resume Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Resume Match Score",
                f"{score:.2f}%"
            )

        with col2:

            if score >= 80:
                result = "Excellent"
            elif score >= 60:
                result = "Good"
            elif score >= 40:
                result = "Average"
            else:
                result = "Needs Improvement"

            st.metric("Resume Quality", result)

        with col3:
            st.metric(
                "AI Model",
                "Transformer"
            )

        # -----------------------------
        # Skill Analysis
        # -----------------------------
        st.subheader("🎯 Skill Analysis")

        common_skills = [
            "python",
            "java",
            "sql",
            "machine learning",
            "deep learning",
            "tensorflow",
            "pytorch",
            "nlp",
            "data science",
            "javascript",
            "html",
            "css",
            "c++",
            "c#",
            "git",
            "github"
        ]

        resume_lower = resume_text.lower()
        job_lower = job_description.lower()

        required_skills = []
        missing_skills = []
        matched_skills = []

        for skill in common_skills:

            if skill in job_lower:

                required_skills.append(skill)

                if skill in resume_lower:
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)

        col1, col2 = st.columns(2)

        with col1:

            st.write("### ✅ Matched Skills")

            if matched_skills:
                for skill in matched_skills:
                    st.success(skill)
            else:
                st.write("No matched skills detected.")

        with col2:

            st.write("### ❌ Missing Skills")

            if missing_skills:
                for skill in missing_skills:
                    st.error(skill)
            else:
                st.success("No major missing skills!")

        # -----------------------------
        # Professional Summary
        # -----------------------------
        st.subheader("✨ AI Generated Professional Summary")

        summary = (
            f"{name} is a motivated student with knowledge of "
            f"{skills}. The candidate has academic experience "
            f"in {projects} and is interested in developing "
            f"practical solutions using Artificial Intelligence "
            f"and Machine Learning technologies."
        )

        st.info(summary)

        # -----------------------------
        # Career Objective
        # -----------------------------
        st.subheader("🎯 Career Objective")

        objective = (
            "To obtain an opportunity where I can apply my "
            "technical knowledge, problem-solving skills and "
            "Artificial Intelligence expertise to develop "
            "innovative solutions while continuously improving "
            "my professional skills."
        )

        st.write(objective)

        # -----------------------------
        # Resume Preview
        # -----------------------------
        st.divider()

        st.header("📄 Resume Preview")

        st.markdown(f"# {name}")

        st.write(
            f"📧 {email}  |  📱 {phone}  |  📍 {location}"
        )

        st.subheader("Professional Summary")
        st.write(summary)

        st.subheader("Career Objective")
        st.write(objective)

        st.subheader("Education")
        st.write(education)

        st.subheader("Skills")
        st.write(skills)

        st.subheader("Projects")
        st.write(projects)

        # -----------------------------
        # Create PDF
        # -----------------------------
        filename = "AI_Resume.pdf"

        pdf = canvas.Canvas(
            filename,
            pagesize=A4
        )

        width, height = A4

        y = height - 50

        pdf.setFont(
            "Helvetica-Bold",
            20
        )

        pdf.drawString(
            50,
            y,
            name
        )

        y -= 25

        pdf.setFont(
            "Helvetica",
            10
        )

        pdf.drawString(
            50,
            y,
            f"{email} | {phone} | {location}"
        )

        y -= 40

        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            50,
            y,
            "Professional Summary"
        )

        y -= 20

        pdf.setFont(
            "Helvetica",
            10
        )

        for line in summary.split("."):

            if line.strip():

                pdf.drawString(
                    50,
                    y,
                    line.strip()[:100]
                )

                y -= 15

        y -= 20

        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            50,
            y,
            "Career Objective"
        )

        y -= 20

        pdf.setFont(
            "Helvetica",
            10
        )

        for line in objective.split("."):

            if line.strip():

                pdf.drawString(
                    50,
                    y,
                    line.strip()[:100]
                )

                y -= 15

        y -= 20

        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            50,
            y,
            "Education"
        )

        y -= 20

        pdf.setFont(
            "Helvetica",
            10
        )

        pdf.drawString(
            50,
            y,
            education[:100]
        )

        y -= 40

        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            50,
            y,
            "Skills"
        )

        y -= 20

        pdf.setFont(
            "Helvetica",
            10
        )

        pdf.drawString(
            50,
            y,
            skills[:100]
        )

        y -= 40

        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            50,
            y,
            "Projects"
        )

        y -= 20

        pdf.setFont(
            "Helvetica",
            10
        )

        pdf.drawString(
            50,
            y,
            projects[:100]
        )

        pdf.save()

        # -----------------------------
        # Download Button
        # -----------------------------
        with open(
            filename,
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Download Resume PDF",
                data=file,
                file_name="AI_Resume.pdf",
                mime="application/pdf"
            )