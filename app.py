import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string

# --- Page Config ---
st.set_page_config(page_title="AI Job Matcher", layout="wide")

# --- Premium CSS Styling ---
st.markdown("""
<style>
/* === 🌟 GLOBAL SETUP === */
.stApp {
    background: linear-gradient(135deg,#ffe1c2 0%, #ff6a00 100%);
    font-family: 'Segoe UI', sans-serif;
    color: #000000;
    padding: 10px;
}

/* === 🏷️ HEADER TITLE === */
.title {
    font-size: 2.7em;
    font-weight: 800;
    color: #2e0a0a;
    text-align: center;
    margin-bottom: 0.2em;
    text-shadow: 1px 1px 1px #ffdab9;
}

/* === 📛 SUBTITLE === */
.subtitle {
    text-align: center;
    font-size: 1.1em;
    font-weight: 500;
    margin-top: -15px;
    margin-bottom: 30px;
    color: #3a1a00;
}

/* === 📤 RESUME / INPUT COMPONENTS === */
.stFileUploader, .stTextInput, .stTextArea, .stSelectbox, .stSlider {
    background-color: #fffaf5;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 0 5px rgba(255, 106, 0, 0.2);
}

/* === 📊 DATAFRAME === */
.stDataFrame {
    background-color: #ffffffcc;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

/* === 📥 DOWNLOAD BUTTON === */
.stDownloadButton > button {
    background-color: #ff7e5f;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
}
.stDownloadButton > button:hover {
    background-color: #ff571f;
    transition: 0.3s ease-in-out;
}

/* === 🧭 SIDEBAR STYLE === */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #ff8a00 0%, #bf360c 100%);
    color: white;
    padding: 1rem;
}

/* === SIDEBAR LABELS === */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: white !important;
    font-weight: 600;
}

/* === SIDEBAR ELEMENTS BG === */
.stSlider, .stMultiSelect, .stFileUploader {
    background-color: rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 10px;
}

/* === EXPANDER STYLE === */
.st-expander > summary {
    background-color: #ffe5d0;
    font-weight: bold;
    border-radius: 6px;
}

/* === MARKDOWN TEXT STYLE === */
h1, h2, h3, h4 {
    color: #3a1a00;
}

/* === TOOLTIP AND HOVER === */
button:hover {
    cursor: pointer;
}
            
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# --- Page Setup ---
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string

st.set_page_config(page_title="AI Job Matcher", layout="wide")

# --- Custom Header ---
st.markdown("<div class='title'>🤞 AI-Powered Resume Job Matcher ✌️</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>🧘‍♂️ Created with ❤️ by Bishwajit 🙏 Jai Shree Ram</div>", unsafe_allow_html=True)
st.markdown("### 📂 Upload your resume and let AI find jobs that match **you** best!")

# --- Load Data ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Sidebar: Upload Dataset
st.sidebar.header("📁 Upload Job Dataset")
uploaded_csv = st.sidebar.file_uploader("Upload `matched_jobs.csv`", type=["csv"])

if uploaded_csv is not None:
    df = load_data(uploaded_csv)
    st.sidebar.success("✅ CSV uploaded successfully.")
else:
    df = load_data("matched_jobs.csv")

# Total Jobs (Right aligned using columns)
col1, col2 = st.columns([1, 1])
with col2:
    st.markdown(f"<div style='text-align:right;font-size:18px;'>💼 <strong>Total Jobs Available:</strong> `{len(df)}`</div>", unsafe_allow_html=True)

# --- Sidebar Filters ---
st.sidebar.header("🎯 Filter Jobs")

min_match_percent = st.sidebar.slider("Minimum Resume Match %", 0, 100, 0)
if min_match_percent > 0:
    st.sidebar.markdown("🧩 _Tip: Lower the match % to discover more jobs_")

remote_options = df["is_remote"].dropna().unique().tolist()
selected_remote = st.sidebar.multiselect("🌍 Remote/Onsite", remote_options, default=remote_options)

if "job_types" in df.columns:
    job_types = df["job_types"].dropna().str.split(", ").explode().unique()
    selected_job_types = st.sidebar.multiselect("🧾 Job Type", job_types, default=job_types)
else:
    selected_job_types = []

# --- Resume Upload Section ---
st.markdown("### 📄 Upload Your Resume")
resume_file = st.file_uploader("Upload your PDF resume here", type=["pdf"])

resume_text = ""
resume_keywords = []

if resume_file:
    st.success("🎉 Resume uploaded successfully!")

    with fitz.open(stream=resume_file.read(), filetype="pdf") as doc:
        for page in doc:
            resume_text += page.get_text()

    with st.expander("📝 Preview Resume Text"):
        st.write(resume_text[:1500])

    nltk.download("stopwords")

    def extract_keywords_from_resume(text, top_n=30):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tfidf = TfidfVectorizer(stop_words="english", max_features=top_n)
        tfidf_matrix = tfidf.fit_transform([text])
        return list(tfidf.get_feature_names_out())

    resume_keywords = extract_keywords_from_resume(resume_text, top_n=30)
    # st.markdown("### 🔑 Extracted Resume Keywords:")
    # st.write(resume_keywords)

# --- Filtering Logic ---
filtered_df = df[
    (df["resume_match_percent"] >= min_match_percent) &
    (df["is_remote"].isin(selected_remote))
]

if selected_job_types:
    filtered_df = filtered_df[
        filtered_df["job_types"].fillna("").apply(lambda x: any(jt in x for jt in selected_job_types))
    ]

# --- Show Results ---
st.markdown(f"### 🎯 `{len(filtered_df)}` Matching Jobs Found")
if len(filtered_df) > 0:
    st.dataframe(filtered_df, use_container_width=True)
else:
    st.warning("😕 No jobs found. Try reducing filters or re-upload your resume.")

# --- Download Button ---
st.download_button(
    label="📥 Download Filtered Jobs as CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_matched_jobs.csv",
    mime="text/csv"
)
