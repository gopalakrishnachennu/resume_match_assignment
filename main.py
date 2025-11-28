import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import time
import base64
from io import BytesIO
import json
from PyPDF2 import PdfReader

# Minimal fallback stopwords used if NLTK data is unavailable
FALLBACK_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was',
    'were', 'will', 'with', 'this', 'these', 'those', 'you', 'your'
}

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download all required NLTK data"""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass

# Download NLTK data on startup
download_nltk_data()

# Page Configuration
st.set_page_config(
    page_title="ResumeMatcher.AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Advanced Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Space Grotesk', sans-serif;
        color: #0f172a;
    }
    
    .main {
        background: radial-gradient(circle at 20% 20%, rgba(255, 209, 220, 0.45), transparent 35%),
                    radial-gradient(circle at 80% 0%, rgba(155, 196, 226, 0.5), transparent 30%),
                    linear-gradient(135deg, #f7f2ec 0%, #eef3ff 60%, #e7f6ff 100%);
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(10deg); }
    }
    
    .stButton>button {
        background: rgba(255, 255, 255, 0.25);
        color: #0f172a;
        border: 1px solid rgba(30,64,175,0.15);
        padding: 15px 30px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.25s ease;
        box-shadow:
            8px 8px 18px rgba(15, 23, 42, 0.12),
            -6px -6px 16px rgba(255, 255, 255, 0.65),
            inset 0 0 0 rgba(255,255,255,0);
        backdrop-filter: blur(8px);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(14,165,233,0.18));
        box-shadow:
            10px 10px 22px rgba(15, 23, 42, 0.14),
            -8px -8px 18px rgba(255, 255, 255, 0.7),
            inset 0 0 0 rgba(255,255,255,0.5);
    }
    
    .header-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(244,244,255,0.9));
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        border: 1px solid rgba(15,23,42,0.08);
        animation: fadeInDown 1s ease;
        box-shadow: 0 10px 35px rgba(15,23,42,0.08);
        position: relative;
    }

    .glass-chip {
        padding: 6px 12px;
        border-radius: 12px;
        background: rgba(255,255,255,0.55);
        border: 1px solid rgba(255,255,255,0.6);
        color: #0f172a;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.4px;
        backdrop-filter: blur(10px);
        box-shadow:
            4px 4px 12px rgba(15,23,42,0.12),
            -4px -4px 12px rgba(255,255,255,0.65);
        animation: chipGlow 6s ease-in-out infinite;
    }

    @keyframes chipGlow {
        0%, 100% { box-shadow: 4px 4px 12px rgba(15,23,42,0.12), -4px -4px 12px rgba(255,255,255,0.65); }
        50% { box-shadow: 6px 6px 16px rgba(59,130,246,0.18), -6px -6px 14px rgba(255,255,255,0.75); }
    }

    .glass-chip.attention {
        animation: chipPop 2.8s ease-in-out infinite, chipGlow 6s ease-in-out infinite;
        transform-origin: center;
    }

    @keyframes chipPop {
        0%, 100% { transform: translateY(0) scale(1); }
        30% { transform: translateY(-2px) scale(1.04); }
        60% { transform: translateY(1px) scale(0.98); }
    }

    .floating-badge {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 10px 24px;
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.9);
        color: #0f172a;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.5px;
        backdrop-filter: blur(20px);
        box-shadow:
            0 8px 32px rgba(31, 38, 135, 0.1),
            inset 0 0 0 1px rgba(255, 255, 255, 0.6);
        animation: floaty 4s ease-in-out infinite;
        transition: all 0.3s ease;
        z-index: 100;
    }

    .floating-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.15);
        background: rgba(255, 255, 255, 0.9);
    }

    .floating-badge .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #3b82f6;
        box-shadow: 0 0 0 rgba(59, 130, 246, 0.4);
        animation: pulse-blue 2s infinite;
    }

    @keyframes pulse-blue {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(59, 130, 246, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
        }
    }

    @keyframes floaty {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }

    @keyframes shimmer {
        0% { box-shadow: 10px 10px 26px rgba(15,23,42,0.14), -8px -8px 18px rgba(255,255,255,0.78), inset 0 0 0 rgba(255,255,255,0.2); }
        50% { box-shadow: 12px 12px 28px rgba(59,130,246,0.16), -10px -10px 20px rgba(255,255,255,0.8), inset 0 0 10px rgba(255,255,255,0.25); }
        100% { box-shadow: 10px 10px 26px rgba(15,23,42,0.14), -8px -8px 18px rgba(255,255,255,0.78), inset 0 0 0 rgba(255,255,255,0.2); }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(239,246,255,0.94));
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 22px;
        border: 1px solid rgba(15,23,42,0.07);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        animation: fadeInUp 0.8s ease;
        box-shadow: 0 12px 30px rgba(15,23,42,0.12);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 40px rgba(30,41,59,0.18);
    }
    
    .score-circle {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        background: conic-gradient(#0ea5e9 0deg, #1e3a8a 360deg);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 20px auto;
        animation: rotate 3s linear infinite;
        position: relative;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }
    
    .keyword-tag {
        display: inline-block;
        background: linear-gradient(120deg, #0ea5e9 0%, #6366f1 100%);
        color: #f8fafc;
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        font-size: 13px;
        animation: slideIn 0.5s ease;
        box-shadow: 0 6px 18px rgba(14,165,233,0.25);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .progress-bar-custom {
        height: 30px;
        border-radius: 15px;
        background: rgba(15,23,42,0.08);
        overflow: hidden;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #1e40af 0%, #0ea5e9 100%);
        border-radius: 15px;
        animation: progressAnimation 2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #f8fafc;
        font-weight: 600;
    }
    
    @keyframes progressAnimation {
        from { width: 0%; }
    }
    
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid rgba(15,23,42,0.08);
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(8px);
        color: #0f172a;
    }
    
    h1, h2, h3 {
        color: #0f172a;
        text-shadow: none;
    }
    
    .stAlert {
        border-radius: 14px;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(240,247,255,0.9));
        backdrop-filter: blur(12px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = {}

class ResumeMatcherAI:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = self.load_stopwords()
        self.warned_tokenizer = False
        self.warned_stopwords = False
        self.warned_lemmatizer = False

    def load_stopwords(self):
        """Load stopwords with graceful fallback"""
        try:
            return set(stopwords.words('english'))
        except LookupError:
            download_nltk_data()
            try:
                return set(stopwords.words('english'))
            except LookupError:
                if not self.warned_stopwords:
                    st.warning("Using fallback stopwords; NLTK stopwords unavailable.")
                    self.warned_stopwords = True
                return FALLBACK_STOPWORDS

    def tokenize_text(self, text):
        """Tokenize text with fallbacks if NLTK punkt data is missing"""
        try:
            return word_tokenize(text)
        except LookupError:
            download_nltk_data()
            try:
                return word_tokenize(text)
            except LookupError:
                if not self.warned_tokenizer:
                    st.warning("Using simple regex tokenizer; NLTK punkt data unavailable.")
                    self.warned_tokenizer = True
                return re.findall(r"[a-zA-Z]+", text)
        
    def extract_text_from_file(self, file):
        """Extract text from uploaded file"""
        try:
            if file.type == "text/plain":
                return file.getvalue().decode("utf-8")
            elif file.type == "application/pdf":
                # Extract PDF text with PyPDF2; fallback to bytes decode if needed
                try:
                    reader = PdfReader(BytesIO(file.getvalue()))
                    text_pages = []
                    for page in reader.pages:
                        text_pages.append(page.extract_text() or "")
                    return "\n".join(text_pages)
                except Exception:
                    return file.getvalue().decode("utf-8", errors='ignore')
            else:
                return file.getvalue().decode("utf-8", errors='ignore')
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = self.tokenize_text(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Lemmatization
        cleaned_tokens = []
        for word in tokens:
            try:
                cleaned_tokens.append(self.lemmatizer.lemmatize(word))
            except LookupError:
                download_nltk_data()
                try:
                    cleaned_tokens.append(self.lemmatizer.lemmatize(word))
                except LookupError:
                    if not self.warned_lemmatizer:
                        st.warning("Using stemmer fallback; NLTK WordNet data unavailable.")
                        self.warned_lemmatizer = True
                    cleaned_tokens.append(self.stemmer.stem(word))
        
        return ' '.join(cleaned_tokens), cleaned_tokens
    
    def extract_keywords(self, tokens, top_n=20):
        """Extract top keywords"""
        word_freq = Counter(tokens)
        return word_freq.most_common(top_n)
    
    def calculate_similarity(self, resume_text, job_text):
        """Calculate TF-IDF based cosine similarity"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100, vectorizer
    
    def find_missing_keywords(self, resume_tokens, job_tokens):
        """Find keywords present in job description but missing in resume"""
        resume_set = set(resume_tokens)
        job_set = set(job_tokens)
        missing = job_set - resume_set
        
        # Get frequency of missing keywords in job description
        job_freq = Counter(job_tokens)
        missing_with_freq = [(word, job_freq[word]) for word in missing]
        missing_with_freq.sort(key=lambda x: x[1], reverse=True)
        
        return missing_with_freq[:15]
    
    def generate_suggestions(self, missing_keywords, match_score):
        """Generate improvement suggestions"""
        suggestions = []
        
        if match_score < 50:
            suggestions.append("🔴 Critical: Your resume needs major improvements to match this job description.")
        elif match_score < 70:
            suggestions.append("🟡 Moderate: Your resume partially matches but needs enhancement.")
        else:
            suggestions.append("🟢 Good: Your resume is well-aligned with the job description!")
        
        if missing_keywords:
            suggestions.append(f"\n📌 Add these important keywords: {', '.join([kw[0] for kw in missing_keywords[:5]])}")
        
        suggestions.append("\n💡 Tailor your resume to include industry-specific terminology from the job posting.")
        suggestions.append("✨ Quantify your achievements with metrics and numbers.")
        suggestions.append("🎯 Use action verbs to describe your responsibilities.")
        
        return suggestions
    
    def analyze(self, resume_text, job_text):
        """Complete analysis pipeline"""
        # Preprocess texts
        resume_clean, resume_tokens = self.preprocess_text(resume_text)
        job_clean, job_tokens = self.preprocess_text(job_text)
        
        # Extract keywords
        resume_keywords = self.extract_keywords(resume_tokens)
        job_keywords = self.extract_keywords(job_tokens)
        
        # Calculate similarity
        match_score, vectorizer = self.calculate_similarity(resume_clean, job_clean)
        
        # Find missing keywords
        missing_keywords = self.find_missing_keywords(resume_tokens, job_tokens)
        
        # Generate suggestions
        suggestions = self.generate_suggestions(missing_keywords, match_score)
        
        return {
            'match_score': match_score,
            'resume_keywords': resume_keywords,
            'job_keywords': job_keywords,
            'missing_keywords': missing_keywords,
            'suggestions': suggestions,
            'resume_tokens': len(resume_tokens),
            'job_tokens': len(job_tokens)
        }

def create_gauge_chart(score):
    """Create animated gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Match Score", 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00d2ff"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255,0,0,0.3)'},
                {'range': [50, 70], 'color': 'rgba(255,255,0,0.3)'},
                {'range': [70, 100], 'color': 'rgba(0,255,0,0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Poppins"},
        height=300
    )
    
    return fig

def create_keyword_comparison(resume_kw, job_kw):
    """Create keyword frequency comparison chart"""
    resume_words = [kw[0] for kw in resume_kw[:10]]
    resume_freq = [kw[1] for kw in resume_kw[:10]]
    
    job_words = [kw[0] for kw in job_kw[:10]]
    job_freq = [kw[1] for kw in job_kw[:10]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=resume_words,
        x=resume_freq,
        name='Resume Keywords',
        orientation='h',
        marker=dict(
            color='rgba(0, 210, 255, 0.7)',
            line=dict(color='rgba(0, 210, 255, 1)', width=2)
        )
    ))
    
    fig.add_trace(go.Bar(
        y=job_words,
        x=job_freq,
        name='Job Keywords',
        orientation='h',
        marker=dict(
            color='rgba(245, 87, 108, 0.7)',
            line=dict(color='rgba(245, 87, 108, 1)', width=2)
        )
    ))
    
    fig.update_layout(
        title="Top Keywords Comparison",
        barmode='group',
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.05)",
        font={'color': "white", 'family': "Poppins"},
        height=400,
        xaxis_title="Frequency",
        yaxis_title="Keywords",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_missing_keywords_chart(missing_kw):
    """Create missing keywords visualization"""
    if not missing_kw:
        return None
    
    words = [kw[0] for kw in missing_kw[:10]]
    freq = [kw[1] for kw in missing_kw[:10]]
    
    fig = go.Figure(go.Bar(
        x=words,
        y=freq,
        marker=dict(
            color=freq,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Frequency")
        ),
        text=freq,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Missing Keywords (High Priority)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.05)",
        font={'color': "white", 'family': "Poppins"},
        height=400,
        xaxis_title="Keywords",
        yaxis_title="Frequency in Job Description"
    )
    
    return fig

def create_word_cloud_data(keywords):
    """Create word cloud data"""
    return {word: freq for word, freq in keywords}

def main():
    # Top Badge
    st.markdown("""
        <div style="display:flex; justify-content:center; margin-bottom: 20px;">
            <div class="floating-badge">
                <div class="dot"></div>
                <span>Group - 2</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="header-container">
            <div style="display:flex; align-items:center; justify-content:center; gap:14px; margin-bottom:6px;">
                <div style="width:52px; height:52px; border-radius:14px; background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(14,165,233,0.25)); display:flex; align-items:center; justify-content:center; box-shadow: inset 2px 2px 8px rgba(15,23,42,0.12), inset -4px -4px 10px rgba(255,255,255,0.75);">
                    <span style="font-size:26px;">📄</span>
                </div>
                <div>
                    <h1 style='text-align: left; font-size: 44px; margin: 0; letter-spacing: -0.5px;'>
                        ResumeMatcher.AI
                    </h1>
                    <div style="display:flex; gap:8px; align-items:center; margin-top:6px;">
                        <span class="glass-chip" style="background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(14,165,233,0.16));">ATS-FREE</span>
                        <span class="glass-chip" style="background: linear-gradient(135deg, rgba(14,165,233,0.14), rgba(99,102,241,0.14));">AI RESUME ANALYZER</span>
                    </div>
                </div>
            </div>
          
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Standard Analysis", "Deep Analysis", "Quick Scan"]
        )
        
        st.markdown("### 📊 Features")
        show_visualizations = st.checkbox("Show Visualizations", value=True)
        show_suggestions = st.checkbox("Show Suggestions", value=True)
        show_keywords = st.checkbox("Show Keywords Analysis", value=True)
        
        st.markdown("### ℹ️ About")
        st.info("""
        **ResumeMatcher.AI** uses advanced NLP algorithms including:
        - TF-IDF Vectorization
        - Cosine Similarity
        - Keyword Extraction
        - Gap Analysis
        
        Built with ❤️ using Devanth
        """)
        
        if st.button("🔄 Reset Analysis"):
            st.session_state.analysis_done = False
            st.session_state.results = {}
            st.rerun()
    
    # Main Content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Resume Input")
        resume_input_type = st.radio(
            "Choose input method:",
            ["Upload File", "Paste Text"],
            horizontal=True,
            key="resume_type"
        )
        
        resume_text = ""
        if resume_input_type == "Upload File":
            resume_file = st.file_uploader(
                "Upload Resume",
                type=['txt', 'pdf', 'doc', 'docx'],
                key="resume_file"
            )
            if resume_file:
                matcher = ResumeMatcherAI()
                resume_text = matcher.extract_text_from_file(resume_file)
                st.success("✅ Resume uploaded successfully!")
        else:
            resume_text = st.text_area(
                "Paste your resume here:",
                height=300,
                placeholder="Enter your resume text...",
                key="resume_text"
            )
    
    with col2:
        st.markdown("### 💼 Job Description Input")
        job_input_type = st.radio(
            "Choose input method:",
            ["Upload File", "Paste Text"],
            horizontal=True,
            key="job_type"
        )
        
        job_text = ""
        if job_input_type == "Upload File":
            job_file = st.file_uploader(
                "Upload Job Description",
                type=['txt', 'pdf', 'doc', 'docx'],
                key="job_file"
            )
            if job_file:
                matcher = ResumeMatcherAI()
                job_text = matcher.extract_text_from_file(job_file)
                st.success("✅ Job description uploaded successfully!")
        else:
            job_text = st.text_area(
                "Paste job description here:",
                height=300,
                placeholder="Enter job description text...",
                key="job_text"
            )
    
    # Analyze Button
    st.markdown("<br>", unsafe_allow_html=True)
    col_center = st.columns([2, 1, 2])
    with col_center[1]:
        analyze_btn = st.button("🚀 ANALYZE NOW", use_container_width=True)
    

    
    if analyze_btn:
        if not resume_text or not job_text:
            st.error("⚠️ Please provide both resume and job description!")
        else:
            # Progress bar animation
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            stages = [
                "📄 Extracting text...",
                "🧹 Preprocessing data...",
                "🔍 Extracting keywords...",
                "📊 Calculating similarity...",
                "🎯 Analyzing gaps...",
                "✨ Generating insights..."
            ]
            
            for i, stage in enumerate(stages):
                status_text.text(stage)
                progress_bar.progress((i + 1) / len(stages))
                time.sleep(0.3)
            
            # Perform analysis
            matcher = ResumeMatcherAI()
            results = matcher.analyze(resume_text, job_text)
            st.session_state.results = results
            st.session_state.analysis_done = True
            
            status_text.text("✅ Analysis Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
    
    # Display Results
    if st.session_state.analysis_done and st.session_state.results:
        results = st.session_state.results
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 📊 Analysis Results")
        
        # Match Score Section
        st.markdown("### 🎯 Match Score")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            score = results['match_score']
            if score >= 70:
                st.markdown(f"""
                    <div class="metric-card pulse">
                        <h2 style='color: #0ea5e9; text-align: center;'>{score:.1f}%</h2>
                        <p style='text-align: center;'>Excellent Match! 🎉</p>
                    </div>
                """, unsafe_allow_html=True)
            elif score >= 50:
                st.markdown(f"""
                    <div class="metric-card pulse">
                        <h2 style='color: #22c55e; text-align: center;'>{score:.1f}%</h2>
                        <p style='text-align: center;'>Good Match 👍</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="metric-card pulse">
                        <h2 style='color: #ef4444; text-align: center;'>{score:.1f}%</h2>
                        <p style='text-align: center;'>Needs Improvement 💪</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if show_visualizations:
                fig_gauge = create_gauge_chart(results['match_score'])
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h3 style='text-align: center;'>📈 Statistics</h3>
                    <p>Resume Words: <b>{results['resume_tokens']}</b></p>
                    <p>Job Words: <b>{results['job_tokens']}</b></p>
                    <p>Missing Keywords: <b>{len(results['missing_keywords'])}</b></p>
                </div>
            """, unsafe_allow_html=True)
        
        # Keywords Analysis
        if show_keywords:
            st.markdown("### 🔑 Keywords Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Resume Keywords")
                keywords_html = ""
                for word, freq in results['resume_keywords'][:15]:
                    keywords_html += f'<span class="keyword-tag">{word} ({freq})</span>'
                st.markdown(f'<div>{keywords_html}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Job Keywords")
                keywords_html = ""
                for word, freq in results['job_keywords'][:15]:
                    keywords_html += f'<span class="keyword-tag">{word} ({freq})</span>'
                st.markdown(f'<div>{keywords_html}</div>', unsafe_allow_html=True)
            
            if show_visualizations:
                fig_keywords = create_keyword_comparison(
                    results['resume_keywords'],
                    results['job_keywords']
                )
                st.plotly_chart(fig_keywords, use_container_width=True)
        
        # Missing Keywords
        st.markdown("### ⚠️ Missing Keywords")
        
        if results['missing_keywords']:
            keywords_html = ""
            for word, freq in results['missing_keywords']:
                keywords_html += f'<span class="keyword-tag" style="background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);">{word} ({freq})</span>'
            st.markdown(f'<div>{keywords_html}</div>', unsafe_allow_html=True)
            
            if show_visualizations:
                fig_missing = create_missing_keywords_chart(results['missing_keywords'])
                if fig_missing:
                    st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("🎉 Great! No critical keywords are missing!")
        
        # Suggestions
        if show_suggestions:
            st.markdown("### 💡 Improvement Suggestions")
            
            for suggestion in results['suggestions']:
                st.markdown(f"""
                    <div class="metric-card" style="margin: 10px 0;">
                        <p style="margin: 0;">{suggestion}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Download Report
        st.markdown("### 📥 Export Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download JSON Report"):
                json_data = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="resume_analysis.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Download CSV Report"):
                df = pd.DataFrame({
                    'Match Score': [results['match_score']],
                    'Resume Words': [results['resume_tokens']],
                    'Job Words': [results['job_tokens']],
                    'Missing Keywords': [len(results['missing_keywords'])]
                })
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="resume_analysis.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📝 Download Summary"):
                summary = f"""
RESUME MATCHER AI - ANALYSIS REPORT
====================================

Match Score: {results['match_score']:.2f}%
Resume Word Count: {results['resume_tokens']}
Job Description Word Count: {results['job_tokens']}
Missing Keywords Count: {len(results['missing_keywords'])}

TOP RESUME KEYWORDS:
{chr(10).join([f"- {word}: {freq}" for word, freq in results['resume_keywords'][:10]])}

MISSING KEYWORDS:
{chr(10).join([f"- {word}: {freq}" for word, freq in results['missing_keywords'][:10]])}

SUGGESTIONS:
{chr(10).join(results['suggestions'])}
                """
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="resume_analysis.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
