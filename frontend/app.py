import streamlit as st
import requests
from PIL import Image
import io
import os

# ============================================================
# Config
# ============================================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🔬",
    layout="wide"
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    .main-title {
        font-size    : 2.5rem;
        font-weight  : 800;
        text-align   : center;
        color        : #2c3e50;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align   : center;
        color        : #7f8c8d;
        font-size    : 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background   : #f8f9fa;
        border-radius: 12px;
        padding      : 1.5rem;
        border-left  : 5px solid #3498db;
        margin-bottom: 1rem;
    }
    .disease-name {
        font-size  : 1.8rem;
        font-weight: 700;
        color      : #2c3e50;
    }
    .confidence-text {
        font-size  : 1.1rem;
        color      : #27ae60;
        font-weight: 600;
    }
    .advice-box {
        background   : #ffffff;
        border-radius: 12px;
        padding      : 1.5rem;
        border       : 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .advice-title {
        font-size    : 1.1rem;
        font-weight  : 700;
        color        : #2c3e50;
        margin-bottom: 0.5rem;
    }
    .disclaimer {
        background   : #fff3cd;
        border-radius: 8px;
        padding      : 1rem;
        border-left  : 4px solid #ffc107;
        font-size    : 0.85rem;
        color        : #856404;
    }
    .top3-box {
        background   : #f8f9fa;
        border-radius: 8px;
        padding      : 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## Skin Disease Detection")
    st.markdown("AI powered skin disease analysis system")
    st.divider()

    # Backend health check
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health.status_code == 200:
            data = health.json()
            st.success("System is Online")
            st.markdown(
                f"**Model:** {'Working' if data['model_loaded'] else 'Not Working'}")
            st.markdown(
                f"**LLM:** {'Working' if data['llm_loaded'] else 'Not Working'}")
            st.markdown(f"**Device:** {data['device']}")
        else:
            st.error("Backend Error")
    except:
        st.error("Backend Offline")
        st.info("Start backend first:\nuvicorn backend.main:app --reload")

    st.divider()

    st.markdown("###Supported Skin Conditions")
    diseases = [
        "Eczema",
        "Warts & Viral Infections",
        "Melanoma",
        "Atopic Dermatitis",
        "Basal Cell Carcinoma",
        "Melanocytic Nevi",
        "Benign Keratosis",
        "Psoriasis & Lichen Planus",
        "Seborrheic Keratoses",
        "Tinea & Fungal Infections"
    ]
    for disease in diseases:
        st.markdown(f"• {disease}")

    st.divider()

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer</strong><br>
    This tool is for educational purposes only.
    Always consult a qualified dermatologist
    for proper medical advice.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Main Page
# ============================================================

st.markdown('<h1 class="main-title">Skin Disease Detection System</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a skin image to get AI powered disease detection and medical recommendations</p>', unsafe_allow_html=True)

st.divider()

# ============================================================
# Upload Section
# ============================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a skin image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a clear image of the skin condition"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown(f"""
        **File Details:**
        - Name : `{uploaded_file.name}`
        - Size : `{uploaded_file.size / 1024:.1f} KB`
        - Type : `{uploaded_file.type}`
        """)

with col2:
    st.markdown("###How to Use")
    st.markdown("""
    1. Upload a clear skin image
    2. Click **Analyze** button
    3. View disease prediction
    4. Read AI recommendations
    
    **For best results:**
    - Use good lighting
    - Keep lesion in focus
    - Include surrounding skin
    - Avoid shadows
    """)

    if uploaded_file:
        st.markdown("---")
        analyze_btn = st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        )

        if analyze_btn:
            with st.spinner("🔄 Analyzing image..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name,
                                      uploaded_file, uploaded_file.type)}
                    response = requests.post(
                        f"{BACKEND_URL}/analyze_skin",
                        files=files,
                        timeout=60
                    )

                    if response.status_code == 200:
                        st.session_state['result'] = response.json()
                        st.session_state['image'] = image
                        st.session_state['has_result'] = True
                    else:
                        st.error(f"API Error: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to backend. Make sure it is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ============================================================
# Results Section
# ============================================================

if st.session_state.get('has_result'):
    result = st.session_state['result']

    st.divider()
    st.markdown("##Analysis Results")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown("### Detection Result")
        st.markdown(f"**Disease:** {result['disease']}")
        st.markdown(f"**Confidence:** {result['confidence'] * 100:.2f}%")
        st.progress(result['confidence'])

        st.markdown("#### Top 3 Predictions")
        for i, pred in enumerate(result['top3']):
            rank = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"{rank} **{pred['disease']}**")
            st.progress(pred['confidence'])
            st.caption(f"{pred['confidence'] * 100:.2f}%")

    with res_col2:
        st.markdown("### 🤖 AI Recommendations")

        st.markdown("**📌 Recommendations**")
        st.info(result['recommendations'])

        st.markdown("**👣 Next Steps**")
        st.info(result['next_steps'])

        st.markdown("**💡 Daily Care Tips**")
        st.info(result['tips'])

    st.divider()
    st.warning("""
    ⚠️ Medical Disclaimer: This AI analysis is for educational purposes only.
    Always consult a qualified dermatologist for proper medical advice.
    """)
