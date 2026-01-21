# ============================================================================
# RESEARCH PAPER ASSISTANT - FIXED & PRODUCTION READY (APPLE SILICON SAFE)
# ============================================================================

# =================== CRITICAL: MUST BE FIRST ===================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =================== STANDARD IMPORTS ===================
import streamlit as st
import pickle
import numpy as np
import warnings
import hashlib

warnings.filterwarnings("ignore")

# =================== STREAMLIT CONFIG ===================
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        margin: 12px 0;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .recommendation-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .recommendation-box a {
        text-decoration: none;
        color: #2d3748;
        transition: color 0.2s ease;
    }
    .recommendation-box a:hover {
        color: #667eea;
        text-decoration: underline;
    }
    .category-badge {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        font-weight: bold;
        color: #2d3748;
    }
    .stats-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SAFE MODEL LOADING (LAZY + CACHED)
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from tensorflow import keras
        import tensorflow as tf

        torch.set_num_threads(1)

        # ---------- Recommendation ----------
        embeddings = pickle.load(open("models/embeddings.pkl", "rb"))
        sentences = pickle.load(open("models/sentences.pkl", "rb"))

        rec_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # ---------- Classification ----------
        loaded_model = keras.models.load_model("models/model.h5", compile=False)

        # Load label vocabulary
        label_vocab = pickle.load(open("models/label_vocab.pkl", "rb"))
        
        # Load text vectorizer vocab
        with open("models/text_vectorizer_vocab.pkl", "rb") as f:
            vectorizer_vocab = pickle.load(f)
        
        # Recreate text vectorizer manually (avoid config issues)
        from tensorflow.keras.layers import TextVectorization
        text_vectorizer = TextVectorization(
            max_tokens=159077,
            output_mode="multi_hot"
        )
        # Adapt using the loaded vocabulary
        text_vectorizer.set_vocabulary(vectorizer_vocab)
        
        # Get model input size - check the model's first layer input shape
        # The Sequential model's first Dense layer expects a specific input size
        try:
            # Try to get from model config
            model_input_size = loaded_model.layers[0].input_shape[-1]
        except:
            # Fallback: use vocab size
            model_input_size = len(vectorizer_vocab)

        return (
            embeddings,
            sentences,
            rec_model,
            loaded_model,
            text_vectorizer,
            label_vocab,
            model_input_size,
        )

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()

with st.spinner("🔄 Loading AI models..."):
    (
        embeddings,
        sentences,
        rec_model,
        loaded_model,
        text_vectorizer,
        label_vocab,
        model_input_size,
    ) = load_models()

# ============================================================================
# HELPERS
# ============================================================================
def recommendation(text):
    if not text or len(text.strip()) < 3:
        return []

    import torch
    from sentence_transformers import util

    query = rec_model.encode(text, convert_to_tensor=True)
    emb = torch.tensor(embeddings)

    scores = util.cos_sim(emb, query).squeeze()
    k = min(5, len(scores))

    top = torch.topk(scores, k).indices.tolist()
    return [sentences[i] for i in top]


def invert_multi_hot(prob_vector, threshold=0.3):
    """Convert probability vector to category labels"""
    categories = [label_vocab[i] for i, p in enumerate(prob_vector) if p >= threshold]
    # Clean up category names - remove prefixes for better display
    cleaned_categories = []
    for cat in categories:
        # Remove common prefixes like 'cs.', 'stat.', 'eess.', 'math.', 'q-bio.', 'physics.', 'quant-', 'cond-mat.'
        cleaned = cat
        prefixes = ['cs.', 'stat.', 'eess.', 'math.', 'q-bio.', 'physics.', 'quant-', 'cond-mat.', 'q-fin.', 'econ.', 'astro-ph.', 'hep-', 'nlin.']
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        cleaned_categories.append(cleaned)
    return cleaned_categories


def pad_vector(vector, target_size):
    """Pads or truncates the vector to match model input size."""
    import tensorflow as tf
    
    current_size = vector.shape[1]
    
    if current_size < target_size:
        padding = tf.zeros((vector.shape[0], target_size - current_size), dtype=tf.float32)
        vector = tf.concat([vector, padding], axis=1)
    elif current_size > target_size:
        vector = vector[:, :target_size]
    
    return vector


def predict_category(abstract, threshold=0.3):
    """Predict subject categories for a given abstract"""
    if not abstract or len(abstract.strip()) < 20:
        return []

    import tensorflow as tf

    try:
        # Step 1: Vectorize abstract using the loaded text vectorizer
        # This produces a multi-hot vector of shape (1, vocab_size)
        vectorized = text_vectorizer([abstract])
        
        # Step 2: Ensure it's float32
        vectorized = tf.cast(vectorized, tf.float32)
        
        # Step 3: Pad or truncate to match model input size if needed
        current_size = vectorized.shape[1]
        if current_size != model_input_size:
            vectorized = pad_vector(vectorized, model_input_size)
        
        # Step 4: Predict probabilities using the loaded MLP
        probs = loaded_model.predict(vectorized, verbose=0)[0]
        
        # Step 5: Convert probabilities to labels using threshold
        categories = invert_multi_hot(probs, threshold)
        
        return categories
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return []


def get_input_hash(title, abstract):
    return hashlib.md5(f"{title}|||{abstract}".encode()).hexdigest()

# ============================================================================
# UI HEADER
# ============================================================================
st.markdown("""
<div style="text-align:center; padding:2rem 0;">
    <h1 class="gradient-text">
        📚 Research Paper Assistant
    </h1>
    <p style="font-size:1.2rem; color:#4a5568;">
        AI-Powered Paper Discovery & Classification
    </p>
</div>

<style>
.gradient-text {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-clip: text;
    -webkit-background-clip: text;
    color: #667eea; /* fallback */
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# STATISTICS
# ============================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='stats-card'>
        <h2 style='color: #667eea; margin: 0;'>{len(sentences):,}</h2>
        <p style='color: #718096; margin: 0;'>Papers Indexed</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='stats-card'>
        <h2 style='color: #764ba2; margin: 0;'>{len(label_vocab)}</h2>
        <p style='color: #718096; margin: 0;'>Subject Categories</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='stats-card'>
        <h2 style='color: #48bb78; margin: 0;'>MLP</h2>
        <p style='color: #718096; margin: 0;'>Classification Model</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# INPUT SECTION
# ============================================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Paper Recommendation")
    
    examples = [
        "Select an example...",
        "Attention is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "Deep Residual Learning for Image Recognition",
        "Generative Adversarial Networks"
    ]
    
    selected = st.selectbox("Try an example:", examples, index=0, key="example_selector")
    
    # Use selected example as default value
    default_title = selected if selected != "Select an example..." else ""
    title = st.text_input("Or enter custom title:", value=default_title, key="title_input")

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    abstract = st.text_area(
        "Paste paper abstract (for category prediction):",
        value="",
        key="abstract_input",
        placeholder="Paste full abstract here for subject area classification...",
        height=200
    )

# ============================================================================
# ANALYZE BUTTON
# ============================================================================
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

# ============================================================================
# SESSION STATE INITIALIZATION (REQUIRED)
# ============================================================================
if "last_input_hash" not in st.session_state:
    st.session_state.last_input_hash = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

# ============================================================================
# PROCESSING LOGIC
# ============================================================================
if analyze_button:
    # Use the actual selected value or the manual input
    actual_title = title.strip() if title.strip() else (selected if selected != "Select an example..." else "")
    actual_abstract = abstract.strip()
    
    current_hash = get_input_hash(actual_title, actual_abstract)
    inputs_changed = current_hash != st.session_state.last_input_hash
    st.session_state.last_input_hash = current_hash

    if inputs_changed:
        st.session_state.recommendations = None
        st.session_state.predictions = None

    if actual_title or actual_abstract:
        st.markdown("---")
        
        # RECOMMENDATIONS
        recommendation_input = actual_title if actual_title else actual_abstract
        
        if recommendation_input:
            st.subheader("📚 Recommended Similar Papers")
            
            if actual_title:
                st.caption("🔍 Based on title")
            else:
                st.caption("🔍 Based on abstract content")
            
            if st.session_state.recommendations is None or inputs_changed:
                with st.spinner("🔍 Finding similar papers..."):
                    st.session_state.recommendations = recommendation(recommendation_input)
            
            papers = st.session_state.recommendations
            
            if papers and len(papers) > 0:
                for i, paper in enumerate(papers, 1):
                    # Create arXiv search URL
                    import urllib.parse
                    search_query = urllib.parse.quote(paper)
                    arxiv_url = f"https://arxiv.org/search/?query={search_query}&searchtype=title"
                    
                    st.markdown(f"""
                    <div class='recommendation-box'>
                        <strong style='color: #667eea; font-size: 1.2rem;'>#{i}</strong>
                        <p style='margin: 5px 0 0 0; color: #2d3748;'>
                            <a href="{arxiv_url}" target="_blank" style='text-decoration: none; color: #2d3748;'>
                                {paper}
                            </a>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("💡 No recommendations found. Try different input.")
        
        # CATEGORY PREDICTION
        if actual_abstract:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🏷️ Predicted Subject Areas")
            
            if st.session_state.predictions is None or inputs_changed:
                with st.spinner("🏷️ Predicting categories..."):
                    st.session_state.predictions = predict_category(actual_abstract, threshold=0.3)
            
            categories = st.session_state.predictions
            
            if categories and len(categories) > 0:
                # Display categories with proper spacing
                html = " ".join([f'<span class="category-badge">{cat}</span>' for cat in categories])
                st.markdown(html, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"""
                💡 **Model Info**: Multi-Layer Perceptron trained on {len(sentences):,} ArXiv papers  
                📊 **Categories Found**: {len(categories)}  
                🎯 **Model Input Size**: {model_input_size:,} features  
                🔧 **Threshold**: 0.3 (30% confidence minimum)
                """)
            else:
                st.warning("""
                ⚠️ No categories predicted. Possible reasons:
                - Abstract too short or unclear
                - No categories exceeded the 30% confidence threshold
                - Try a more detailed abstract or lower the threshold
                """)
        elif actual_title and not actual_abstract:
            st.info("💡 **Tip**: Paste an abstract to get subject area predictions!")
    else:
        st.warning("⚠️ Please enter at least a paper title or abstract")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(f"""
    **Capabilities:**
    - 🔍 Semantic paper search
    - 🏷️ Subject classification
    - 📊 {len(sentences):,} papers indexed
    - 🎯 {len(label_vocab)} categories
    """)
    
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    st.markdown("""
    **For Recommendations:**
    1. Enter title OR paste abstract
    2. Click "Analyze"
    
    **For Classification:**
    1. Paste full abstract
    2. Click "Analyze"
    3. View predicted categories
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Technology")
    st.markdown(f"""
    - **Backend**: TensorFlow, PyTorch
    - **NLP**: Sentence Transformers
    - **Model**: MLP (512→256→{len(label_vocab)})
    - **Features**: {model_input_size:,} multi-hot encoded
    - **Vectorizer**: TextVectorization layer
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #718096; padding: 20px;'>
    <p><strong>Research Paper Assistant v2.0</strong></p>
    <p style='font-size: 0.9rem;'>
        Papers: {len(sentences):,} | 
        Categories: {len(label_vocab)} | 
        Model Input: {model_input_size:,} features
    </p>
    <p style='font-size: 0.85rem; margin-top: 10px;'>
        TensorFlow • PyTorch • Sentence Transformers • Streamlit
    </p>
</div>
""", unsafe_allow_html=True)