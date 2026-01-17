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

import os, requests, zipfile

MODELS_ZIP_PATH = "models/models.zip"
MODELS_DIR = "models"
RELEASE_URL = "https://github.com/ashhad25/researchai-assistant/releases/download/v2.0/models.zip"

# Create folder if missing
os.makedirs(MODELS_DIR, exist_ok=True)

# Download ZIP if missing
if not os.path.exists(MODELS_ZIP_PATH):
    print("Downloading models.zip from GitHub Release...")
    r = requests.get(RELEASE_URL, stream=True)
    with open(MODELS_ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")

# Unzip files if they haven't been extracted yet
with zipfile.ZipFile(MODELS_ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(MODELS_DIR)
    print("Models extracted!")



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
    """Load recommendation and classification models"""
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        from sentence_transformers import SentenceTransformer

        # Load recommendation model components
        with open("models/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        with open("models/sentences.pkl", "rb") as f:
            sentences = pickle.load(f)

        rec_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

        # Load classification model components
        from tensorflow import keras
        from tensorflow.keras.layers import TextVectorization
        import tensorflow as tf
        
        loaded_model = keras.models.load_model("models/model.h5", compile=False)
        
        with open("models/vocab.pkl", "rb") as f:
            loaded_vocab = pickle.load(f)
        with open("models/idf_weights.pkl", "rb") as f:
            loaded_idf_weights = pickle.load(f)
        with open("models/text_vectorizer_config.pkl", "rb") as f:
            vectorizer_config = pickle.load(f)

        # Clean config for compatibility
        for key in ["batch_input_shape", "dtype", "name", "trainable", "ragged"]:
            vectorizer_config.pop(key, None)

        # Recreate text vectorizer
        text_vectorizer = TextVectorization.from_config(vectorizer_config)
        text_vectorizer.set_vocabulary(loaded_vocab, idf_weights=loaded_idf_weights)
        
        # Get model input size from the model's input
        # Try multiple methods to get input size
        try:
            model_input_size = loaded_model.input_shape[1]
        except:
            try:
                model_input_size = loaded_model.layers[0].input_shape[1]
            except:
                # Fallback: infer from vectorizer vocab size
                model_input_size = len(loaded_vocab)
        
        return (embeddings, sentences, rec_model, loaded_model, 
                text_vectorizer, loaded_vocab, model_input_size)

    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

with st.spinner("🔄 Loading AI models..."):
    (embeddings, sentences, rec_model, loaded_model, 
     text_vectorizer, loaded_vocab, model_input_size) = load_models()

# ============================================================================
# SESSION STATE
# ============================================================================
for k in ["last_input_hash", "recommendations", "predictions"]:
    st.session_state.setdefault(k, None)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def recommendation(text):
    """Find similar papers using semantic similarity - FIXED VERSION"""
    if not text or len(text.strip()) < 3:
        return []

    try:
        import torch
        from sentence_transformers import util

        # Encode the input text
        input_embedding = rec_model.encode(text)
        
        # Ensure embeddings is a numpy array or tensor
        if isinstance(embeddings, np.ndarray):
            embeddings_tensor = torch.tensor(embeddings)
        else:
            embeddings_tensor = embeddings
        
        # Calculate cosine similarity
        cosine_scores = util.cos_sim(embeddings_tensor, input_embedding)
        
        # FIXED: Flatten and ensure we don't ask for more than available
        cosine_scores_flat = cosine_scores.flatten()
        k = min(5, len(cosine_scores_flat))
        
        if k == 0:
            return []
        
        # Get top k similar papers
        top_results = torch.topk(cosine_scores_flat, k=k, sorted=True)
        
        # Extract paper titles
        papers_list = []
        for idx in top_results.indices:
            idx_val = idx.item()
            if idx_val < len(sentences):
                papers_list.append(sentences[idx_val])
        
        return papers_list

    except Exception as e:
        st.error(f"❌ Recommendation error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []

def invert_multi_hot(encoded_labels):
    """Convert multi-hot encoded predictions to category names"""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    categories = np.take(loaded_vocab, hot_indices)
    # Filter out special tokens
    categories = [cat for cat in categories if cat not in ['[UNK]', '', ' ']]
    return categories

def pad_vector(vector, target_size):
    """Pad or truncate vector to match model input size"""
    import tensorflow as tf
    
    current_size = vector.shape[1]
    if current_size < target_size:
        padding = tf.zeros((vector.shape[0], target_size - current_size), dtype=vector.dtype)
        vector = tf.concat([vector, padding], axis=1)
    elif current_size > target_size:
        vector = vector[:, :target_size]
    return vector

def predict_category(abstract):
    """Predict subject categories from abstract"""
    if not abstract or len(abstract.strip()) < 10:
        return []
    
    try:
        # Vectorize abstract
        vectorized = text_vectorizer([abstract])
        
        # Pad to model input size
        vectorized_padded = pad_vector(vectorized, model_input_size)
        
        # Get predictions
        predictions = loaded_model.predict(vectorized_padded, verbose=0)
        
        # Apply threshold
        binary_predictions = (predictions > 0.5).astype(int)[0]
        
        # Convert to category names
        predicted_categories = invert_multi_hot(binary_predictions)
        
        return list(set(predicted_categories)) if predicted_categories else []
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []

def get_input_hash(title, abstract):
    """Create hash for input change detection"""
    return hashlib.md5(f"{title}|||{abstract}".encode()).hexdigest()

# ============================================================================
# UI HEADER
# ============================================================================
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        📚 Research Paper Assistant
    </h1>
    <p style='font-size: 1.2rem; color: #4a5568;'>AI-Powered Paper Discovery & Classification</p>
</div>
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
        <h2 style='color: #764ba2; margin: 0;'>{len(loaded_vocab)}</h2>
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
                    st.markdown(f"""
                    <div class='recommendation-box'>
                        <strong style='color: #667eea; font-size: 1.2rem;'>#{i}</strong>
                        <p style='margin: 5px 0 0 0; color: #2d3748;'>{paper}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                papers_text = "\n\n".join([f"{i}. {p}" for i, p in enumerate(papers, 1)])
                st.download_button("📥 Download", papers_text, "recommendations.txt", "text/plain")
            else:
                st.info("💡 No recommendations found. Try different input.")
        
        # CATEGORY PREDICTION
        if actual_abstract:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🏷️ Predicted Subject Areas")
            
            if st.session_state.predictions is None or inputs_changed:
                with st.spinner("🏷️ Predicting categories..."):
                    st.session_state.predictions = predict_category(actual_abstract)
            
            categories = st.session_state.predictions
            
            if categories and len(categories) > 0:
                html = "".join([f'<span class="category-badge">{cat}</span>' for cat in categories])
                st.markdown(html, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"""
                💡 **Model Info**: Multi-Layer Perceptron trained on {len(sentences):,} ArXiv papers  
                📊 **Categories Found**: {len(categories)}  
                🎯 **Model Input Size**: {model_input_size:,} features
                """)
            else:
                st.warning("""
                ⚠️ No categories predicted. Possible reasons:
                - Abstract too short or unclear
                - Model confidence below threshold (0.5)
                - Try a more detailed abstract
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
    - 🎯 {len(loaded_vocab)} categories
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
    - **Model**: MLP (512→256→{len(loaded_vocab)})
    - **Features**: {model_input_size:,} TF-IDF
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
        Categories: {len(loaded_vocab)} | 
        Model Input: {model_input_size:,} features
    </p>
    <p style='font-size: 0.85rem; margin-top: 10px;'>
        TensorFlow • PyTorch • Sentence Transformers • Streamlit
    </p>
</div>
""", unsafe_allow_html=True)