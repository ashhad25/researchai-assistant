# 📚 Research Paper Assistant

AI-powered research paper discovery and classification system using deep learning and natural language processing.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)

## 🎯 Features

- **🔍 Semantic Search**: Find similar papers using Sentence Transformers
- **🏷️ Auto-Classification**: Predict subject areas with Multi-Layer Perceptron
- **⚡ Fast**: Query 41,000+ papers instantly
- **📊 Accurate**: Trained on ArXiv dataset with comprehensive evaluation

## 🛠️ Technology Stack

- **Backend**: TensorFlow, PyTorch, Scikit-learn
- **NLP**: Sentence Transformers (all-MiniLM-L6-v2)
- **Frontend**: Streamlit
- **Models**: 
  - Recommendation: Cosine similarity on sentence embeddings
  - Classification: MLP (512→256→165 categories)

## 📊 Dataset

- **Source**: ArXiv research papers
- **Size**: 41,105 papers
- **Categories**: 165 subject areas (multi-label)
- **Fields**: Computer Science, Mathematics, Physics, Statistics

## 🚀 Live Demo

[🌐 Try it here!](https://your-app-name.streamlit.app)

## 💻 Local Installation
```bash
# Clone repository
git clone https://github.com/yourusername/research-paper-assistant.git
cd research-paper-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (see below)

# Run application
streamlit run app.py
```

## 📥 Model Files

Due to file size limitations, model files are hosted separately:

**Download from**: [Google Drive Link](YOUR_LINK_HERE)

Place downloaded files in `models/` directory:
- `model.h5`
- `embeddings.pkl`
- `sentences.pkl`
- `vocab.pkl`
- `idf_weights.pkl`
- `text_vectorizer_config.pkl`

## 📖 Usage

1. **For Recommendations**: Enter a paper title or paste an abstract
2. **For Classification**: Paste a full abstract to get subject categories
3. Click "Analyze" to get results

## 🏗️ Architecture

### Classification Model
Input: TF-IDF vectors (90K+ features)
↓
Dense(512, ReLU) + Dropout(0.5)
↓
Dense(256, ReLU) + Dropout(0.5)
↓
Dense(165, Sigmoid) → Multi-label output

### Recommendation Model
Input: Text (title or abstract)
↓
Sentence-BERT Encoding (384D)
↓
Cosine Similarity
↓
Top-5 Similar Papers

## 📈 Performance

- **Element-wise Accuracy**: ~73%
- **F1 Score (Weighted)**: ~68%
- **Precision**: ~76%
- **Recall**: ~62%

*Note: Evaluated on multi-label classification across 165 categories*

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- ArXiv for the dataset
- Sentence Transformers team
- Streamlit for the framework

---

Built with ❤️ for researchers