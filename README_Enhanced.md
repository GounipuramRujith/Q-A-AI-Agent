# 🏛️ MusicRAG - Enhanced Temple Information RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system for exploring Indian temples, built with Streamlit, FAISS, and advanced NLP models.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Optimization](#data-optimization)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Data Validation](#data-validation)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This enhanced RAG system provides intelligent question-answering capabilities about Indian temples using:

- **Advanced Embedding Models**: Sentence-Transformers for semantic search
- **FAISS Vector Search**: Efficient similarity search with normalized embeddings
- **T5 Generation**: High-quality answer generation with context
- **Streamlit Interface**: Interactive web application
- **Comprehensive Data**: 55+ temples with detailed information
- **Data Validation**: Automated quality checks and cleaning

## ✨ Features

### Core Features
- 🔍 **Semantic Search**: Natural language queries with contextual understanding
- 🤖 **AI Generation**: Intelligent answers using T5-based models
- 💬 **Conversation Memory**: Multi-turn conversation support
- 📊 **Rich Data**: Enhanced temple dataset with 16+ attributes per temple
- 🗺️ **Geographic Data**: Accurate coordinates and distance information
- 📱 **Responsive UI**: Modern Streamlit interface with chat functionality

### Enhanced Features
- ✅ **Data Validation**: Comprehensive data quality checks
- 🔧 **Data Preprocessing**: Automated cleaning and standardization
- 📈 **Performance Monitoring**: Real-time metrics and statistics
- 🎨 **Multiple Architectures**: Support for various temple architectural styles
- 🌟 **Cultural Context**: Rich descriptions with historical significance

## 📁 Project Structure

```
MusicRAG/
├── 📄 temples.csv                 # Original temple dataset
├── 📄 temples_optimized.csv       # Enhanced and cleaned dataset
├── 📄 requirements.txt            # Python dependencies
├── 📄 README_Enhanced.md          # This comprehensive documentation
├── 🐍 app.py                      # Main Streamlit application
├── 🐍 rag_pipeline.py             # RAG processing pipeline
├── 🐍 data_handler.py             # Data loading and processing
├── 🐍 ui_components.py            # UI components and styling
├── 🐍 config.py                   # Configuration settings
├── 🐍 data_validator.py           # Data validation and cleaning
└── 🐍 run_app.py                  # Application launcher
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (recommended for embedding models)
- Internet connection (for model downloads)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd MusicRAG
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv temple_rag_env
   source temple_rag_env/bin/activate  # On Windows: temple_rag_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Validate Data (Optional)**
   ```bash
   python data_validator.py
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📊 Data Optimization

### What's New in the Optimized Dataset

#### Enhanced Columns
- **State**: Separate state information for better geographic organization
- **Category**: Temple classification (Jyotirlinga, Char Dham, UNESCO Site, etc.)
- **Significance**: Cultural and religious importance
- **Architecture**: Architectural style (Dravidian, Kalinga, Modern, etc.)
- **BestTimeToVisit**: Optimal visiting periods
- **EstimatedVisitDuration**: Recommended visit duration

#### Data Quality Improvements
- ✅ **Corrected Coordinates**: Fixed incorrect locations (e.g., Konark Sun Temple)
- ✅ **Standardized Formatting**: Consistent naming and description styles
- ✅ **Enhanced Descriptions**: More detailed and informative content
- ✅ **Geographical Accuracy**: Verified coordinates within India boundaries
- ✅ **Metadata Enrichment**: Added cultural and architectural context

#### Validation Features
- Coordinate boundary checking for India
- Description length validation
- Duplicate detection
- Missing data identification
- Statistical analysis

## 💻 Usage

### Basic Usage

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Navigate to the Interface**
   - Open your browser to `http://localhost:8501`

3. **Ask Questions**
   - "Tell me about Char Dham temples"
   - "Which temples are in Tamil Nadu?"
   - "What is special about Khajuraho temples?"
   - "Show me temples with Dravidian architecture"

### Advanced Features

#### Data Upload
- Upload your own CSV files with temple data
- Automatic validation and cleaning
- Support for multiple file formats (CSV, JSON, TXT)

#### Conversation Management
- Multi-turn conversations with memory
- Clear conversation history
- Export conversation logs

#### Search Options
- Adjust number of retrieved documents
- Enable/disable conversation memory
- Configure search parameters

## 🔧 API Reference

### Core Classes

#### `TempleDataValidator`
```python
validator = TempleDataValidator()
results = validator.validate_dataframe(df)
```

#### `RAGPipeline`
```python
pipeline = RAGPipeline(retriever, model_name="google/flan-t5-base")
answer = pipeline.generate_answer(question, top_k=3)
```

#### `CustomFaissRetriever`
```python
retriever = CustomFaissRetriever(embeddings, documents, embed_model)
results = retriever.retrieve(query, top_k=5)
```

### Configuration Options

```python
# Model Configuration
MODEL_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "generation_model": "google/flan-t5-base",
    "max_length": 512,
    "temperature": 0.7,
    "top_k_retrieval": 3,
    "batch_size": 64
}
```

## ✅ Data Validation

### Automatic Validation
Run the data validator to check:
- Coordinate accuracy
- Description quality
- Missing information
- Data consistency

```bash
python data_validator.py
```

### Validation Results
- **Errors**: Critical issues that must be fixed
- **Warnings**: Minor issues or suggestions
- **Statistics**: Comprehensive data analysis
- **Cleaned Data**: Automatically processed dataset

## ⚡ Performance

### Optimization Features
- **FAISS Indexing**: Efficient vector search with normalized embeddings
- **Caching**: Streamlit caching for models and embeddings
- **Batch Processing**: Optimized batch sizes for embedding computation
- **Memory Management**: Conversation memory with configurable limits

### Benchmarks
- **Search Speed**: ~100ms for similarity search
- **Generation Time**: ~2-3s for T5-based answers
- **Memory Usage**: ~2GB for loaded models
- **Dataset Size**: 55 temples with full metadata

## 🗺️ Temple Coverage

### Geographic Distribution
- **North India**: 15 temples (Char Dham, Kashmir, Punjab)
- **South India**: 25 temples (Tamil Nadu, Karnataka, Kerala, Andhra Pradesh)
- **East India**: 8 temples (West Bengal, Odisha, Assam)
- **West India**: 7 temples (Gujarat, Maharashtra, Rajasthan)

### Architectural Styles
- **Dravidian**: 18 temples
- **Kalinga**: 6 temples
- **Indo-Islamic**: 3 temples
- **Modern**: 4 temples
- **Traditional Regional**: 24 temples

### Religious Significance
- **12 Jyotirlingas**: 6 covered
- **4 Char Dhams**: All 4 covered
- **108 Divya Desams**: 8 covered
- **52 Shakti Peethas**: 7 covered
- **UNESCO Sites**: 5 covered

## 🤝 Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Add new temples or enhance existing data
4. Run data validation
5. Submit a pull request

### Data Contribution Guidelines
- Follow the established CSV format
- Provide accurate coordinates
- Include comprehensive descriptions
- Verify cultural and historical information
- Add architectural and significance details

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **Cultural Heritage**: Information sourced from archaeological surveys and cultural documentation
- **Technical Stack**: Built with Streamlit, FAISS, Transformers, and Sentence-Transformers
- **Community**: Thanks to contributors and temple documentation projects

## 🆘 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the validation reports

---

**Happy Exploring! 🏛️✨**

*Experience the rich cultural heritage of Indian temples through intelligent conversation.* 