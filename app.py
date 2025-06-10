"""
Main Streamlit Application for RAG Q&A System
Integrates all components for a complete RAG experience
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import time
import traceback

# Import custom modules
from config import get_config, SAMPLE_DATA
from data_handler import DataHandler, load_cached_data
from rag_pipeline import create_rag_pipeline, RAGPipeline
from ui_components import (
    render_sidebar, render_file_uploader, render_data_summary,
    render_question_input, render_answer_display, render_conversation_history,
    render_system_stats, render_analytics_dashboard, render_error_message,
    render_success_message, render_info_message, show_loading_spinner
)

# Page configuration
ui_config = get_config("ui")
st.set_page_config(
    page_title=ui_config["page_title"],
    page_icon=ui_config["page_icon"],
    layout=ui_config["layout"],
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    session_keys = get_config("session")
    
    # Initialize all session state variables
    if session_keys["rag_pipeline"] not in st.session_state:
        st.session_state[session_keys["rag_pipeline"]] = None
    
    if session_keys["conversation_history"] not in st.session_state:
        st.session_state[session_keys["conversation_history"]] = []
    
    if session_keys["uploaded_data"] not in st.session_state:
        st.session_state[session_keys["uploaded_data"]] = None
    
    if session_keys["embeddings_computed"] not in st.session_state:
        st.session_state[session_keys["embeddings_computed"]] = False
    
    if session_keys["processing_status"] not in st.session_state:
        st.session_state[session_keys["processing_status"]] = "idle"
    
    # Additional UI state
    if "clear_history" not in st.session_state:
        st.session_state.clear_history = False
    
    if "current_config" not in st.session_state:
        st.session_state.current_config = {}

def handle_file_upload(uploaded_file, data_handler: DataHandler):
    """Handle file upload and data processing"""
    session_keys = get_config("session")
    
    if uploaded_file is not None:
        # Check if this is a new file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if (st.session_state[session_keys["uploaded_data"]] is None or 
            st.session_state[session_keys["uploaded_data"]].get("file_id") != file_id):
            
            try:
                with st.spinner("Processing uploaded file..."):
                    # Process the file
                    texts, df, message = data_handler.process_uploaded_file(uploaded_file)
                    
                    if texts:
                        # Get data summary
                        summary = data_handler.get_data_summary(df, texts)
                        
                        # Store in session state
                        st.session_state[session_keys["uploaded_data"]] = {
                            "file_id": file_id,
                            "texts": texts,
                            "dataframe": df,
                            "summary": summary,
                            "filename": uploaded_file.name
                        }
                        
                        # Reset pipeline and embeddings
                        st.session_state[session_keys["rag_pipeline"]] = None
                        st.session_state[session_keys["embeddings_computed"]] = False
                        
                        render_success_message(message)
                        return True
                    else:
                        render_error_message("Failed to process file", message)
                        return False
                        
            except Exception as e:
                render_error_message("Error processing file", str(e))
                return False
    
    return st.session_state[session_keys["uploaded_data"]] is not None

def setup_sample_data():
    """Setup sample data for demonstration"""
    session_keys = get_config("session")
    data_handler = DataHandler()
    
    if st.session_state[session_keys["uploaded_data"]] is None:
        try:
            # Create sample DataFrame
            df = data_handler.create_sample_data()
            texts = df['Description'].tolist()
            summary = data_handler.get_data_summary(df, texts)
            
            # Store in session state
            st.session_state[session_keys["uploaded_data"]] = {
                "file_id": "sample_data",
                "texts": texts,
                "dataframe": df,
                "summary": summary,
                "filename": "Sample Temple Data"
            }
            
            render_info_message("Using sample temple data for demonstration")
            return True
            
        except Exception as e:
            render_error_message("Error setting up sample data", str(e))
            return False
    
    return True

def create_pipeline(config: Dict):
    """Create or update RAG pipeline"""
    session_keys = get_config("session")
    
    uploaded_data = st.session_state[session_keys["uploaded_data"]]
    if not uploaded_data:
        return False
    
    try:
        # Update model configuration
        model_config = {
            "embedding_model": config["embedding_model"],
            "generation_model": config["generation_model"],
            "max_length": config["max_length"],
            "temperature": config["temperature"],
            "batch_size": get_config("model")["batch_size"]
        }
        
        # Create pipeline
        texts = uploaded_data["texts"]
        rag_pipeline = create_rag_pipeline(texts, model_config)
        
        if rag_pipeline:
            st.session_state[session_keys["rag_pipeline"]] = rag_pipeline
            st.session_state[session_keys["embeddings_computed"]] = True
            st.session_state.current_config = config.copy()
            render_success_message("RAG pipeline ready!")
            return True
        else:
            render_error_message("Failed to create RAG pipeline")
            return False
            
    except Exception as e:
        render_error_message("Error creating pipeline", str(e))
        return False

def process_question(question: str, config: Dict):
    """Process user question and generate answer"""
    session_keys = get_config("session")
    
    rag_pipeline = st.session_state[session_keys["rag_pipeline"]]
    if not rag_pipeline:
        render_error_message("RAG pipeline not initialized")
        return None
    
    try:
        with st.spinner("Generating answer..."):
            # Generate answer
            result = rag_pipeline.generate_answer(
                question=question,
                top_k=config["top_k"],
                use_memory=config["use_memory"]
            )
            
            # Store in conversation history
            if result and result.get("answer"):
                conversation_entry = {
                    "question": question,
                    "answer": result["answer"],
                    "timestamp": time.time()
                }
                st.session_state[session_keys["conversation_history"]].append(conversation_entry)
            
            return result
            
    except Exception as e:
        render_error_message("Error processing question", str(e))
        return None

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🔍 RAG Q&A System")
    st.markdown("*Retrieval-Augmented Generation for Question Answering*")
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Data handler
    data_handler = DataHandler()
    session_keys = get_config("session")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data & Setup", "❓ Q&A Interface", "💬 History", "📊 Analytics"])
    
    with tab1:
        st.header("Data Management")
        
        # File upload section
        uploaded_file = render_file_uploader()
        
        # Handle file upload or setup sample data
        if uploaded_file:
            data_ready = handle_file_upload(uploaded_file, data_handler)
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📋 Use Sample Data", help="Load sample temple data for testing"):
                    data_ready = setup_sample_data()
                else:
                    data_ready = st.session_state[session_keys["uploaded_data"]] is not None
            
            with col2:
                if st.button("🗑️ Clear Data", help="Clear current data and start fresh"):
                    # Clear all session state
                    for key in session_keys.values():
                        st.session_state[key] = None
                    st.session_state[session_keys["conversation_history"]] = []
                    st.session_state[session_keys["embeddings_computed"]] = False
                    render_info_message("Data cleared. Please upload new data or use sample data.")
                    data_ready = False
        
        # Show data summary if available
        if st.session_state[session_keys["uploaded_data"]]:
            uploaded_data = st.session_state[session_keys["uploaded_data"]]
            render_data_summary(uploaded_data["summary"])
            
            # Pipeline setup
            st.header("RAG Pipeline Setup")
            
            # Check if configuration changed
            config_changed = (st.session_state.current_config != config)
            pipeline_exists = st.session_state[session_keys["rag_pipeline"]] is not None
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🚀 Initialize Pipeline", disabled=not data_ready):
                    create_pipeline(config)
            
            with col2:
                if config_changed and pipeline_exists:
                    st.warning("⚠️ Configuration changed. Re-initialize pipeline to apply changes.")
            
            # Show pipeline status
            if st.session_state[session_keys["rag_pipeline"]]:
                st.success("✅ Pipeline ready for questions!")
                
                # Show system stats
                stats = st.session_state[session_keys["rag_pipeline"]].get_pipeline_stats()
                render_system_stats(stats)
    
    with tab2:
        if st.session_state[session_keys["rag_pipeline"]] is None:
            st.warning("⚠️ Please initialize the RAG pipeline in the 'Data & Setup' tab first.")
        else:
            # Question input
            question = render_question_input()
            
            # Process question
            if question and st.button("🔍 Get Answer", type="primary"):
                result = process_question(question, config)
                if result:
                    render_answer_display(result)
            
            # Show recent answer if available
            if st.session_state[session_keys["conversation_history"]]:
                st.divider()
                latest = st.session_state[session_keys["conversation_history"]][-1]
                st.write("**Latest Q&A:**")
                st.write(f"**Q:** {latest['question']}")
                st.write(f"**A:** {latest['answer']}")
    
    with tab3:
        # Handle clear history action
        if st.session_state.get("clear_history", False):
            st.session_state[session_keys["conversation_history"]] = []
            if st.session_state[session_keys["rag_pipeline"]]:
                st.session_state[session_keys["rag_pipeline"]].clear_memory()
            st.session_state.clear_history = False
            render_success_message("Conversation history cleared!")
        
        # Render conversation history
        history = st.session_state[session_keys["conversation_history"]]
        render_conversation_history(history)
    
    with tab4:
        # Analytics dashboard
        history = st.session_state[session_keys["conversation_history"]]
        
        if history:
            render_analytics_dashboard(history)
        else:
            st.info("📊 No conversation data available for analytics. Start asking questions to see analytics!")
        
        # Additional system information
        if st.session_state[session_keys["rag_pipeline"]]:
            st.divider()
            stats = st.session_state[session_keys["rag_pipeline"]].get_pipeline_stats()
            render_system_stats(stats)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🔍 RAG Q&A System | Built with Streamlit, FAISS, and Transformers</p>
        <p>Upload your data or use sample data to get started!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred:")
        st.code(traceback.format_exc())
        st.info("Please refresh the page and try again.") 