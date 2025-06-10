"""
UI Components for RAG Streamlit Application
Reusable UI components and widgets
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional, Any
import time
import json
from datetime import datetime
from config import get_config

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.title("🔍 RAG Configuration")
        
        # Model settings
        st.header("Model Settings")
        embedding_models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2", 
            "paraphrase-MiniLM-L6-v2",
            "multi-qa-MiniLM-L6-cos-v1"
        ]
        
        generation_models = [
            "google/flan-t5-base",
            "google/flan-t5-small",
            "google/flan-t5-large"
        ]
        
        embedding_model = st.selectbox(
            "Embedding Model",
            embedding_models,
            index=0,
            help="Model used for creating text embeddings"
        )
        
        generation_model = st.selectbox(
            "Generation Model", 
            generation_models,
            index=0,
            help="Model used for generating answers"
        )
        
        # Retrieval settings
        st.header("Retrieval Settings")
        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of documents to retrieve"
        )
        
        # Generation settings
        st.header("Generation Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in generation"
        )
        
        max_length = st.slider(
            "Max Length",
            min_value=128,
            max_value=1024,
            value=512,
            step=64,
            help="Maximum length of generated text"
        )
        
        # Memory settings
        st.header("Memory Settings")
        use_memory = st.checkbox(
            "Use Conversation Memory",
            value=True,
            help="Remember previous conversations"
        )
        
        max_memory_turns = st.slider(
            "Max Memory Turns",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of conversation turns to remember"
        )
        
        return {
            "embedding_model": embedding_model,
            "generation_model": generation_model,
            "top_k": top_k,
            "temperature": temperature,
            "max_length": max_length,
            "use_memory": use_memory,
            "max_memory_turns": max_memory_turns
        }

def render_file_uploader() -> Optional[Any]:
    """Render file upload component"""
    st.header("📁 Data Upload")
    
    file_config = get_config("file")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'txt', 'json'],
        help=f"Supported formats: {', '.join(file_config['supported_formats'])}"
    )
    
    if uploaded_file:
        # Show file details
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB",
            "File Type": uploaded_file.type
        }
        
        with st.expander("File Details"):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
    
    return uploaded_file

def render_data_summary(summary: Dict) -> None:
    """Render data summary component"""
    if not summary:
        return
    
    st.header("📊 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", summary.get("total_documents", 0))
    
    with col2:
        st.metric("Avg Text Length", f"{summary.get('avg_text_length', 0):.0f}")
    
    with col3:
        st.metric("Min Length", summary.get("min_text_length", 0))
    
    with col4:
        st.metric("Max Length", summary.get("max_text_length", 0))
    
    # Additional details in expander
    with st.expander("Detailed Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Information:**")
            columns = summary.get("column_names", [])
            for col in columns:
                unique_count = summary.get(f"{col}_unique_values", "N/A")
                st.write(f"- {col}: {unique_count} unique values")
        
        with col2:
            st.write("**Sample Text:**")
            sample_text = summary.get("sample_text", "No sample available")
            st.text_area("Sample", sample_text, height=150, disabled=True)

def render_question_input() -> str:
    """Render question input component"""
    st.header("❓ Ask a Question")
    
    # Predefined example questions
    example_questions = [
        "Where is the Kedarnath temple located?",
        "What is the significance of Badrinath temple?",
        "Tell me about the Golden Temple in Amritsar",
        "Which god is worshipped in Jagannath temple?",
        "What are the main features of Meenakshi temple?"
    ]
    
    # Question input methods
    input_method = st.radio(
        "Input Method",
        ["Type Question", "Select Example"],
        horizontal=True
    )
    
    if input_method == "Type Question":
        question = st.text_input(
            "Your Question:",
            placeholder="Ask anything about the uploaded data...",
            help="Type your question here"
        )
    else:
        question = st.selectbox(
            "Example Questions:",
            [""] + example_questions,
            help="Select from predefined questions"
        )
    
    return question

def render_answer_display(result: Dict) -> None:
    """Render answer display component"""
    if not result:
        return
    
    answer = result.get("answer", "")
    retrieved_docs = result.get("retrieved_docs", [])
    
    # Main answer
    st.header("💡 Answer")
    
    # Confidence indicator
    confidence = result.get("confidence", 0.0)
    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(answer)
    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}",
            delta=None,
            help="Based on retrieval similarity scores"
        )
    
    # Performance metrics
    with st.expander("Performance Metrics"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Retrieval Time", f"{result.get('retrieval_time', 0):.3f}s")
        
        with col2:
            st.metric("Generation Time", f"{result.get('generation_time', 0):.3f}s")
        
        with col3:
            st.metric("Context Length", result.get("context_length", 0))
        
        with col4:
            st.metric("Memory Turns", result.get("memory_turns", 0))
    
    # Retrieved documents
    if retrieved_docs:
        with st.expander(f"Retrieved Documents ({len(retrieved_docs)})"):
            for i, (doc, score) in enumerate(retrieved_docs, 1):
                st.write(f"**Document {i}** (Score: {score:.3f})")
                st.write(doc)
                st.divider()

def render_conversation_history(history: List[Dict]) -> None:
    """Render conversation history component"""
    if not history:
        st.info("No conversation history yet. Start by asking a question!")
        return
    
    st.header("💬 Conversation History")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Export as JSON"):
            json_data = json.dumps(history, indent=2, ensure_ascii=False)
            st.download_button(
                "Download JSON",
                json_data,
                "conversation_history.json",
                "application/json"
            )
    
    with col2:
        if st.button("📊 Export as CSV"):
            df = pd.DataFrame(history)
            csv_data = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                "conversation_history.csv",
                "text/csv"
            )
    
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.clear_history = True
    
    # Display history
    for i, entry in enumerate(reversed(history), 1):
        with st.expander(f"Conversation {len(history) - i + 1} - {datetime.fromtimestamp(entry['timestamp']).strftime('%H:%M:%S')}"):
            st.write("**Q:**", entry["question"])
            st.write("**A:**", entry["answer"])

def render_system_stats(stats: Dict) -> None:
    """Render system statistics component"""
    if not stats:
        return
    
    st.header("📈 System Statistics")
    
    retriever_stats = stats.get("retriever", {})
    generation_config = stats.get("generation_config", {})
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents", retriever_stats.get("num_documents", 0))
    
    with col2:
        st.metric("Embedding Dim", retriever_stats.get("embedding_dimension", 0))
    
    with col3:
        st.metric("Memory Size", stats.get("memory_size", 0))
    
    with col4:
        st.metric("Max Memory", stats.get("max_memory_turns", 0))
    
    # Detailed configuration
    with st.expander("Detailed Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Models:**")
            st.write(f"- Generator: {stats.get('generator_model', 'N/A')}")
            st.write(f"- Index Size: {retriever_stats.get('index_size', 0)}")
        
        with col2:
            st.write("**Generation Config:**")
            st.write(f"- Max Length: {generation_config.get('max_length', 'N/A')}")
            st.write(f"- Temperature: {generation_config.get('temperature', 'N/A')}")

def render_analytics_dashboard(history: List[Dict]) -> None:
    """Render analytics dashboard"""
    if not history:
        return
    
    st.header("📊 Analytics Dashboard")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(history)
    
    if 'timestamp' not in df.columns:
        return
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    
    # Question length analysis
    df['question_length'] = df['question'].str.len()
    df['answer_length'] = df['answer'].str.len()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Questions over time
        fig_time = px.scatter(
            df, 
            x='datetime', 
            y=range(len(df)),
            title="Questions Over Time",
            labels={'y': 'Question Number', 'datetime': 'Time'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Question length distribution
        fig_length = px.histogram(
            df,
            x='question_length',
            title="Question Length Distribution",
            labels={'question_length': 'Characters', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_length, use_container_width=True)
    
    # Activity by hour
    if len(df) > 1:
        hourly_activity = df.groupby('hour').size()
        fig_hourly = px.bar(
            x=hourly_activity.index,
            y=hourly_activity.values,
            title="Questions by Hour",
            labels={'x': 'Hour of Day', 'y': 'Number of Questions'}
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

def show_loading_spinner(message: str = "Processing..."):
    """Show loading spinner with message"""
    with st.spinner(message):
        time.sleep(0.1)  # Small delay for visual effect

def render_error_message(error: str, details: str = None):
    """Render error message component"""
    st.error(f"❌ {error}")
    if details:
        with st.expander("Error Details"):
            st.code(details)

def render_success_message(message: str):
    """Render success message component"""
    st.success(f"✅ {message}")

def render_info_message(message: str):
    """Render info message component"""
    st.info(f"ℹ️ {message}")

def render_warning_message(message: str):
    """Render warning message component"""
    st.warning(f"⚠️ {message}") 