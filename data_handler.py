"""
Data Handler for RAG Streamlit Application
Handles data loading, preprocessing, and validation
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Optional, Tuple
import json
from config import SAMPLE_DATA, get_config

class DataHandler:
    def __init__(self):
        self.file_config = get_config("file")
        self.supported_formats = self.file_config["supported_formats"]
        
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration"""
        return pd.DataFrame(SAMPLE_DATA)
    
    def validate_file(self, uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size
        if uploaded_file.size > self.file_config["upload_max_size"] * 1024 * 1024:
            return False, f"File size exceeds {self.file_config['upload_max_size']}MB limit"
        
        # Check file format
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension not in self.supported_formats:
            return False, f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}"
        
        return True, "File validation successful"
    
    def load_csv_data(self, file_path_or_buffer, text_column: str = None) -> Tuple[List[str], pd.DataFrame]:
        """Load data from CSV file or buffer"""
        try:
            if isinstance(file_path_or_buffer, str):
                df = pd.read_csv(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)
            
            # Auto-detect text column if not specified
            if text_column is None:
                text_column = self._detect_text_column(df)
            
            if text_column not in df.columns:
                available_columns = df.columns.tolist()
                raise ValueError(f"Column '{text_column}' not found. Available columns: {available_columns}")
            
            # Clean and extract text data
            df_clean = df[~df[text_column].isna()].reset_index(drop=True)
            texts = df_clean[text_column].astype(str).tolist()
            
            return texts, df_clean
            
        except Exception as e:
            st.error(f"Error loading CSV data: {str(e)}")
            return [], pd.DataFrame()
    
    def load_txt_data(self, file_buffer) -> Tuple[List[str], pd.DataFrame]:
        """Load data from TXT file"""
        try:
            content = file_buffer.read().decode('utf-8')
            # Split by double newlines or single newlines
            texts = [text.strip() for text in content.split('\n\n') if text.strip()]
            if len(texts) <= 1:
                texts = [text.strip() for text in content.split('\n') if text.strip()]
            
            # Create DataFrame for consistency
            df = pd.DataFrame({'text': texts, 'source': 'uploaded_txt'})
            
            return texts, df
            
        except Exception as e:
            st.error(f"Error loading TXT data: {str(e)}")
            return [], pd.DataFrame()
    
    def load_json_data(self, file_buffer) -> Tuple[List[str], pd.DataFrame]:
        """Load data from JSON file"""
        try:
            content = json.load(file_buffer)
            
            if isinstance(content, list):
                # List of dictionaries or strings
                if all(isinstance(item, dict) for item in content):
                    df = pd.DataFrame(content)
                    text_column = self._detect_text_column(df)
                    texts = df[text_column].astype(str).tolist()
                else:
                    texts = [str(item) for item in content]
                    df = pd.DataFrame({'text': texts, 'source': 'uploaded_json'})
            elif isinstance(content, dict):
                # Single dictionary or nested structure
                df = pd.DataFrame([content])
                text_column = self._detect_text_column(df)
                texts = df[text_column].astype(str).tolist()
            else:
                texts = [str(content)]
                df = pd.DataFrame({'text': texts, 'source': 'uploaded_json'})
            
            return texts, df
            
        except Exception as e:
            st.error(f"Error loading JSON data: {str(e)}")
            return [], pd.DataFrame()
    
    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """Auto-detect the most likely text column"""
        # Priority order for text column detection
        priority_columns = ['description', 'text', 'content', 'summary', 'details', 'info']
        
        # Check for priority columns
        for col in priority_columns:
            matching_cols = [c for c in df.columns if col.lower() in c.lower()]
            if matching_cols:
                return matching_cols[0]
        
        # Fall back to first string column with reasonable length
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 20:  # Reasonable text length
                    return col
        
        # Last resort: first column
        return df.columns[0] if len(df.columns) > 0 else 'text'
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[List[str], pd.DataFrame, str]:
        """Process uploaded file and return texts and dataframe"""
        is_valid, message = self.validate_file(uploaded_file)
        
        if not is_valid:
            return [], pd.DataFrame(), message
        
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        try:
            if file_extension == '.csv':
                texts, df = self.load_csv_data(uploaded_file)
                message = f"Successfully loaded {len(texts)} documents from CSV file"
            elif file_extension == '.txt':
                texts, df = self.load_txt_data(uploaded_file)
                message = f"Successfully loaded {len(texts)} text segments from TXT file"
            elif file_extension == '.json':
                texts, df = self.load_json_data(uploaded_file)
                message = f"Successfully loaded {len(texts)} documents from JSON file"
            else:
                return [], pd.DataFrame(), f"Unsupported file format: {file_extension}"
            
            if len(texts) == 0:
                return [], pd.DataFrame(), "No valid text data found in the uploaded file"
            
            return texts, df, message
            
        except Exception as e:
            return [], pd.DataFrame(), f"Error processing file: {str(e)}"
    
    def get_data_summary(self, df: pd.DataFrame, texts: List[str]) -> Dict:
        """Get summary statistics of the loaded data"""
        if df.empty or not texts:
            return {}
        
        summary = {
            "total_documents": len(texts),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "avg_text_length": np.mean([len(text) for text in texts]),
            "min_text_length": min([len(text) for text in texts]),
            "max_text_length": max([len(text) for text in texts]),
            "sample_text": texts[0][:200] + "..." if texts and len(texts[0]) > 200 else texts[0] if texts else ""
        }
        
        # Add column-specific info
        for col in df.columns:
            if df[col].dtype == 'object':
                summary[f"{col}_unique_values"] = df[col].nunique()
            elif df[col].dtype in ['int64', 'float64']:
                summary[f"{col}_mean"] = df[col].mean()
        
        return summary
    
    def export_conversation(self, conversation_history: List[Dict], format: str = "json") -> str:
        """Export conversation history to specified format"""
        try:
            if format.lower() == "json":
                return json.dumps(conversation_history, indent=2, ensure_ascii=False)
            elif format.lower() == "csv":
                df = pd.DataFrame(conversation_history)
                return df.to_csv(index=False)
            elif format.lower() == "txt":
                lines = []
                for i, conv in enumerate(conversation_history, 1):
                    lines.append(f"Q{i}: {conv.get('question', '')}")
                    lines.append(f"A{i}: {conv.get('answer', '')}")
                    lines.append("-" * 50)
                return "\n".join(lines)
            else:
                return "Unsupported export format"
        except Exception as e:
            return f"Error exporting conversation: {str(e)}"

@st.cache_data
def load_cached_data(file_content: bytes, file_name: str) -> Tuple[List[str], pd.DataFrame, str]:
    """Cached data loading function for better performance"""
    data_handler = DataHandler()
    
    # Create a temporary file-like object
    import io
    file_buffer = io.BytesIO(file_content)
    file_buffer.name = file_name
    
    return data_handler.process_uploaded_file(file_buffer) 