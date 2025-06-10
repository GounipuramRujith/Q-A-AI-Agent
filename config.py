"""
Configuration settings for the RAG Streamlit Application
"""

import os
from typing import Dict, Any

# Model configurations
MODEL_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "generation_model": "google/flan-t5-base",
    "max_length": 512,
    "temperature": 0.7,
    "top_k_retrieval": 3,
    "batch_size": 64
}

# UI configurations
UI_CONFIG = {
    "page_title": "RAG Q&A System",
    "page_icon": "🔍",
    "layout": "wide",
    "sidebar_width": 300,
    "max_memory_turns": 5
}

# File configurations
FILE_CONFIG = {
    "default_csv_path": "temples_optimized.csv",
    "text_column": "Description",
    "upload_max_size": 10,  # MB
    "supported_formats": [".csv", ".txt", ".json"],
    "validation_enabled": True,  # Enable automatic validation
    "auto_clean": True  # Enable automatic data cleaning
}

# Session state keys
SESSION_KEYS = {
    "rag_pipeline": "rag_pipeline",
    "conversation_history": "conversation_history",
    "uploaded_data": "uploaded_data",
    "embeddings_computed": "embeddings_computed",
    "processing_status": "processing_status"
}

# Sample data for demonstration
SAMPLE_DATA = [
    {
        "templeName": "Kedarnath Temple",
        "Description": "Ancient temple dedicated to Lord Shiva, located in the Garhwal Himalayas of Uttarakhand. One of the twelve Jyotirlingas and part of the Char Dham pilgrimage.",
        "Location": "Uttarakhand",
        "Coordinates": "30.7346°N, 79.0669°E"
    },
    {
        "templeName": "Badrinath Temple", 
        "Description": "Sacred Hindu temple dedicated to Lord Vishnu, situated in the town of Badrinath in Uttarakhand. One of the Char Dham pilgrimage sites.",
        "Location": "Uttarakhand",
        "Coordinates": "30.7433°N, 79.4938°E"
    },
    {
        "templeName": "Jagannath Temple",
        "Description": "Famous Hindu temple dedicated to Jagannath (Lord Krishna) located in Puri, Odisha. Known for the annual Rath Yatra festival.",
        "Location": "Puri, Odisha",
        "Coordinates": "19.8135°N, 85.8312°E"
    },
    {
        "templeName": "Amarnath Temple",
        "Description": "Sacred cave temple dedicated to Lord Shiva, located 141km from Srinagar in Jammu and Kashmir. Famous for the natural ice lingam formation.",
        "Location": "Jammu and Kashmir",
        "Coordinates": "34.2190°N, 75.4867°E"
    },
    {
        "templeName": "Vaishno Devi Temple",
        "Description": "Popular Hindu temple dedicated to Goddess Vaishno Devi, located in the Trikuta Mountains of Jammu and Kashmir.",
        "Location": "Jammu and Kashmir", 
        "Coordinates": "33.0311°N, 74.9500°E"
    },
    {
        "templeName": "Golden Temple",
        "Description": "Most sacred Sikh temple, also known as Harmandir Sahib, located in Amritsar, Punjab. Famous for its golden architecture and free community kitchen.",
        "Location": "Amritsar, Punjab",
        "Coordinates": "31.6200°N, 74.8765°E"
    },
    {
        "templeName": "Meenakshi Temple",
        "Description": "Historic Hindu temple dedicated to Goddess Meenakshi and Lord Sundareshwar, located in Madurai, Tamil Nadu. Known for its colorful gopurams.",
        "Location": "Madurai, Tamil Nadu",
        "Coordinates": "9.9195°N, 78.1193°E"
    },
    {
        "templeName": "Kashi Vishwanath Temple",
        "Description": "One of the most famous Hindu temples dedicated to Lord Shiva, located in Varanasi, Uttar Pradesh. One of the twelve Jyotirlingas.",
        "Location": "Varanasi, Uttar Pradesh",
        "Coordinates": "25.3103°N, 83.0077°E"
    },
    {
        "templeName": "Somnath Temple",
        "Description": "Ancient temple dedicated to Lord Shiva, located in Prabhas Patan, Gujarat. First among the twelve Jyotirlinga shrines.",
        "Location": "Gujarat",
        "Coordinates": "20.8880°N, 70.4013°E"
    },
    {
        "templeName": "Tirupati Temple",
        "Description": "Famous Hindu temple dedicated to Lord Venkateswara, located in the hill town of Tirumala, Andhra Pradesh. One of the richest temples in the world.",
        "Location": "Tirumala, Andhra Pradesh",
        "Coordinates": "13.6833°N, 79.3500°E"
    }
]

def get_config(config_type: str) -> Dict[str, Any]:
    """Get configuration dictionary by type"""
    configs = {
        "model": MODEL_CONFIG,
        "ui": UI_CONFIG,
        "file": FILE_CONFIG,
        "session": SESSION_KEYS
    }
    return configs.get(config_type, {})

def update_config(config_type: str, key: str, value: Any) -> bool:
    """Update a configuration value"""
    try:
        configs = {
            "model": MODEL_CONFIG,
            "ui": UI_CONFIG,
            "file": FILE_CONFIG,
            "session": SESSION_KEYS
        }
        
        if config_type in configs and key in configs[config_type]:
            configs[config_type][key] = value
            return True
        return False
    except Exception:
        return False 