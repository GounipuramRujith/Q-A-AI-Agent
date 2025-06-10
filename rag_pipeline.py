"""
RAG Pipeline for Streamlit Application
Implements FAISS-based retrieval and T5 generation with memory
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from config import get_config  # Import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Placeholder for non-Streamlit environments ---
class StreamlitPlaceholder:
    def info(self, message):
        logger.info(f"[Streamlit Info] {message}")

    def error(self, message):
        logger.error(f"[Streamlit Error] {message}")

    def success(self, message):
        logger.info(f"[Streamlit Success] {message}")

    def warning(self, message):
        logger.warning(f"[Streamlit Warning] {message}")

    def spinner(self, text="Processing..."):
        class SpinnerContext:
            def __enter__(self):
                logger.info(f"[Streamlit Spinner] {text}")

            def __exit__(self, exc_type, exc_val, exc_tb):
                logger.info("[Streamlit Spinner] Finished.")

        return SpinnerContext()

try:
    import streamlit as st
except ImportError:
    st = StreamlitPlaceholder()
# -----------------------------------------------------------

@st.cache_resource
def load_embedding_model(model_name: str) -> Optional[SentenceTransformer]:
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Embedding model {model_name} loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        return None

@st.cache_resource
def load_generation_pipeline(model_name: str) -> Optional[Any]:
    logger.info(f"Loading generation model: {model_name}")
    try:
        gen_pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info(f"Generation model {model_name} loaded successfully.")
        return gen_pipeline
    except Exception as e:
        logger.error(f"Failed to load generation model {model_name}: {e}")
        return None

class CustomFaissRetriever:
    def __init__(self, documents: List[str], embedding_model_name: str):
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self.embedding_model = load_embedding_model(embedding_model_name)
        self.index = None
        self.document_embeddings = None
        self._build_index()

    def _build_index(self):
        if self.embedding_model is None:
            logger.error("Embedding model not loaded. Cannot build FAISS index.")
            return

        logger.info("Computing embeddings...")
        try:
            if self.document_embeddings is None:
                self.document_embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
                logger.info(f"Computed embeddings shape: {self.document_embeddings.shape}")

            logger.info("Creating FAISS retriever...")
            self.index = faiss.IndexFlatL2(self.document_embeddings.shape[1])
            self.index.add(np.array(self.document_embeddings).astype('float32'))
            logger.info(f"FAISS index built with {self.index.ntotal} vectors of dimension {self.index.d}")
        except Exception as e:
            logger.error(f"Failed to compute embeddings or build FAISS index: {e}")
            self.index = None

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if self.index is None or self.embedding_model is None:
            logger.error("Retriever not initialized. Cannot perform retrieval.")
            return []

        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype("float32"), top_k)

        results = []
        for i, doc_idx in enumerate(I[0]):
            if doc_idx < len(self.documents):
                results.append((self.documents[doc_idx], float(D[0][i])))
        return results

    def get_stats(self) -> Dict:
        return {
            "num_documents": len(self.documents),
            "embedding_dimension": self.document_embeddings.shape[1] if self.document_embeddings is not None else 0,
            "index_size": self.index.ntotal if self.index is not None else 0
        }

class RAGPipeline:
    def __init__(self, documents: List[str], model_config: Dict):
        self.model_config = model_config
        self.retriever = CustomFaissRetriever(documents, model_config["embedding_model"])
        self.generator = load_generation_pipeline(model_config["generation_model"])
        self.conversation_memory = []
        if self.retriever is None or self.generator is None:
            raise ValueError("Failed to initialize retriever or generator.")
        logger.info("RAG pipeline initialized.")

    def _get_memory_context(self) -> str:
        max_turns = get_config("ui").get("max_memory_turns", 3)
        recent_history = self.conversation_memory[-max_turns:]
        context_parts = []
        for entry in recent_history:
            context_parts.append(f"Q: {entry['question']}\nA: {entry['answer']}")
        return "\n".join(context_parts)

    def generate_answer(self, question: str, top_k: int = 3, use_memory: bool = False) -> Dict:
        if self.retriever is None or self.generator is None:
            logger.error("RAG pipeline not fully initialized. Cannot generate answer.")
            return {"answer": "Error: Pipeline not ready.", "retrieved_docs": [], "confidence": 0.0}

        start_time = time.time()
        retrieval_start_time = time.time()
        retrieved_docs_with_scores = self.retriever.retrieve(question, top_k=top_k)
        retrieved_docs_text = [doc for doc, score in retrieved_docs_with_scores]
        retrieval_time = time.time() - retrieval_start_time

        context = "\n".join(retrieved_docs_text)
        memory_context = ""
        if use_memory and self.conversation_memory:
            memory_context = self._get_memory_context()
            if memory_context:
                context = f"{memory_context}\n{context}"

        generation_start_time = time.time()
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        try:
            result = self.generator(
                prompt,
                max_new_tokens=self.model_config["max_length"],
                temperature=self.model_config["temperature"],
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.generator.model.config.eos_token_id
            )
            answer = result[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            answer = "Error: Could not generate answer."

        generation_time = time.time() - generation_start_time

        self.conversation_memory.append({"question": question, "answer": answer})
        total_time = time.time() - start_time

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs_with_scores,
            "confidence": 0.0,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "context_length": len(context),
            "memory_turns": len(self.conversation_memory)
        }

    def clear_memory(self):
        self.conversation_memory = []
        logger.info("Conversation memory cleared.")

    def get_pipeline_stats(self) -> Dict:
        retriever_stats = self.retriever.get_stats() if self.retriever else {}
        return {
            "generator_model": self.model_config["generation_model"],
            "retriever": retriever_stats,
            "memory_size": len(self.conversation_memory),
            "max_memory_turns": get_config("ui").get("max_memory_turns", 3),
            "generation_config": {
                "max_length": self.model_config["max_length"],
                "temperature": self.model_config["temperature"]
            }
        }

@st.cache_resource
def create_rag_pipeline(documents: List[str], model_config: Dict) -> Optional[RAGPipeline]:
    logger.info("Initializing RAG pipeline...")
    try:
        pipeline_instance = RAGPipeline(documents, model_config)
        return pipeline_instance
    except Exception as e:
        logger.error(f"Failed to create RAG pipeline: {e}")
        return None

