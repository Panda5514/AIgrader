# analysis/rag_pipeline.py

import os
import pickle
import warnings
from typing import List, Optional, Tuple

# Langchain components
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings # Base class for type hinting

# Import specific embedding classes based on provider
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings # Use for OpenRouter embedding models

# Import project config
from utils import config

# --- Initialize Embeddings ---
embeddings_model: Optional[Embeddings] = None # Use base class for typing

try:
    print(f"Initializing Embeddings using Provider: {config.EMBEDDING_MODEL_PROVIDER}")
    if config.EMBEDDING_MODEL_PROVIDER == 'openrouter':
        if config.OPENROUTER_API_KEY:
            print(f"Using OpenRouter Embedding Model: {config.EMBEDDING_MODEL_NAME}")
            embeddings_model = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL_NAME,
                openai_api_key=config.OPENROUTER_API_KEY,
                openai_api_base=config.OPENROUTER_API_BASE,
                # Check if specific OR embedding models need headers
                # headers={"HTTP-Referer": "YOUR_SITE_URL", "X-Title": "YOUR_APP_NAME"}
                chunk_size=500 # Adjust chunk size for embedding requests if needed
            )
            print("OpenRouter Embeddings initialized via OpenAIEmbeddings.")
        else:
            warnings.warn("OpenRouter Embedding provider selected but API key missing.", UserWarning)

    elif config.EMBEDDING_MODEL_PROVIDER == 'google':
        if config.GOOGLE_API_KEY:
            print(f"Using Google Embedding Model: {config.EMBEDDING_MODEL_NAME}")
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model=config.EMBEDDING_MODEL_NAME,
                google_api_key=config.GOOGLE_API_KEY,
                task_type="retrieval_document" # Specify task type
            )
            print("Google Embeddings initialized.")
        else:
            warnings.warn("Google Embedding provider selected but API key missing.", UserWarning)
    else:
        warnings.warn(f"Unsupported EMBEDDING_MODEL_PROVIDER: {config.EMBEDDING_MODEL_PROVIDER}", UserWarning)

except Exception as e:
    warnings.warn(f"Fatal Error: Could not initialize embedding model: {e}", UserWarning)
    embeddings_model = None
    # Optional: Make this fatal if embeddings are essential
    # raise SystemExit("RAG Pipeline cannot function without embeddings.")

if not embeddings_model:
    print("WARNING: Embeddings model could not be initialized. RAG features will fail.")


# --- Vector Store Handling (No changes needed in logic, relies on initialized embeddings_model) ---

def create_vector_store(
        docs: List[Document],
        save_path: str = os.path.join(config.VECTOR_STORE_DIR, config.VECTOR_STORE_INDEX_NAME)
    ) -> Optional[FAISS]:
    """Creates and saves a FAISS vector store."""
    if not embeddings_model:
        print("Error: Embeddings model not initialized. Cannot create vector store.")
        return None
    # Rest of the function remains the same...
    if not docs:
        print("Warning: No documents provided to create vector store.")
        return None
    try:
        print(f"Creating FAISS vector store from {len(docs)} documents using {config.EMBEDDING_MODEL_PROVIDER} embeddings...")
        vector_store = FAISS.from_documents(docs, embeddings_model)
        print("Vector store created in memory.")
        vector_store.save_local(folder_path=save_path)
        print(f"Vector store saved successfully to: {save_path}")
        return vector_store
    except Exception as e:
        print(f"Error creating or saving vector store at '{save_path}': {e}")
        return None


def load_vector_store(
        load_path: str = os.path.join(config.VECTOR_STORE_DIR, config.VECTOR_STORE_INDEX_NAME)
    ) -> Optional[FAISS]:
    """Loads a FAISS vector store."""
    if not embeddings_model:
        print("Error: Embeddings model not initialized. Cannot load vector store.")
        return None
    # Rest of the function remains the same...
    faiss_file = os.path.join(load_path, "index.faiss")
    pkl_file = os.path.join(load_path, "index.pkl")
    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        print(f"Vector store not found at path: {load_path}")
        return None
    try:
        print(f"Loading vector store from: {load_path} using {config.EMBEDDING_MODEL_PROVIDER} embeddings...")
        vector_store = FAISS.load_local(
            folder_path=load_path,
            embeddings=embeddings_model, # Pass the correctly initialized model
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
        return vector_store
    except ModuleNotFoundError as mnfe:
         print(f"Error loading vector store: {mnfe}. Required class likely not found.")
         return None
    except Exception as e:
        print(f"Error loading vector store from '{load_path}': {e}")
        return None

# --- Text Processing and Document Creation (No changes needed) ---
def split_text_into_documents( # Function remains the same
    text: str, chunk_size: int = 1000, chunk_overlap: int = 150,
    source_identifier: str = "answer_sheet"
) -> List[Document]:
     if not text: return []
     try:
         text_splitter = RecursiveCharacterTextSplitter(
             chunk_size=chunk_size, chunk_overlap=chunk_overlap,
             length_function=len, add_start_index=True,
         )
         chunks = text_splitter.split_text(text)
         documents = [Document(page_content=chunk, metadata={"source": source_identifier, "chunk_number": i})
                      for i, chunk in enumerate(chunks)]
         print(f"Split text into {len(documents)} documents.")
         return documents
     except Exception as e:
         print(f"Error splitting text: {e}")
         return []


# --- Retrieval Function (No changes needed) ---
def retrieve_relevant_documents( # Function remains the same
        query: str, vector_store: FAISS, num_results: int = 3
    ) -> List[Tuple[Document, float]]:
     if not vector_store: return []
     if not query: return []
     try:
         print(f"Retrieving top {num_results} documents for query: '{query[:50]}...'")
         results_with_scores = vector_store.similarity_search_with_score(query, k=num_results)
         print(f"Retrieved {len(results_with_scores)} documents.")
         return results_with_scores
     except Exception as e:
         print(f"Error during document retrieval: {e}")
         return []

# --- Main Pipeline Orchestration (No changes needed) ---
def build_or_load_answer_sheet_store(answer_sheet_text: str) -> Optional[FAISS]: # Function remains the same
    store_path = os.path.join(config.VECTOR_STORE_DIR, config.VECTOR_STORE_INDEX_NAME)
    vector_store = load_vector_store(load_path=store_path)
    if vector_store:
        print("Successfully loaded existing answer sheet vector store.")
        return vector_store
    else:
        print("Existing vector store not found or failed to load. Building new store...")
        if not answer_sheet_text: return None
        documents = split_text_into_documents(answer_sheet_text, source_identifier="answer_sheet")
        if not documents: return None
        vector_store = create_vector_store(documents, save_path=store_path)
        if vector_store:
             print("Successfully built and saved new answer sheet vector store.")
             return vector_store
        else:
             print("Error: Failed to build the new vector store.")
             return None

# Example Usage
if __name__ == '__main__':
    print("\n--- Testing RAG Pipeline (with Provider Logic) ---")
    if not embeddings_model:
         print("Embeddings model not loaded. Cannot run RAG tests.")
    else:
        # Rest of the test code can remain similar, it uses the functions
        # which now internally rely on the configured embeddings_model
        print("\nTesting Text Splitting...")
        test_text = "Sentence one. Sentence two. " * 5
        test_docs = split_text_into_documents(test_text, chunk_size=20, chunk_overlap=5)
        print(f"Split into {len(test_docs)} documents.")

        print("\nTesting Vector Store Operations...")
        test_store_path = os.path.join(config.VECTOR_STORE_DIR, "test_index_openrouter")
        if os.path.exists(test_store_path):
            import shutil; shutil.rmtree(test_store_path) # Clean previous test

        sample_docs = [Document(page_content="Paris is France's capital."), Document(page_content="Water is H2O.")]
        created_store = create_vector_store(sample_docs, save_path=test_store_path)
        if created_store:
            loaded_store = load_vector_store(load_path=test_store_path)
            if loaded_store:
                results = retrieve_relevant_documents("What is water?", loaded_store, 1)
                print(f"Retrieval results for 'What is water?': {results}")
            else: print("Failed to load test store.")
        else: print("Failed to create test store.")
        # Clean up test index
        # if os.path.exists(test_store_path): import shutil; shutil.rmtree(test_store_path)