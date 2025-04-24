# analysis/text_analyzer.py

from sentence_transformers import SentenceTransformer, util
import torch # Sentence Transformers uses PyTorch
import numpy as np
from typing import List, Union

# Load a pre-trained sentence transformer model
# Options: 'all-MiniLM-L6-v2' (fast, good baseline), 'all-mpnet-base-v2' (better quality)
# Choose based on desired speed vs accuracy trade-off.
try:
    # It's often better to load the model once when the module is imported
    # or within a class initializer if you structure it that way.
    MODEL_NAME = 'all-MiniLM-L6-v2'
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    SIMILARITY_MODEL = SentenceTransformer(MODEL_NAME)
    print("Sentence transformer model loaded successfully.")
except Exception as e:
    print(f"Error loading sentence transformer model '{MODEL_NAME}': {e}")
    print("Ensure 'sentence-transformers' and 'torch' are installed: pip install sentence-transformers torch")
    SIMILARITY_MODEL = None # Set to None if loading fails

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculates the cosine similarity between the embeddings of two texts.

    Args:
        text1: The first string.
        text2: The second string.

    Returns:
        The cosine similarity score (float between -1 and 1, typically 0 to 1 for sentence embeddings).
        Returns 0.0 if the model failed to load or an error occurs.
    """
    if SIMILARITY_MODEL is None:
        print("Error: Similarity model not loaded.")
        return 0.0
    if not text1 or not text2:
        # Handle empty strings to avoid errors in embedding - similarity is arguably 0
        return 0.0

    try:
        # Encode the texts into embeddings
        # Use normalize_embeddings=True for cosine similarity
        embeddings = SIMILARITY_MODEL.encode(
            [text1, text2],
            convert_to_tensor=True,
            normalize_embeddings=True # Normalizing makes dot product equivalent to cosine similarity
        )

        # Calculate cosine similarity
        # Since normalized, dot product is sufficient and efficient
        cosine_score = torch.mm(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0).T)

        # Convert tensor result to float
        similarity = cosine_score.item()

        # Ensure the value is within a reasonable range (e.g., clamp potential small floating point errors)
        similarity = max(0.0, min(1.0, similarity))

        return similarity

    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        # Consider logging the texts involved if debugging is needed
        return 0.0

def tokenize_text(text: str) -> List[str]:
    """
    Simple text tokenization (splitting by whitespace).
    Note: For advanced RAG/LLM use, specific tokenizers (like those from
    Hugging Face or used internally by LangChain/LLMs) are often more relevant
    for context window limits, but this provides basic splitting.

    Args:
        text: The input string.

    Returns:
        A list of tokens (words).
    """
    if not text:
        return []
    return text.split() # Simple whitespace split

if __name__ == '__main__':
    print("\n--- Testing Text Analyzer ---")

    if SIMILARITY_MODEL:
        sample1 = "The quick brown fox jumps over the lazy dog."
        sample2 = "A fast, dark-colored fox leaps above a sleepy canine."
        sample3 = "This sentence is completely different."
        empty_sample = ""

        sim12 = calculate_cosine_similarity(sample1, sample2)
        sim13 = calculate_cosine_similarity(sample1, sample3)
        sim1_empty = calculate_cosine_similarity(sample1, empty_sample)

        print(f"\nSimilarity ('{sample1}' vs '{sample2}'): {sim12:.4f} (Expected: High)")
        print(f"Similarity ('{sample1}' vs '{sample3}'): {sim13:.4f} (Expected: Low)")
        print(f"Similarity ('{sample1}' vs ''): {sim1_empty:.4f} (Expected: 0.0)")

        tokens = tokenize_text(sample1)
        print(f"\nTokens for '{sample1}': {tokens}")
    else:
        print("\nSkipping similarity tests as the model could not be loaded.")