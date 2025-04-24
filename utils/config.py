# utils/config.py

import os
import warnings # Use warnings module for better control

# Define the expected path for the API config file relative to this config file's location
# Assuming config.py is in utils/ and api.txt is in the parent directory (project root)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_CONFIG_FILE = os.path.join(_project_root, 'api.txt')

# --- Default Values ---
# Initialize with None or default values
OPENROUTER_API_KEY = None
GOOGLE_API_KEY = None
TEXT_MODEL_PROVIDER = 'openrouter' # Default provider assumption
TEXT_MODEL_NAME = 'mistralai/mistral-7b-instruct' # Default model
MULTIMODAL_MODEL_PROVIDER = 'openrouter'
MULTIMODAL_MODEL_NAME = 'google/gemini-pro-vision' # Default model (check OR availability)
EMBEDDING_MODEL_PROVIDER = 'openrouter'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2' # Default model (check OR availability)

# --- OpenRouter Specific ---
# Standard OpenRouter base URL
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# --- Parsing Function ---
def _parse_api_config(filepath):
    """Parses the api.txt file and returns a dictionary."""
    config_data = {}
    if not os.path.exists(filepath):
        warnings.warn(f"API configuration file not found at: {filepath}\n"
                      "Please create 'api.txt' in the project root with your API keys and model choices.\n"
                      "Using default models and providers. LLM calls will likely fail without API keys.",
                      UserWarning)
        return config_data # Return empty dict, defaults will be used

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config_data[key.strip()] = value.strip().strip('"\'') # Remove potential quotes
                    else:
                        warnings.warn(f"Skipping malformed line in {filepath}: {line}", UserWarning)
    except Exception as e:
        warnings.warn(f"Error reading API configuration file {filepath}: {e}\n"
                      "Using default models and providers. LLM calls may fail.",
                      UserWarning)
    return config_data

# --- Load Configuration ---
_loaded_config = _parse_api_config(API_CONFIG_FILE)

# --- Update Global Variables from Loaded Config ---
# Prioritize values from api.txt if they exist
OPENROUTER_API_KEY = _loaded_config.get('OPENROUTER_API_KEY', OPENROUTER_API_KEY)
GOOGLE_API_KEY = _loaded_config.get('GOOGLE_API_KEY', GOOGLE_API_KEY)

TEXT_MODEL_PROVIDER = _loaded_config.get('TEXT_MODEL_PROVIDER', TEXT_MODEL_PROVIDER).lower()
TEXT_MODEL_NAME = _loaded_config.get('TEXT_MODEL_NAME', TEXT_MODEL_NAME)

MULTIMODAL_MODEL_PROVIDER = _loaded_config.get('MULTIMODAL_MODEL_PROVIDER', MULTIMODAL_MODEL_PROVIDER).lower()
MULTIMODAL_MODEL_NAME = _loaded_config.get('MULTIMODAL_MODEL_NAME', MULTIMODAL_MODEL_NAME)

EMBEDDING_MODEL_PROVIDER = _loaded_config.get('EMBEDDING_MODEL_PROVIDER', EMBEDDING_MODEL_PROVIDER).lower()
EMBEDDING_MODEL_NAME = _loaded_config.get('EMBEDDING_MODEL_NAME', EMBEDDING_MODEL_NAME)

# --- Validation and Warnings ---
if TEXT_MODEL_PROVIDER == 'openrouter' and not OPENROUTER_API_KEY:
    warnings.warn("TEXT_MODEL_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is missing in api.txt.", UserWarning)
if MULTIMODAL_MODEL_PROVIDER == 'openrouter' and not OPENROUTER_API_KEY:
     warnings.warn("MULTIMODAL_MODEL_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is missing in api.txt.", UserWarning)
if EMBEDDING_MODEL_PROVIDER == 'openrouter' and not OPENROUTER_API_KEY:
     warnings.warn("EMBEDDING_MODEL_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is missing in api.txt.", UserWarning)

if TEXT_MODEL_PROVIDER == 'google' and not GOOGLE_API_KEY:
    warnings.warn("TEXT_MODEL_PROVIDER is 'google' but GOOGLE_API_KEY is missing in api.txt.", UserWarning)
if MULTIMODAL_MODEL_PROVIDER == 'google' and not GOOGLE_API_KEY:
     warnings.warn("MULTIMODAL_MODEL_PROVIDER is 'google' but GOOGLE_API_KEY is missing in api.txt.", UserWarning)
if EMBEDDING_MODEL_PROVIDER == 'google' and not GOOGLE_API_KEY:
     warnings.warn("EMBEDDING_MODEL_PROVIDER is 'google' but GOOGLE_API_KEY is missing in api.txt.", UserWarning)

# --- Other Grading Parameters (remain the same) ---
SIMILARITY_THRESHOLD = 0.70
SCORE_COMBINATION_METHOD = "average"
DEFAULT_MAX_SCORE_PER_QUESTION = 10
SCORE_MAPPING = {
    "correct": 1.0, "mostly correct": 0.8, "partially correct": 0.5,
    "incorrect": 0.0, "unsure": 0.2,
}
LLM_CORRECTNESS_KEYWORDS = ["correct", "accurate", "satisfactory", "matches", "valid", "right"]
LLM_PARTIAL_KEYWORDS = ["partially", "mostly", "some elements", "part correct"]

# --- File Paths (remain the same) ---
DATA_DIR = "data"
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
VECTOR_STORE_INDEX_NAME = "answer_sheet_index"

# --- OCR Configuration ---
# TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows

# Ensure data directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

print("-" * 50)
print("Configuration Loaded:")
print(f"  API Config File: {API_CONFIG_FILE}")
print(f"  OpenRouter Key Loaded: {'Yes' if OPENROUTER_API_KEY else 'No'}")
print(f"  Google Key Loaded: {'Yes' if GOOGLE_API_KEY else 'No'}")
print(f"  Text Model: {TEXT_MODEL_PROVIDER} / {TEXT_MODEL_NAME}")
print(f"  Multimodal Model: {MULTIMODAL_MODEL_PROVIDER} / {MULTIMODAL_MODEL_NAME}")
print(f"  Embedding Model: {EMBEDDING_MODEL_PROVIDER} / {EMBEDDING_MODEL_NAME}")
print("-" * 50)