# analysis/llm_interface.py

import base64
import io
import json
import warnings
from typing import List, Dict, Any, Optional, Union

from PIL import Image

# Import Langchain components for different providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI # Use this for OpenRouter (OpenAI compatible)
from langchain_core.messages import HumanMessage, SystemMessage # Standard message types
from langchain_core.exceptions import OutputParserException

# Import project config and prompts
from utils import config
from prompts import prompt_templates

# --- Initialize Models based on Config ---

TEXT_MODEL = None
MULTIMODAL_MODEL = None

# --- Text Model Init ---
try:
    if config.TEXT_MODEL_PROVIDER == 'openrouter':
        if config.OPENROUTER_API_KEY:
            print(f"Initializing OpenRouter Text LLM: {config.TEXT_MODEL_NAME}")
            TEXT_MODEL = ChatOpenAI(
                model=config.TEXT_MODEL_NAME,
                openai_api_key=config.OPENROUTER_API_KEY,
                openai_api_base=config.OPENROUTER_API_BASE,
                # Add headers if needed by OpenRouter specific models (check their docs)
                # model_kwargs={"headers": {"HTTP-Referer": "YOUR_SITE_URL", "X-Title": "YOUR_APP_NAME"}}
                temperature=0.3, # Adjust temperature as needed
                max_tokens=500 # Set reasonable max tokens for grading tasks
            )
            print("OpenRouter Text LLM initialized.")
        else:
            warnings.warn("OpenRouter Text provider selected but API key missing.", UserWarning)
    elif config.TEXT_MODEL_PROVIDER == 'google':
        if config.GOOGLE_API_KEY:
            print(f"Initializing Google Text LLM: {config.TEXT_MODEL_NAME}")
            TEXT_MODEL = ChatGoogleGenerativeAI(
                model=config.TEXT_MODEL_NAME,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.3,
                convert_system_message_to_human=True # Some models work better this way
            )
            print("Google Text LLM initialized.")
        else:
            warnings.warn("Google Text provider selected but API key missing.", UserWarning)
    else:
        warnings.warn(f"Unsupported TEXT_MODEL_PROVIDER: {config.TEXT_MODEL_PROVIDER}", UserWarning)

except Exception as e:
    warnings.warn(f"Error initializing Text LLM: {e}", UserWarning)
    TEXT_MODEL = None

# --- Multimodal Model Init ---
# This is more complex due to differing API structures for multimodal input
try:
    if config.MULTIMODAL_MODEL_PROVIDER == 'openrouter':
        # Assumes using an OpenAI-compatible vision model via OpenRouter (like GPT-4V or Gemini via OR)
        if config.OPENROUTER_API_KEY:
            print(f"Initializing OpenRouter Multimodal LLM: {config.MULTIMODAL_MODEL_NAME}")
            MULTIMODAL_MODEL = ChatOpenAI(
                model=config.MULTIMODAL_MODEL_NAME,
                openai_api_key=config.OPENROUTER_API_KEY,
                openai_api_base=config.OPENROUTER_API_BASE,
                temperature=0.3,
                max_tokens=1000 # Allow more tokens for potentially detailed image analysis
            )
            print("OpenRouter Multimodal LLM initialized (assuming OpenAI vision format).")
        else:
            warnings.warn("OpenRouter Multimodal provider selected but API key missing.", UserWarning)
    elif config.MULTIMODAL_MODEL_PROVIDER == 'google':
        if config.GOOGLE_API_KEY:
            print(f"Initializing Google Multimodal LLM: {config.MULTIMODAL_MODEL_NAME}")
            # Google's native SDK handles multimodal input differently
            # Use ChatGoogleGenerativeAI but construct messages carefully later
            MULTIMODAL_MODEL = ChatGoogleGenerativeAI(
                model=config.MULTIMODAL_MODEL_NAME, # Use the specified vision model name
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            print("Google Multimodal LLM initialized.")
        else:
            warnings.warn("Google Multimodal provider selected but API key missing.", UserWarning)
    else:
        warnings.warn(f"Unsupported MULTIMODAL_MODEL_PROVIDER: {config.MULTIMODAL_MODEL_PROVIDER}", UserWarning)

except Exception as e:
    warnings.warn(f"Error initializing Multimodal LLM: {e}", UserWarning)
    MULTIMODAL_MODEL = None

# --- Helper Function for Safe JSON Parsing ---
def parse_json_output(llm_output: str, default_value: Any = None) -> Any:
    """Safely parses a JSON string from LLM output."""
    try:
        # Clean potential markdown code blocks
        text_content = llm_output.strip()
        if text_content.startswith("```json"):
            text_content = text_content[7:-3].strip()
        elif text_content.startswith("```"):
             text_content = text_content[3:-3].strip()

        return json.loads(text_content)
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse LLM output as JSON:\n{llm_output[:200]}...")
        return default_value
    except Exception as e:
        print(f"Warning: An unexpected error occurred during JSON parsing: {e}")
        return default_value

# --- Helper Function to encode images for OpenAI API ---
def encode_image_to_base64(image: Image.Image, format="JPEG") -> str:
    """Encodes a PIL image to base64 string."""
    buffered = io.BytesIO()
    # Handle PNG transparency if needed, otherwise save as JPEG
    if format == "PNG" and image.mode == "RGBA":
         image.save(buffered, format="PNG")
         mime_type = "image/png"
    else:
        image.convert("RGB").save(buffered, format="JPEG")
        mime_type = "image/jpeg"

    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{img_str}"


# --- Core LLM Interaction Functions (Updated for .invoke and provider logic) ---

def llm_compare_text_answers(question_number: str, expected_answer: str, student_answer: str) -> Dict[str, Union[bool, str]]:
    """Uses the configured text LLM to compare student and expected answers."""
    if not TEXT_MODEL:
        print("LLM Error: Text model not available.")
        return {"is_correct": False, "explanation": "LLM Text model not available."}
    if not student_answer:
        return {"is_correct": False, "explanation": "Student answer is empty."}

    prompt = prompt_templates.COMPARE_TEXT_ANSWERS_PROMPT.format(
        question_number=question_number,
        expected_answer=expected_answer,
        student_answer=student_answer
    )
    messages = [HumanMessage(content=prompt)] # Standard message format
    default_error_response = {"is_correct": False, "explanation": "LLM call failed or returned invalid format."}

    try:
        print(f"Sending text comparison request for Q:{question_number} to {config.TEXT_MODEL_PROVIDER}...")
        response = TEXT_MODEL.invoke(messages)
        response_text = response.content # Access content from AIMessage

        print(f"LLM comparison response received: {response_text[:150]}...")
        parsed_response = parse_json_output(response_text, default_value=default_error_response)

        if isinstance(parsed_response, dict) and "is_correct" in parsed_response and "explanation" in parsed_response:
             if isinstance(parsed_response["is_correct"], str):
                 parsed_response["is_correct"] = parsed_response["is_correct"].lower() == 'true'
             return parsed_response
        else:
            # Fallback: try to return the raw text if JSON parsing failed
             return {"is_correct": False, "explanation": f"Invalid format. Raw response: {response_text}"}

    except Exception as e:
        print(f"Error during LLM text comparison for Q:{question_number} ({config.TEXT_MODEL_PROVIDER}): {e}")
        return default_error_response

def llm_generate_justification( # Signature remains the same
    question_number: str, expected_answer: str, student_answer: str, similarity_score: float,
    llm_assessment: str, assigned_score: Union[int, float], max_score: Union[int, float]
) -> str:
    """Uses the configured text LLM to generate a justification."""
    if not TEXT_MODEL:
        return "LLM Text model not available for justification."

    prompt = prompt_templates.GENERATE_JUSTIFICATION_PROMPT.format(
        question_number=question_number, expected_answer=expected_answer[:500], student_answer=student_answer[:500],
        similarity_score=similarity_score, llm_assessment=llm_assessment, assigned_score=assigned_score, max_score=max_score
    )
    messages = [HumanMessage(content=prompt)]

    try:
        print(f"Sending justification request for Q:{question_number} to {config.TEXT_MODEL_PROVIDER}...")
        response = TEXT_MODEL.invoke(messages)
        justification = response.content
        print("LLM justification response received.")
        return justification.strip()
    except Exception as e:
        print(f"Error during LLM justification generation for Q:{question_number} ({config.TEXT_MODEL_PROVIDER}): {e}")
        return f"Error generating justification: {e}"


def llm_compare_multimodal_answers( # Signature remains the same
    question_number: str, expected_text: str, expected_image: Optional[Image.Image],
    student_text: str, student_image: Optional[Image.Image]
) -> Dict[str, Union[bool, str]]:
    """Uses the configured multimodal LLM to compare answers involving text and images."""
    if not MULTIMODAL_MODEL:
        return {"is_satisfactory": False, "explanation": "LLM Multimodal model not available."}

    prompt_text = prompt_templates.COMPARE_MULTIMODAL_ANSWERS_PROMPT.format(
         question_number=question_number, expected_text=expected_text if expected_text else "N/A",
         student_text=student_text if student_text else "N/A",
    )
    default_error_response = {"is_satisfactory": False, "explanation": "Multimodal LLM call failed or returned invalid format."}

    # --- Construct message based on provider ---
    message_content = []
    message_content.append({"type": "text", "text": prompt_text})

    try:
        # Append images using the appropriate format for the provider
        if config.MULTIMODAL_MODEL_PROVIDER == 'openrouter':
            # Assumes OpenAI-compatible API structure for vision
            if expected_image:
                message_content.append({"type": "text", "text": "\nExpected Image:"})
                message_content.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(expected_image)}})
            if student_image:
                message_content.append({"type": "text", "text": "\nStudent Image:"})
                message_content.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(student_image)}})
            messages = [HumanMessage(content=message_content)]

        elif config.MULTIMODAL_MODEL_PROVIDER == 'google':
            # Google's ChatGoogleGenerativeAI handles interleaved PIL Images directly in content list
            content_parts = [prompt_text]
            if expected_image:
                 content_parts.append("\nExpected Image:")
                 content_parts.append(expected_image) # Add PIL image directly
            if student_image:
                 content_parts.append("\nStudent Image:")
                 content_parts.append(student_image) # Add PIL image directly
            messages = [HumanMessage(content=content_parts)]

        else: # Should not happen if config validation works
             return {"is_satisfactory": False, "explanation": "Unsupported multimodal provider."}

        # --- Invoke LLM ---
        print(f"Sending multimodal comparison request for Q:{question_number} to {config.MULTIMODAL_MODEL_PROVIDER}...")
        response = MULTIMODAL_MODEL.invoke(messages)
        response_text = response.content
        print(f"LLM multimodal comparison response received: {response_text[:150]}...")

        parsed_response = parse_json_output(response_text, default_value=default_error_response)

        if isinstance(parsed_response, dict) and "is_satisfactory" in parsed_response and "explanation" in parsed_response:
            if isinstance(parsed_response["is_satisfactory"], str):
                parsed_response["is_satisfactory"] = parsed_response["is_satisfactory"].lower() == 'true'
            return parsed_response
        else:
             return {"is_satisfactory": False, "explanation": f"Invalid format. Raw response: {response_text}"}

    except Exception as e:
        print(f"Error during LLM multimodal comparison for Q:{question_number} ({config.MULTIMODAL_MODEL_PROVIDER}): {e}")
        # Check for specific API errors if possible (content filtering, etc.)
        return default_error_response


def llm_extract_qna_from_text(text_chunk: str) -> List[Dict[str, str]]:
    """Uses the configured text LLM to extract Question-Answer pairs."""
    if not TEXT_MODEL:
        print("LLM Error: Text model not available for Q&A extraction.")
        return []
    if not text_chunk:
        return []

    prompt = prompt_templates.EXTRACT_QNA_PROMPT.format(answer_sheet_chunk=text_chunk)
    messages = [HumanMessage(content=prompt)]
    default_error_response = []

    try:
        print(f"Sending Q&A extraction request to {config.TEXT_MODEL_PROVIDER}...")
        response = TEXT_MODEL.invoke(messages)
        response_text = response.content
        print("LLM Q&A extraction response received.")

        parsed_response = parse_json_output(response_text, default_value=default_error_response)

        if isinstance(parsed_response, list):
            # Basic validation remains the same
            validated_list = []
            for item in parsed_response:
                if isinstance(item, dict) and 'question_number' in item and 'answer_text' in item:
                    validated_list.append({
                        'question_number': str(item['question_number']),
                        'answer_text': str(item['answer_text'])
                        })
                else:
                    print(f"Warning: Skipping invalid item in Q&A extraction output: {item}")
            return validated_list
        else:
            print(f"Warning: LLM Q&A extraction output was not a list: {response_text}")
            return default_error_response

    except Exception as e:
        print(f"Error during LLM Q&A extraction ({config.TEXT_MODEL_PROVIDER}): {e}")
        return default_error_response


# --- Optional: Image Summarization (Update needed for provider logic if used) ---
# This wasn't directly used in the grading flow but kept for reference.
# Needs adaptation similar to multimodal comparison if you intend to use it.
def llm_summarize_image(image: Image.Image) -> str:
    if not MULTIMODAL_MODEL: return "LLM Multimodal model not available."
    if not isinstance(image, Image.Image): return "Invalid input: Not a PIL Image."

    prompt_text = prompt_templates.SUMMARIZE_IMAGE_PROMPT
    message_content = []

    try:
         if config.MULTIMODAL_MODEL_PROVIDER == 'openrouter':
              message_content.append({"type": "text", "text": prompt_text})
              message_content.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(image)}})
              messages = [HumanMessage(content=message_content)]
         elif config.MULTIMODAL_MODEL_PROVIDER == 'google':
              messages = [HumanMessage(content=[prompt_text, image])]
         else:
              return "Unsupported multimodal provider for summarization."

         print(f"Sending image summarization request to {config.MULTIMODAL_MODEL_PROVIDER}...")
         response = MULTIMODAL_MODEL.invoke(messages)
         summary = response.content
         print("LLM image summary received.")
         return summary.strip()

    except Exception as e:
         print(f"Error during LLM image summarization ({config.MULTIMODAL_MODEL_PROVIDER}): {e}")
         return f"Error generating image summary: {e}"


# Example Usage (Conceptual - requires api.txt setup)
if __name__ == '__main__':
    print("\n--- Testing LLM Interface (with Provider Logic) ---")
    # Ensure api.txt is configured correctly.

    if TEXT_MODEL:
        print("\nTesting Text Comparison...")
        comparison_result = llm_compare_text_answers("DemoQ1", "Capital is Paris.", "Paris is the capital.")
        print(f"Comparison Result: {comparison_result}")
    else:
        print("\nText Model not available, skipping text tests.")

    if MULTIMODAL_MODEL:
         print("\nTesting Multimodal Comparison (Conceptual - requires image)...")
         test_img_path = "test.png"
         if os.path.exists(test_img_path):
             try:
                 img = Image.open(test_img_path).convert('RGB')
                 multi_result = llm_compare_multimodal_answers(
                     "DemoIMG", "Text context", img, "Student text", img
                 )
                 print(f"Multimodal Result: {multi_result}")
             except Exception as e:
                 print(f"Could not test multimodal comparison: {e}")
         else:
             print(f"Skipping multimodal test: '{test_img_path}' not found.")
    else:
        print("\nMultimodal Model not available, skipping multimodal tests.")

    if TEXT_MODEL:
        print("\nTesting Q&A Extraction...")
        sample_sheet_text = "Q1: Capital? A: Paris.\nQ2: 2+2? A: 4."
        extracted_qna = llm_extract_qna_from_text(sample_sheet_text)
        print(f"Extracted Q&A: {extracted_qna}")