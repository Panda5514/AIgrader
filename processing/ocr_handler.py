# This requires pytesseract and the Tesseract OCR engine installed on your system. You also need Pillow (PIL).

# Python

# processing/ocr_handler.py

import pytesseract
from PIL import Image
import io
import os
from utils import config # To potentially get Tesseract path

# Optional: If Tesseract is not in your system's PATH, uncomment and set the path
# if hasattr(config, 'TESSERACT_CMD_PATH'):
#     pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD_PATH
# else:
#     # Check common paths or rely on PATH
#     pass # Or add logic to find Tesseract if needed

def extract_text_from_image(image_obj: Image.Image) -> str:
    """
    Extracts text from a PIL Image object using Tesseract OCR.

    Args:
        image_obj: A PIL Image object.

    Returns:
        The extracted text as a string. Returns an empty string if OCR fails.
    """
    if not isinstance(image_obj, Image.Image):
        raise TypeError("Input must be a PIL Image object")

    try:
        # Perform OCR using pytesseract
        text = pytesseract.image_to_string(image_obj)
        print(f"OCR successful for image (snippet): {text[:100]}...") # Log snippet
        return text
    except pytesseract.TesseractNotFoundError:
        print("\nERROR: Tesseract executable not found.")
        print("Please install Tesseract OCR engine: https://github.com/tesseract-ocr/tesseract")
        print("And ensure 'tesseract' command is in your system's PATH or set TESSERACT_CMD_PATH in utils/config.py\n")
        raise # Re-raise the error to stop execution if Tesseract is essential
    except Exception as e:
        print(f"An error occurred during OCR: {e}")
        return "" # Return empty string on other errors

def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    Extracts text from image bytes using Tesseract OCR.

    Args:
        image_bytes: Raw bytes of the image file (e.g., from file upload).

    Returns:
        The extracted text as a string.
    """
    try:
        # Open image bytes using Pillow
        image_obj = Image.open(io.BytesIO(image_bytes))
        return extract_text_from_image(image_obj)
    except Exception as e:
        print(f"Error opening image bytes for OCR: {e}")
        return ""

if __name__ == '__main__':
    # Example usage (requires a test image named 'test.png' in the same directory)
    try:
        # Create a dummy white image for testing if no file exists
        test_image_path = "test.png"
        if not os.path.exists(test_image_path):
             print(f"'{test_image_path}' not found. Create a sample image file for testing.")
             # Example: Create a simple dummy image with Pillow if needed
             # img = Image.new('RGB', (400, 100), color = 'white')
             # from PIL import ImageDraw
             # d = ImageDraw.Draw(img)
             # d.text((10,10), "This is a test text for OCR.", fill=(0,0,0))
             # img.save(test_image_path)
             # print("Created dummy 'test.png'")
             img = Image.open(test_image_path)
        else:
             img = Image.open(test_image_path)


        print("--- Testing OCR Handler ---")
        text_from_obj = extract_text_from_image(img)
        print("\nText extracted from Image Object:")
        print(text_from_obj)

        # To test bytes, read the file
        with open(test_image_path, "rb") as f:
             img_bytes = f.read()
        text_from_bytes = extract_text_from_image_bytes(img_bytes)
        print("\nText extracted from Image Bytes:")
        print(text_from_bytes)

    except ImportError:
        print("Pillow not installed. Run: pip install Pillow")
    except pytesseract.TesseractNotFoundError:
         # Error already printed in function, pass here
         pass
    except FileNotFoundError:
         print(f"Error: Test file '{test_image_path}' not found.")
    except Exception as e:
         print(f"An unexpected error occurred during testing: {e}")