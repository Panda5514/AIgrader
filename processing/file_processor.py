# This needs pdf2image (and its dependency poppler) installed if you want PDF support. Also needs Pillow.

# Python

# processing/file_processor.py

import os
from typing import List, Tuple, Union, Optional, Dict
from PIL import Image
import io
from .ocr_handler import extract_text_from_image, extract_text_from_image_bytes
from utils import config

# Attempt to import pdf2image, handle gracefully if not installed/configured
try:
    from pdf2image import convert_from_bytes, convert_from_path
    PDF2IMAGE_INSTALLED = True
except ImportError:
    print("Warning: pdf2image not installed or Poppler not found. PDF processing will be disabled.")
    print("Install pdf2image: pip install pdf2image")
    print("Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases/ (Windows) or via package manager (Linux/macOS)")
    PDF2IMAGE_INSTALLED = False
except Exception as e: # Catch potential Poppler errors during import time on some systems
     print(f"Warning: Error importing pdf2image (check Poppler installation): {e}")
     PDF2IMAGE_INSTALLED = False


def process_uploaded_file(
    uploaded_file,
    save_path: Optional[str] = None
) -> Tuple[str, List[Image.Image], Optional[str]]:
    """
    Processes an uploaded file (image or PDF).
    Extracts text using OCR and returns PIL Image objects.

    Args:
        uploaded_file: File object from Streamlit upload (has `name`, `type`, `read()` methods).
        save_path: Optional path to save the uploaded file temporarily.

    Returns:
        A tuple containing:
        - extracted_text (str): The combined text extracted from the file.
        - image_objects (List[Image.Image]): A list of PIL Image objects
          (one for images, one per page for PDFs).
        - error_message (Optional[str]): An error message if processing failed.
    """
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue() # Use getvalue() for Streamlit UploadedFile
    file_type = uploaded_file.type # e.g., 'application/pdf', 'image/png'

    print(f"Processing file: {filename}, Type: {file_type}")

    extracted_text = ""
    image_objects = []
    error_message = None

    if save_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            print(f"File saved temporarily to: {save_path}")
        except Exception as e:
            error_message = f"Error saving file temporarily: {e}"
            print(error_message)
            # Continue processing from bytes if saving failed

    try:
        if file_type in ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"]:
            try:
                img = Image.open(io.BytesIO(file_bytes))
                # Ensure image is in RGB format for consistency (some models prefer it)
                if img.mode != 'RGB':
                     img = img.convert('RGB')
                image_objects.append(img)
                extracted_text = extract_text_from_image(img) # Use the object directly
            except Exception as e:
                error_message = f"Error processing image file '{filename}': {e}"
                print(error_message)

        elif file_type == "application/pdf":
            if not PDF2IMAGE_INSTALLED:
                error_message = "PDF processing requires pdf2image and Poppler. Please install them."
                print(error_message)
                return "", [], error_message # Cannot proceed with PDF

            try:
                # Convert PDF bytes to list of PIL images
                print("Converting PDF to images...")
                # Use dpi=300 for potentially better OCR quality, adjust as needed
                pdf_images = convert_from_bytes(file_bytes, dpi=300)
                print(f"PDF converted to {len(pdf_images)} page images.")

                all_page_texts = []
                for i, page_image in enumerate(pdf_images):
                     # Ensure image is in RGB format
                     if page_image.mode != 'RGB':
                         page_image = page_image.convert('RGB')
                     image_objects.append(page_image)
                     print(f"Running OCR on page {i+1}...")
                     page_text = extract_text_from_image(page_image)
                     all_page_texts.append(page_text)
                     # Optional: Add page breaks or markers
                     all_page_texts.append(f"\n--- Page {i+1} End ---\n")

                extracted_text = "".join(all_page_texts)

            except Exception as e:
                error_message = f"Error processing PDF file '{filename}': {e}"
                print(error_message)
                # Attempt OCR on any images successfully converted before the error
                if image_objects and not extracted_text:
                    print("Attempting OCR on successfully converted pages before error...")
                    all_page_texts = []
                    for i, img in enumerate(image_objects):
                         try:
                             page_text = extract_text_from_image(img)
                             all_page_texts.append(page_text)
                             all_page_texts.append(f"\n--- Page {i+1} End ---\n")
                         except Exception as ocr_e:
                             print(f"OCR failed for page {i+1} after PDF error: {ocr_e}")
                             all_page_texts.append(f"\n--- OCR Error on Page {i+1} ---\n")
                    extracted_text = "".join(all_page_texts)


        else:
            error_message = f"Unsupported file type: {file_type} for file '{filename}'"
            print(error_message)

    except pytesseract.TesseractNotFoundError:
        # Error is handled and printed within ocr_handler, but catch again
        error_message = "Tesseract OCR engine not found. Please install and configure it."
        print(error_message)
        # Return whatever might have been processed before the error if applicable
        return extracted_text, image_objects, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred processing file '{filename}': {e}"
        print(error_message)

    # Clean up temporary file if it was saved
    # if save_path and os.path.exists(save_path):
    #     try:
    #         os.remove(save_path)
    #         print(f"Removed temporary file: {save_path}")
    #     except Exception as e:
    #         print(f"Warning: Could not remove temporary file {save_path}: {e}")

    print(f"Finished processing {filename}. Extracted text length: {len(extracted_text)}. Images generated: {len(image_objects)}.")
    return extracted_text, image_objects, error_message

if __name__ == '__main__':
    # Example Usage requires creating dummy files or using actual test files
    print("\n--- Testing File Processor ---")
    print("Note: This test requires dummy files (e.g., 'test.png', 'test.pdf')")
    print("And necessary libraries (Pillow, pdf2image+Poppler, pytesseract+Tesseract) installed.")

    # Mock Streamlit UploadedFile class for testing
    class MockUploadedFile:
        def __init__(self, name, type, file_path):
            self.name = name
            self.type = type
            self._file_path = file_path

        def getvalue(self):
            try:
                with open(self._file_path, "rb") as f:
                    return f.read()
            except FileNotFoundError:
                print(f"Test file not found: {self._file_path}")
                return b"" # Return empty bytes if file missing

    # --- Test with an image file ---
    image_file_path = "test.png" # Make sure this exists
    if os.path.exists(image_file_path):
        print(f"\n--- Testing with Image: {image_file_path} ---")
        mock_image_file = MockUploadedFile("test.png", "image/png", image_file_path)
        img_text, img_objs, img_err = process_uploaded_file(mock_image_file)
        if img_err:
            print(f"Image Processing Error: {img_err}")
        else:
            print(f"Image Text (first 50 chars): {img_text[:50]}...")
            print(f"Number of Image Objects: {len(img_objs)}")
    else:
        print(f"\nSkipping image test: '{image_file_path}' not found.")

    # --- Test with a PDF file ---
    pdf_file_path = "test.pdf" # Make sure this exists
    if os.path.exists(pdf_file_path):
        print(f"\n--- Testing with PDF: {pdf_file_path} ---")
        mock_pdf_file = MockUploadedFile("test.pdf", "application/pdf", pdf_file_path)
        pdf_text, pdf_objs, pdf_err = process_uploaded_file(mock_pdf_file)
        if pdf_err:
            print(f"PDF Processing Error: {pdf_err}")
        else:
            print(f"PDF Text (first 100 chars): {pdf_text[:100]}...")
            print(f"Number of Page Image Objects: {len(pdf_objs)}")
    else:
        print(f"\nSkipping PDF test: '{pdf_file_path}' not found.")

    # --- Test with an unsupported file type ---
    print("\n--- Testing with Unsupported File Type ---")
    mock_txt_file = MockUploadedFile("test.txt", "text/plain", "dummy_path.txt") # path doesn't need to exist for this test
    txt_text, txt_objs, txt_err = process_uploaded_file(mock_txt_file)
    if txt_err:
        print(f"Unsupported File Error (Expected): {txt_err}")
    else:
        print("Error: Unsupported file type test failed.")