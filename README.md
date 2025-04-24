# AI Grading Assistant

This project provides a Streamlit web application designed to automate parts of the grading process for teachers or teaching assistants. It utilizes Retrieval-Augmented Generation (RAG) with multimodal Large Language Models (LLMs) to compare student answers (text and images) against a teacher's answer key.

## Features

* **File Upload:** Upload teacher's answer sheet and multiple student answer sheets (PDF, PNG, JPG supported).
* **OCR Integration:** Extracts text from uploaded documents using Tesseract OCR via `pytesseract`. Handles PDF conversion to images using `pdf2image` and Poppler.
* **Multimodal RAG:**
    * Builds a vector store (using FAISS and configurable embedding models) from the teacher's answer sheet for efficient retrieval.
    * Uses LLMs to attempt Q&A pair extraction from both teacher and student documents.
    * Retrieves relevant teacher answer sections based on student answers.
* **Automated Grading:**
    * Calculates text similarity (cosine similarity) between student and teacher answers using Sentence Transformers.
    * Uses LLMs (configurable via OpenRouter/Google) for semantic comparison of text answers.
    * Leverages multimodal LLMs (configurable) to compare answers involving images.
    * Calculates a final score based on combined text/multimodal assessments and similarity scores.
    * Generates textual justifications for the assigned score using an LLM.
* **Output Generation:**
    * Creates individual `.txt` feedback files for each student, detailing scores and justifications per question.
    * Generates a summary `.txt` file with total scores for all students.
    * Packages individual feedback files into a downloadable `.zip` archive.
* **Web Interface:** Simple UI built with Streamlit for easy interaction.
* **Configurable Models:** Uses an `api.txt` file to configure API keys and select different LLM/embedding models via OpenRouter or directly from Google.

