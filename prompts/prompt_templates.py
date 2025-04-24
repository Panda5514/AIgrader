# prompts/prompt_templates.py

# --- Q&A Extraction ---
EXTRACT_QNA_PROMPT = """
Analyze the following text segment from a teacher's answer sheet. Identify and extract all distinct question/answer pairs.
Format the output as a list of dictionaries, where each dictionary has 'question_number' (str) and 'answer_text' (str) keys.
Be precise with question numbers (e.g., '1a', '2.b', 'III'). If an answer spans multiple lines, combine them.
If a question number or answer is unclear, use 'unknown'.

Text Segment:
---
{answer_sheet_chunk}
---

Extracted Q&A Pairs (Strict JSON format: [{"question_number": "...", "answer_text": "..."}, ...]):
""" # Added Strict JSON emphasis

# --- Answer Comparison (Text-based) ---
# Aim for clearer boolean and explanation consistent with SCORE_MAPPING/interpretation logic
COMPARE_TEXT_ANSWERS_PROMPT = """
You are an AI grading assistant. Compare the student's answer to the expected answer for question '{question_number}'.

Expected Answer:
---
{expected_answer}
---

Student's Answer:
---
{student_answer}
---

Assessment Task:

Output Format (Strict JSON):
{{
  "is_correct": boolean, // True if substantially correct, False otherwise.
  "explanation": "Your reasoning here (e.g., 'Correct.', 'Incorrect, missed mentioning X.', 'Partially correct, concept right but calculation error.')"
}}
""" # Added Strict JSON emphasis and clearer explanation guidance

# --- Score Justification ---
# Ensure it uses the inputs from score_calculator correctly
GENERATE_JUSTIFICATION_PROMPT = """
You are an AI grading assistant generating feedback for a student.
Question Number: {question_number}
Expected Answer Snippet: {expected_answer}
Student's Answer Snippet: {student_answer}
Similarity Score (Text): {similarity_score:.2f} (0.0-1.0, -1.0 if N/A)
Assessment Category: {llm_assessment} # e.g., CORRECT, PARTIALLY CORRECT, INCORRECT
Assigned Score: {assigned_score:.1f} out of {max_score:.1f}

Task: Write constructive feedback for the student explaining *why* this score was given.
- If the assessment is CORRECT, briefly confirm the answer meets requirements.
- If PARTIALLY CORRECT or INCORRECT, clearly state what was missing, inaccurate, or could be improved compared to the expected answer. Be specific.
- Avoid overly harsh language. Aim to help the student understand.
- Keep the feedback concise (2-4 sentences).

Feedback:
""" # Updated placeholders to match score_calculator calls, refined task description


# --- Multimodal Comparison ---
# Align output keys with text comparison if possible, use 'is_satisfactory'
COMPARE_MULTIMODAL_ANSWERS_PROMPT = """
You are an AI grading assistant comparing a student's answer (text and/or image) to an expected answer (text and/or image) for question '{question_number}'.

Expected Answer Context:
Text: {expected_text}
[Image Placeholder for Expected Answer: Provided if available]

Student's Answer Context:
Text: {student_text}
[Image Placeholder for Student Answer: Provided if available]

Assessment Task:

Output Format (Strict JSON):
{{
  "is_satisfactory": boolean, // True if the student's submission is deemed satisfactory, False otherwise.
  "explanation": "Your reasoning considering text and/or image content (e.g., 'Satisfactory, diagram correct.', 'Unsatisfactory, image missing key labels.', 'Partially satisfactory, text okay but image incorrect.')"
}}
""" # Added Strict JSON, clarified satisfactory, aligned keys somewhat


# --- Image Summarization ---
SUMMARIZE_IMAGE_PROMPT = """
Analyze the provided image, which is likely part of an answer to an academic question.
Describe the key visual elements, labels, steps, or concepts shown in the image relevant to assessing its correctness or completeness.
Focus on factual content. Output should be a concise text summary.

Image: [Image Placeholder]

Summary:
""" # No changes needed

# Example of accessing a prompt
if __name__ == '__main__':
    print("--- Example Prompts (Review) ---")
    print("\nCOMPARE_TEXT_ANSWERS_PROMPT (Formatted):")
    print(COMPARE_TEXT_ANSWERS_PROMPT.format(
        question_number="Q1",
        expected_answer="The Earth revolves around the Sun due to gravity.",
        student_answer="The sun is orbited by the earth because of gravity."
    ))

    print("\nGENERATE_JUSTIFICATION_PROMPT (Formatted):")
    print(GENERATE_JUSTIFICATION_PROMPT.format(
        question_number="Q2",
        expected_answer="2 * (x + 3) = 14 -> x + 3 = 7 -> x = 4",
        student_answer="2x + 6 = 14 -> 2x = 8 -> x = 3",
        similarity_score=0.65,
        llm_assessment="PARTIALLY CORRECT", # Use assessment category
        assigned_score=5.0,
        max_score=10.0
    ))

    print("\nCOMPARE_MULTIMODAL_ANSWERS_PROMPT (Formatted):")
    print(COMPARE_MULTIMODAL_ANSWERS_PROMPT.format(
        question_number="BioQ3",
        expected_text="Diagram of a plant cell with labels for nucleus, cell wall, chloroplasts.",
        student_text="My plant cell diagram:"
        # Image placeholders would be handled by the API call structure
    ))