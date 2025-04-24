# grading/score_calculator.py

import os
from typing import Dict, Any, Optional, List, Union
from PIL import Image

from analysis.text_analyzer import calculate_cosine_similarity
from analysis.llm_interface import (
    llm_compare_text_answers,
    llm_compare_multimodal_answers,
    llm_generate_justification
)
from utils import config

def interpret_llm_assessment(explanation: str, is_correct_flag: Optional[bool] = None) -> str:
    """
    Interprets the LLM's assessment explanation (and optional flag)
    into a simple category (correct, partially correct, incorrect).
    """
    explanation_lower = explanation.lower() if explanation else ""

    if isinstance(is_correct_flag, bool):
        # Prioritize explicit boolean flag if available and seems reliable
        if is_correct_flag:
            # Check if explanation suggests partial correctness despite flag
            if any(keyword in explanation_lower for keyword in config.LLM_PARTIAL_KEYWORDS):
                 return "partially correct"
            return "correct"
        else:
            # Check if explanation suggests partial correctness despite flag being False
            if any(keyword in explanation_lower for keyword in config.LLM_PARTIAL_KEYWORDS):
                 return "partially correct" # Might indicate minor error but some credit
            return "incorrect"

    # Fallback: analyze explanation text if boolean flag is missing/unreliable
    if any(keyword in explanation_lower for keyword in config.LLM_CORRECTNESS_KEYWORDS):
         # Check for negation or partial qualifiers
         if "not correct" in explanation_lower or "incorrect" in explanation_lower:
              return "incorrect"
         if any(keyword in explanation_lower for keyword in config.LLM_PARTIAL_KEYWORDS):
              return "partially correct"
         return "correct"
    elif any(keyword in explanation_lower for keyword in config.LLM_PARTIAL_KEYWORDS):
        return "partially correct"
    elif "incorrect" in explanation_lower or "wrong" in explanation_lower or "missing" in explanation_lower:
        return "incorrect"
    else:
        # Default assumption if keywords are unhelpful
        print(f"Warning: Could not determine assessment category from explanation: '{explanation[:100]}...' Assuming partially correct/unsure.")
        return "partially correct" # Or 'unsure' if defined in SCORE_MAPPING

def calculate_final_score(
    interpreted_assessment: str, # 'correct', 'partially correct', 'incorrect'
    similarity_score: Optional[float] = None,
    max_score: int = config.DEFAULT_MAX_SCORE_PER_QUESTION
    ) -> float:
    """
    Calculates the numerical score based on the interpreted assessment and similarity.
    """
    score_multiplier = config.SCORE_MAPPING.get(interpreted_assessment, 0.0) # Default to 0 if assessment category unknown

    # Optional refinement: Adjust score based on similarity, especially for partial cases
    # Example: If 'partially correct', maybe scale score between 0.3*max and 0.7*max based on similarity?
    # if interpreted_assessment == "partially correct" and similarity_score is not None:
    #     # Simple linear scaling within a partial range based on similarity
    #     min_partial_multiplier = config.SCORE_MAPPING.get("incorrect", 0.0) # e.g., 0.0
    #     max_partial_multiplier = config.SCORE_MAPPING.get("mostly correct", 0.8) # e.g., 0.8
    #     # Scale similarity (0-1) to the partial multiplier range
    #     scaled_multiplier = min_partial_multiplier + (max_partial_multiplier - min_partial_multiplier) * similarity_score
    #     # Take the average or max with the category-based multiplier? Let's use the category one for simplicity now.
    #     pass # Keep score_multiplier from category for now, this needs careful design

    # Another idea: Boost score slightly if similarity is very high, even if LLM found minor flaw?
    # if interpreted_assessment != "correct" and similarity_score is not None and similarity_score > 0.9:
    #    score_multiplier = max(score_multiplier, 0.8) # Ensure at least 80% if highly similar? Risky.

    final_score = score_multiplier * max_score
    # Ensure score is within bounds [0, max_score]
    final_score = max(0.0, min(float(max_score), final_score))

    return round(final_score, 1) # Round to one decimal place

def grade_single_answer(
    question_id: str,
    student_text: Optional[str],
    student_images: Optional[List[Image.Image]],
    teacher_text: Optional[str],
    teacher_images: Optional[List[Image.Image]],
    max_score: int = config.DEFAULT_MAX_SCORE_PER_QUESTION
) -> Dict[str, Any]:
    """
    Grades a single student answer against the teacher's answer using similarity and LLM(s).

    Args:
        question_id: Identifier for the question (e.g., "Q1", "2b").
        student_text: Extracted text of the student's answer.
        student_images: List of PIL Images for the student's answer.
        teacher_text: Text of the teacher's expected answer.
        teacher_images: List of PIL Images for the teacher's expected answer.
        max_score: Maximum possible score for this question.

    Returns:
        A dictionary containing:
        - 'question_id': str
        - 'assigned_score': float
        - 'max_score': int
        - 'justification': str
        - 'similarity_score': float (or None)
        - 'llm_text_assessment': dict (or None)
        - 'llm_multi_assessment': dict (or None)
        - 'error': str (if any critical error occurred)
    """
    print(f"\n--- Grading Question: {question_id} ---")
    result = {
        'question_id': question_id,
        'assigned_score': 0.0,
        'max_score': max_score,
        'justification': "Grading failed.",
        'similarity_score': None,
        'llm_text_assessment': None,
        'llm_multi_assessment': None,
        'error': None
    }

    # Determine answer type (text, image, multimodal)
    has_student_text = bool(student_text and student_text.strip())
    has_student_image = bool(student_images)
    has_teacher_text = bool(teacher_text and teacher_text.strip())
    has_teacher_image = bool(teacher_images)

    is_text_only = has_student_text and has_teacher_text and not (has_student_image or has_teacher_image)
    is_multimodal = (has_student_text or has_student_image) and (has_teacher_text or has_teacher_image) and (has_student_image or has_teacher_image) # Requires at least one image overall
    is_image_only = has_student_image and has_teacher_image and not (has_student_text or has_teacher_text) # Less common?

    final_assessment_category = "incorrect" # Default
    llm_combined_explanation = ""

    # --- Step 1: Text Similarity (if applicable) ---
    if has_student_text and has_teacher_text:
        try:
            sim_score = calculate_cosine_similarity(student_text, teacher_text)
            result['similarity_score'] = sim_score
            print(f"  Similarity Score: {sim_score:.4f}")
            if sim_score >= config.SIMILARITY_THRESHOLD:
                print(f"  Similarity above threshold ({config.SIMILARITY_THRESHOLD}).")
        except Exception as e:
            print(f"  Error calculating similarity: {e}")
            result['error'] = f"Similarity calculation failed: {e}"
            # Continue grading if possible, but similarity won't be reliable

    # --- Step 2: LLM Text Comparison (if applicable) ---
    if has_student_text and has_teacher_text:
        try:
            llm_text_eval = llm_compare_text_answers(question_id, teacher_text, student_text)
            result['llm_text_assessment'] = llm_text_eval
            print(f"  LLM Text Assessment: {llm_text_eval}")
            llm_combined_explanation += f"Text Analysis: {llm_text_eval.get('explanation', 'N/A')}\n"
        except Exception as e:
            print(f"  Error during LLM text comparison: {e}")
            # Don't assign error to result['error'] yet, multimodal might still work
            llm_combined_explanation += "Text Analysis: Error during LLM call.\n"


    # --- Step 3: LLM Multimodal Comparison (if applicable) ---
    # Use multimodal if *either* answer has an image associated with it
    if has_student_image or has_teacher_image:
        # We need *some* text context if possible, even if it's just the question ID or basic description
        effective_student_text = student_text if has_student_text else f"Visual answer for {question_id}"
        effective_teacher_text = teacher_text if has_teacher_text else f"Expected visual for {question_id}"
        # Get the first image if multiple exist (LLM might handle multiple, but use first for simplicity now)
        student_img_obj = student_images[0] if has_student_image else None
        teacher_img_obj = teacher_images[0] if has_teacher_image else None

        try:
            llm_multi_eval = llm_compare_multimodal_answers(
                question_number=question_id,
                expected_text=effective_teacher_text,
                expected_image=teacher_img_obj,
                student_text=effective_student_text,
                student_image=student_img_obj
            )
            result['llm_multi_assessment'] = llm_multi_eval
            print(f"  LLM Multimodal Assessment: {llm_multi_eval}")
            llm_combined_explanation += f"Multimodal Analysis: {llm_multi_eval.get('explanation', 'N/A')}\n"
        except Exception as e:
            print(f"  Error during LLM multimodal comparison: {e}")
            llm_combined_explanation += "Multimodal Analysis: Error during LLM call.\n"

    # --- Step 4: Determine Final Assessment Category ---
    text_assessment = result.get('llm_text_assessment')
    multi_assessment = result.get('llm_multi_assessment')

    # How to combine results? Use config.SCORE_COMBINATION_METHOD
    # 'average': Average scores derived from text and multi assessments.
    # 'strict': Requires both (if applicable) to be good.
    # 'weighted': Define weights (e.g., 0.4*text + 0.6*multi).

    text_category = "incorrect"
    multi_category = "incorrect"
    final_category_explanation = "" # For debugging/logging the decision logic

    if text_assessment:
        text_category = interpret_llm_assessment(
            text_assessment.get('explanation'),
            text_assessment.get('is_correct')
        )
        final_category_explanation += f"Text category: {text_category}. "
    if multi_assessment:
        multi_category = interpret_llm_assessment(
            multi_assessment.get('explanation'),
            multi_assessment.get('is_satisfactory') # Use 'is_satisfactory' key from multimodal prompt
        )
        final_category_explanation += f"Multi category: {multi_category}. "

    if is_multimodal or is_image_only:
        if text_assessment and multi_assessment: # Both text and images involved and assessed
            if config.SCORE_COMBINATION_METHOD == 'strict':
                # If either is incorrect, final is incorrect. If either is partial, final is partial.
                if text_category == "incorrect" or multi_category == "incorrect":
                    final_assessment_category = "incorrect"
                elif text_category == "partially correct" or multi_category == "partially correct":
                    final_assessment_category = "partially correct"
                else:
                    final_assessment_category = "correct"
                final_category_explanation += f"Combined (Strict): {final_assessment_category}. "
            # Default to average-like logic otherwise (simplification: take the 'worse' assessment)
            elif text_category == "incorrect" or multi_category == "incorrect":
                 final_assessment_category = "incorrect"
            elif text_category == "partially correct" or multi_category == "partially correct":
                 final_assessment_category = "partially correct"
            else: # Both must be correct
                 final_assessment_category = "correct"
            final_category_explanation += f"Combined (Average/Worse): {final_assessment_category}. "

        elif multi_assessment: # Only multimodal assessment was possible/successful
            final_assessment_category = multi_category
            final_category_explanation += f"Combined (Multi only): {final_assessment_category}. "
        elif text_assessment: # Only text assessment possible (e.g., multimodal failed)
            final_assessment_category = text_category
            final_category_explanation += f"Combined (Text only fallback): {final_assessment_category}. "
        else: # Both failed
             final_assessment_category = "incorrect" # Or maybe handle error explicitly?
             final_category_explanation += "Combined (Both failed): incorrect. "
             result['error'] = result['error'] or "Both text and multimodal LLM assessments failed."

    elif is_text_only:
        if text_assessment:
            final_assessment_category = text_category
            final_category_explanation += f"Combined (Text Only): {final_assessment_category}. "
        else:
             final_assessment_category = "incorrect"
             final_category_explanation += "Combined (Text assessment failed): incorrect. "
             result['error'] = result['error'] or "LLM text assessment failed."
    else: # No text or image, or some other edge case
        print("  Warning: Could not determine assessment type or required components missing.")
        result['error'] = result['error'] or "Cannot grade - missing student or teacher answer components."
        final_assessment_category = "incorrect"
        llm_combined_explanation = "Cannot grade - missing required answer components."

    print(f"  Decision Logic: {final_category_explanation}")
    print(f"  Final Assessment Category: {final_assessment_category}")


    # --- Step 5: Calculate Numerical Score ---
    result['assigned_score'] = calculate_final_score(
        final_assessment_category,
        result['similarity_score'],
        max_score
    )
    print(f"  Assigned Score: {result['assigned_score']} / {max_score}")

    # --- Step 6: Generate Justification ---
    # Use the combined explanation from LLM calls as primary input for the justification LLM
    # Provide key context elements to the justification prompt
    try:
        justification_text = llm_generate_justification(
            question_number=question_id,
            # Provide snippets or full answers depending on length/token limits
            expected_answer=teacher_text[:300] if teacher_text else "N/A (Visual expected)",
            student_answer=student_text[:300] if student_text else "N/A (Visual provided)",
            similarity_score=result['similarity_score'] if result['similarity_score'] is not None else -1.0, # Use -1 if N/A
            llm_assessment=final_assessment_category.upper(), # Pass the final decision
            assigned_score=result['assigned_score'],
            max_score=max_score
        )
        # Prepend the LLM's reasoning if helpful and not redundant
        # justification_text = f"LLM Reasoning: {llm_combined_explanation.strip()}\n\nFeedback:\n{justification_text}"
        result['justification'] = justification_text
        print(f"  Generated Justification (Snippet): {justification_text[:100]}...")
    except Exception as e:
        print(f"  Error generating justification: {e}")
        result['justification'] = f"Score: {result['assigned_score']}/{max_score}. Failed to generate detailed feedback. LLM Reasoning hints: {llm_combined_explanation.strip()}"
        # Update error if justification failed significantly
        result['error'] = result['error'] or f"Justification generation failed: {e}"


    return result

# Example Usage (Conceptual - requires actual data structures)
if __name__ == '__main__':
    print("\n--- Testing Score Calculator ---")
    # This requires mocked LLM responses or actual LLM calls (if API key is set)
    # and potentially dummy image files.

    # Mock data (replace with actual processing results)
    q_id = "SciQ1"
    s_text = "The powerhouse of the cell is the mitochondria."
    t_text = "Mitochondria are known as the powerhouses of the cell."
    s_img = None # Assume text only for first test
    t_img = None

    # Make sure necessary models are loaded in llm_interface
    # Requires API Key to be set correctly in config / .env
    if config.API_KEY: # Only run if API key seems configured
        print(f"\n--- Grading Text-Only Answer ({q_id}) ---")
        grade_result_text = grade_single_answer(q_id, s_text, s_img, t_text, t_img)
        print("\nText-Only Grade Result:")
        import json
        print(json.dumps(grade_result_text, indent=2))

        # --- Test Multimodal ---
        q_id_multi = "DiagramQ1"
        s_text_multi = "See diagram."
        t_text_multi = "The diagram should show a simple circuit with battery, switch, bulb."
        # Load dummy images if available
        s_img_multi = None
        t_img_multi = None
        test_img_path = "test.png" # Assumes test.png exists
        if os.path.exists(test_img_path):
             try:
                 img = Image.open(test_img_path).convert('RGB')
                 s_img_multi = [img] # Use same image for student/teacher for demo
                 t_img_multi = [img]
             except Exception as e:
                 print(f"Could not load test image {test_img_path}: {e}")

        if s_img_multi and t_img_multi:
             print(f"\n--- Grading Multimodal Answer ({q_id_multi}) ---")
             grade_result_multi = grade_single_answer(q_id_multi, s_text_multi, s_img_multi, t_text_multi, t_img_multi)
             print("\nMultimodal Grade Result:")
             print(json.dumps(grade_result_multi, indent=2))
        else:
            print("\nSkipping multimodal test: Test image not loaded.")

        # --- Test Incorrect ---
        q_id_inc = "MathQ1"
        s_text_inc = "2 + 2 = 5"
        t_text_inc = "2 + 2 = 4"
        print(f"\n--- Grading Incorrect Text Answer ({q_id_inc}) ---")
        grade_result_inc = grade_single_answer(q_id_inc, s_text_inc, None, t_text_inc, None)
        print("\nIncorrect Text Grade Result:")
        print(json.dumps(grade_result_inc, indent=2))

    else:
        print("\nSkipping grading tests as GOOGLE_API_KEY is not configured.")