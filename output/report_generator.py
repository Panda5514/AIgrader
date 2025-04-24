# output/report_generator.py

import os
from typing import Dict, List, Any

from typing import Optional 
from utils import config, helpers
import datetime

def create_student_feedback_file(
    student_id: str,
    graded_results: List[Dict[str, Any]],
    output_dir: str = config.RESULTS_DIR
    ) -> Optional[str]:
    """
    Creates a text file with detailed feedback for a single student.

    Args:
        student_id: A unique identifier for the student (e.g., sanitized filename).
        graded_results: A list of result dictionaries from grade_single_answer.
        output_dir: The directory to save the feedback file.

    Returns:
        The path to the created file, or None on failure.
    """
    filename = f"{student_id}_feedback_{helpers.get_timestamp_string()}.txt"
    filepath = os.path.join(output_dir, filename)

    total_score = 0.0
    total_max_score = 0.0

    try:
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Feedback Report for: {student_id}\n")
            f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 40 + "\n\n")

            if not graded_results:
                f.write("No questions were graded for this student.\n")
            else:
                for result in graded_results:
                    q_id = result.get('question_id', 'Unknown Q')
                    score = result.get('assigned_score', 0.0)
                    max_s = result.get('max_score', config.DEFAULT_MAX_SCORE_PER_QUESTION)
                    justification = result.get('justification', 'No feedback available.')
                    error = result.get('error')

                    f.write(f"--- Question: {q_id} ---\n")
                    f.write(f"Score: {score:.1f} / {max_s:.1f}\n")
                    f.write("Feedback:\n")
                    f.write(f"{justification}\n")
                    if error:
                        f.write(f"\nNote: A processing error occurred for this question: {error}\n")
                    # Optional: Include similarity score or LLM raw assessment details
                    # sim = result.get('similarity_score')
                    # if sim is not None: f.write(f"(Similarity Score: {sim:.3f})\n")
                    f.write("-" * 30 + "\n\n")

                    total_score += score
                    total_max_score += max_s

                f.write("=" * 40 + "\n")
                f.write("Summary\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Score: {total_score:.1f} / {total_max_score:.1f}\n")
                percentage = (total_score / total_max_score * 100) if total_max_score > 0 else 0
                f.write(f"Percentage: {percentage:.1f}%\n")
                f.write("=" * 40 + "\n")

        print(f"Successfully generated feedback file: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error creating feedback file for {student_id} at {filepath}: {e}")
        return None

def create_summary_file(
    all_student_results: Dict[str, List[Dict[str, Any]]],
    output_dir: str = config.RESULTS_DIR
    ) -> Optional[str]:
    """
    Creates a summary text file listing total scores for all students.

    Args:
        all_student_results: A dictionary where keys are student_ids and values
                             are lists of their graded results.
        output_dir: The directory to save the summary file.

    Returns:
        The path to the created summary file, or None on failure.
    """
    filename = f"summary_scores_{helpers.get_timestamp_string()}.txt"
    filepath = os.path.join(output_dir, filename)

    try:
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Grading Summary Report\n")
            f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'Student ID':<30} | {'Total Score':<15} | {'Max Score':<15} | {'Percentage':<10}\n")
            f.write("-" * 75 + "\n")

            if not all_student_results:
                f.write("No student results available to summarize.\n")
            else:
                sorted_student_ids = sorted(all_student_results.keys())
                for student_id in sorted_student_ids:
                    results = all_student_results[student_id]
                    total_score = sum(r.get('assigned_score', 0.0) for r in results)
                    total_max_score = sum(r.get('max_score', config.DEFAULT_MAX_SCORE_PER_QUESTION) for r in results)
                    percentage = (total_score / total_max_score * 100) if total_max_score > 0 else 0.0

                    f.write(f"{student_id:<30} | {total_score:<15.1f} | {total_max_score:<15.1f} | {percentage:<10.1f}%\n")

            f.write("=" * 75 + "\n")

        print(f"Successfully generated summary file: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error creating summary file at {filepath}: {e}")
        return None




if __name__ == '__main__':
    print("\n--- Testing Report Generator ---")
    # Create dummy results data
    dummy_results_student1 = [
        {'question_id': 'Q1', 'assigned_score': 8.0, 'max_score': 10, 'justification': 'Mostly correct, minor detail missing.', 'error': None},
        {'question_id': 'Q2', 'assigned_score': 5.0, 'max_score': 10, 'justification': 'Partially correct, missed second part.', 'error': None},
        {'question_id': 'Q3', 'assigned_score': 0.0, 'max_score': 5, 'justification': 'Incorrect approach.', 'error': "LLM assessment failed"},
    ]
    dummy_results_student2 = [
        {'question_id': 'Q1', 'assigned_score': 10.0, 'max_score': 10, 'justification': 'Excellent, fully correct.', 'error': None},
        {'question_id': 'Q2', 'assigned_score': 9.0, 'max_score': 10, 'justification': 'Correct concept, slightly unclear explanation.', 'error': None},
        {'question_id': 'Q3', 'assigned_score': 5.0, 'max_score': 5, 'justification': 'Correct.', 'error': None},
    ]
    all_results = {
        "student_alpha": dummy_results_student1,
        "student_beta_long_name": dummy_results_student2,
    }

    # Test individual file creation
    print("\nTesting individual student file creation...")
    feedback_path1 = create_student_feedback_file("student_alpha", dummy_results_student1)
    feedback_path2 = create_student_feedback_file("student_beta_long_name", dummy_results_student2)

    if feedback_path1: print(f"Created: {feedback_path1}")
    if feedback_path2: print(f"Created: {feedback_path2}")

    # Test summary file creation
    print("\nTesting summary file creation...")
    summary_path = create_summary_file(all_results)
    if summary_path:
        print(f"Created: {summary_path}")
        print("--- Summary File Content: ---")
        try:
            with open(summary_path, 'r', encoding='utf-8') as f_sum:
                print(f_sum.read())
        except Exception as e:
            print(f"Error reading summary file: {e}")
        print("--- End Summary File Content ---")

    print("\nCheck the 'data/results' directory for the generated files.")