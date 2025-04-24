# main_app.py

import streamlit as st
import os
from PIL import Image
import time # For simulating progress/sleep

# Import project modules
from utils import config, helpers
from processing import file_processor
from analysis import rag_pipeline, llm_interface # text_analyzer is used by grading
from grading import score_calculator
from output import report_generator, zip_creator

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Grading Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Application State Initialization ---
# Use session state to store data across reruns
if 'teacher_processed' not in st.session_state:
    st.session_state.teacher_processed = False
    st.session_state.teacher_data = None # Store {'text': ..., 'images': [...]}
    st.session_state.teacher_vector_store = None
if 'students_processed' not in st.session_state:
    st.session_state.students_processed = False
    st.session_state.student_files_uploaded = False # Track if files were uploaded in this session
    st.session_state.student_data = {} # Store {student_id: {'text':..., 'images': [...]}}
if 'grading_complete' not in st.session_state:
    st.session_state.grading_complete = False
    st.session_state.grading_results = {} # Store {student_id: [grade_result_dict, ...]}
    st.session_state.summary_file_path = None
    st.session_state.zip_file_path = None


# --- Helper Functions for Streamlit ---

def display_error(message):
    """Displays an error message in Streamlit."""
    st.error(f"❌ Error: {message}")

def display_warning(message):
    """Displays a warning message in Streamlit."""
    st.warning(f"⚠️ Warning: {message}")

def display_success(message):
    """Displays a success message in Streamlit."""
    st.success(f"✅ Success: {message}")

def reset_state_on_upload():
    """Resets processing and grading state when new files are potentially uploaded."""
    st.session_state.teacher_processed = False
    st.session_state.teacher_data = None
    st.session_state.teacher_vector_store = None
    st.session_state.students_processed = False
    # Keep track if student files were uploaded in *this run* to avoid processing empty list later
    # st.session_state.student_files_uploaded = False # Resetting this might be tricky with multiple uploads
    st.session_state.student_data = {}
    st.session_state.grading_complete = False
    st.session_state.grading_results = {}
    st.session_state.summary_file_path = None
    st.session_state.zip_file_path = None
    print("State reset due to potential file upload change.")

# --- Main Application UI ---

st.title("📚 AI Grading Assistant")
st.markdown("Upload the teacher's answer sheet and student answer sheets to automatically grade them.")
st.markdown("---")

# --- Configuration Check ---
api_key_ok = False
if config.TEXT_MODEL_PROVIDER == 'openrouter' and config.OPENROUTER_API_KEY:
    api_key_ok = True
elif config.TEXT_MODEL_PROVIDER == 'google' and config.GOOGLE_API_KEY:
     api_key_ok = True
# Check multimodal and embedding providers similarly if they might differ
if config.MULTIMODAL_MODEL_PROVIDER == 'openrouter' and not config.OPENROUTER_API_KEY:
     api_key_ok = False # If OR is chosen but key missing, it's not ok
if config.MULTIMODAL_MODEL_PROVIDER == 'google' and not config.GOOGLE_API_KEY:
     api_key_ok = False # If Google is chosen but key missing, it's not ok
# Similar checks for EMBEDDING_MODEL_PROVIDER

if not api_key_ok:
    display_error("Required API Key (OpenRouter or Google) based on provider selection in api.txt is missing. Please check 'api.txt'.")
    st.stop() # Stop execution if necessary keys are missing

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("📁 File Uploads")

    # Teacher Answer Sheet Upload
    uploaded_teacher_file = st.file_uploader(
        "1. Upload Teacher's Answer Sheet (PDF/PNG/JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload the master answer key provided by the teacher.",
        key="teacher_uploader",
        # on_change=reset_state_on_upload # Reset state if file changes
    )

    # Student Answer Sheets Upload
    uploaded_student_files = st.file_uploader(
        "2. Upload Student Answer Sheets (PDF/PNG/JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload one or more student answer files.",
        key="student_uploader",
        # on_change=reset_state_on_upload # Reset state if files change
    )

    # --- Control Buttons ---
    st.markdown("---")
    # Disable button until both teacher and at least one student file are uploaded
    process_and_grade_disabled = not (uploaded_teacher_file and uploaded_student_files)
    if process_and_grade_disabled:
         st.info("Please upload both the teacher's sheet and at least one student sheet.")

    process_button = st.button(
        "🚀 Process Files & Start Grading",
        disabled=process_and_grade_disabled,
        type="primary",
        use_container_width=True
    )

    # Add a button to clear state manually if needed
    if st.button("🔄 Clear All & Restart", use_container_width=True):
         reset_state_on_upload()
         # Clear file uploaders too? Might require rerunning script `st.rerun()`
         st.rerun() # Rerun to clear UI elements potentially tied to state

# --- Main Area for Processing and Results ---

# Keep track if processing/grading has started in this run
processing_started = False

if process_button:
    processing_started = True
    st.session_state.grading_complete = False # Ensure results are hidden until grading finishes

    # --- Step 1: Process Teacher's Answer Sheet ---
    if not st.session_state.teacher_processed and uploaded_teacher_file:
        st.markdown("### Processing Teacher's Answer Sheet...")
        progress_bar_teacher = st.progress(0, text="Starting teacher sheet processing...")
        try:
            start_time = time.time()
            # Save temporarily if needed for some processors, though processing bytes is often fine
            # save_path = os.path.join(config.UPLOADS_DIR, helpers.sanitize_filename(uploaded_teacher_file.name))
            text, images, error_msg = file_processor.process_uploaded_file(uploaded_teacher_file)
            progress_bar_teacher.progress(30, text="File processed (OCR/Extraction)...")

            if error_msg:
                display_error(f"Teacher sheet processing failed: {error_msg}")
                st.stop()

            if not text and not images:
                 display_error("Teacher sheet processing resulted in no content.")
                 st.stop()

            st.session_state.teacher_data = {'text': text, 'images': images}
            progress_bar_teacher.progress(50, text="Building/Loading RAG Vector Store...")

            # Build/Load RAG Store
            vector_store = rag_pipeline.build_or_load_answer_sheet_store(text)
            progress_bar_teacher.progress(90, text="Vector Store Ready.")

            if not vector_store:
                display_error("Failed to build or load the vector store for the teacher's answers.")
                st.stop()

            st.session_state.teacher_vector_store = vector_store
            st.session_state.teacher_processed = True
            end_time = time.time()
            progress_bar_teacher.progress(100, text=f"Teacher sheet processed successfully! ({end_time - start_time:.2f}s)")
            display_success("Teacher's answer sheet processed and RAG store initialized.")

        except Exception as e:
            display_error(f"An unexpected error occurred during teacher sheet processing: {e}")
            # Log traceback here if possible
            st.stop()
    elif uploaded_teacher_file is None:
         st.info("Waiting for teacher file upload...") # Should not happen if button logic is right
         st.stop()
    # If already processed, skip but ensure data is loaded (should be in session state)
    elif st.session_state.teacher_processed:
         display_success("Teacher sheet already processed.")


    # --- Step 2: Process Student Answer Sheets ---
    if not st.session_state.students_processed and uploaded_student_files:
        st.markdown("### Processing Student Answer Sheets...")
        st.session_state.student_data = {} # Reset student data for this processing run
        num_students = len(uploaded_student_files)
        progress_bar_students = st.progress(0, text=f"Starting processing for {num_students} student sheets...")
        processing_errors = []

        for i, file in enumerate(uploaded_student_files):
            student_id = helpers.sanitize_filename(os.path.splitext(file.name)[0]) # Use filename base as ID
            progress_text = f"Processing student {i+1}/{num_students}: {file.name}..."
            progress_bar_students.progress((i / num_students), text=progress_text)
            print(progress_text) # Log progress

            try:
                s_text, s_images, s_error = file_processor.process_uploaded_file(file)
                if s_error:
                    display_warning(f"Error processing '{file.name}': {s_error}. Skipping this file.")
                    processing_errors.append(f"{file.name}: {s_error}")
                    continue
                if not s_text and not s_images:
                    display_warning(f"No content extracted from '{file.name}'. Skipping this file.")
                    processing_errors.append(f"{file.name}: No content extracted.")
                    continue

                st.session_state.student_data[student_id] = {'text': s_text, 'images': s_images, 'original_filename': file.name}

            except Exception as e:
                error_msg = f"Unexpected error processing '{file.name}': {e}"
                display_warning(error_msg) # Show specific error but try to continue
                processing_errors.append(error_msg)
                continue # Skip this file

        progress_bar_students.progress(100, text=f"Finished processing student sheets.")
        if not st.session_state.student_data:
             display_error("No student sheets could be processed successfully.")
             st.stop()
        elif processing_errors:
             st.warning("Some student files could not be processed. Check details below:")
             st.json({"processing_errors": processing_errors})

        st.session_state.students_processed = True
        display_success(f"{len(st.session_state.student_data)} student sheets processed.")

    elif not uploaded_student_files:
         st.info("Waiting for student file uploads...") # Should not happen if button logic is right
         st.stop()
    # If already processed, skip but ensure data is loaded
    elif st.session_state.students_processed:
         display_success("Student sheets already processed.")


    # --- Step 3: Perform Grading ---
    # Ensure both teacher and students are processed before grading
    if st.session_state.teacher_processed and st.session_state.students_processed:
        st.markdown("### Grading Answers...")
        st.session_state.grading_results = {} # Reset grading results
        total_students_to_grade = len(st.session_state.student_data)
        if total_students_to_grade == 0:
             display_warning("No processed student data available for grading.")
             st.stop()

        grading_errors = {}
        with st.spinner("🧠 AI is grading... This may take some time..."):
            progress_bar_grading = st.progress(0, text="Starting grading...")

            # --- Q&A Extraction Strategy ---
            # Attempt to extract Q&A pairs from the teacher's sheet first
            teacher_qna_map = {}
            try:
                print("Attempting to extract Q&A pairs from teacher sheet...")
                teacher_full_text = st.session_state.teacher_data['text']
                extracted_teacher_qna = llm_interface.llm_extract_qna_from_text(teacher_full_text)
                if extracted_teacher_qna:
                    teacher_qna_map = {item['question_number']: item['answer_text'] for item in extracted_teacher_qna}
                    print(f"Extracted {len(teacher_qna_map)} Q&A pairs from teacher sheet.")
                else:
                    print("Warning: Could not extract structured Q&A from teacher sheet via LLM.")
                    # Will rely solely on RAG fallback later
            except Exception as e:
                print(f"Error during teacher Q&A extraction: {e}")
                # Proceed, relying on RAG

            # --- Loop Through Students ---
            for i, (student_id, student_content) in enumerate(st.session_state.student_data.items()):
                student_grades = []
                student_errors = []
                progress_text = f"Grading student {i+1}/{total_students_to_grade}: {student_content.get('original_filename', student_id)}..."
                progress_bar_grading.progress(i / total_students_to_grade, text=progress_text)
                print(progress_text)

                student_text = student_content['text']
                student_images = student_content['images'] # List of PIL images

                # --- Extract Q&A from Student Sheet ---
                student_qna_list = []
                try:
                    print(f"Attempting to extract Q&A from student: {student_id}")
                    student_qna_list = llm_interface.llm_extract_qna_from_text(student_text)
                    print(f"Extracted {len(student_qna_list)} Q&A pairs for student {student_id}.")
                    if not student_qna_list:
                         print(f"Warning: No structured Q&A found for student {student_id}. Grading might be incomplete.")
                         student_errors.append("Could not extract structured Q&A pairs from the submission.")
                         # TODO: Alternative? Try grading the whole text as one block? Too unreliable.
                except Exception as e:
                    print(f"Error extracting Q&A for student {student_id}: {e}")
                    student_errors.append(f"LLM Error during Q&A extraction: {e}")
                    # Continue to next student if extraction fails fundamentally? Or try grading without pairs?

                # --- Grade Each Extracted Student Answer ---
                for sqa in student_qna_list:
                    q_num = sqa.get('question_number', 'Unknown')
                    s_text = sqa.get('answer_text', '')
                    print(f"  Grading Q: {q_num} for student {student_id}")

                    if q_num == 'Unknown' or not s_text:
                        print(f"  Skipping Q: {q_num} due to missing number or text.")
                        continue # Skip if Q number or answer text couldn't be extracted

                    # --- Find Corresponding Teacher Answer ---
                    teacher_answer_text = None
                    teacher_answer_images = st.session_state.teacher_data.get('images', []) # Use all teacher images for now

                    # Strategy 1: Use extracted teacher map
                    if q_num in teacher_qna_map:
                        teacher_answer_text = teacher_qna_map[q_num]
                        print(f"  Found teacher answer for {q_num} via direct Q&A map.")
                    else:
                        # Strategy 2: Fallback to RAG using student answer/question number
                        print(f"  Teacher answer for {q_num} not in map, attempting RAG retrieval...")
                        query = f"Question {q_num}: {s_text}" # Use combined query
                        retrieved_docs = rag_pipeline.retrieve_relevant_documents(
                            query=query,
                            vector_store=st.session_state.teacher_vector_store,
                            num_results=1 # Get the single most relevant chunk
                        )
                        if retrieved_docs:
                            # Combine content from retrieved docs (using only the best match here)
                            teacher_answer_text = retrieved_docs[0][0].page_content
                            print(f"  Found teacher answer for {q_num} via RAG (Score: {retrieved_docs[0][1]:.4f}).")
                        else:
                            print(f"  Warning: Could not find relevant teacher answer for {q_num} via RAG either.")
                            student_errors.append(f"Could not find matching teacher answer for Question '{q_num}'.")
                            # Grade as incorrect or skip? Let's grade against 'None' which score_calculator should handle as incorrect.
                            teacher_answer_text = None # Explicitly set to None

                    # --- Call the Grading Function ---
                    try:
                        grade_result = score_calculator.grade_single_answer(
                            question_id=q_num,
                            student_text=s_text,
                            student_images=student_images, # Pass all student images
                            teacher_text=teacher_answer_text,
                            teacher_images=teacher_answer_images, # Pass all teacher images
                            # max_score can be passed if extracted, otherwise default is used
                        )
                        student_grades.append(grade_result)
                        if grade_result.get('error'):
                             student_errors.append(f"Q{q_num}: {grade_result['error']}")
                    except Exception as grade_e:
                         print(f"  ERROR grading Q{q_num} for {student_id}: {grade_e}")
                         student_errors.append(f"Critical error grading Q{q_num}: {grade_e}")
                         # Add a placeholder result indicating failure
                         student_grades.append({
                             'question_id': q_num, 'assigned_score': 0.0, 'max_score': config.DEFAULT_MAX_SCORE_PER_QUESTION,
                             'justification': f"Critical error during grading: {grade_e}", 'error': str(grade_e)
                         })

                # Store results for the student
                st.session_state.grading_results[student_id] = student_grades
                if student_errors:
                    grading_errors[student_id] = student_errors

            progress_bar_grading.progress(100, text="Grading complete!")

            # --- Step 4: Generate Output Files ---
            st.markdown("### Generating Output Files...")
            if not st.session_state.grading_results:
                 display_error("Grading process completed, but no results were generated.")
                 st.stop()

            # Create individual feedback files
            output_dir = config.RESULTS_DIR
            for student_id, results in st.session_state.grading_results.items():
                 report_generator.create_student_feedback_file(student_id, results, output_dir)

            # Create summary file
            summary_path = report_generator.create_summary_file(st.session_state.grading_results, output_dir)
            if summary_path:
                st.session_state.summary_file_path = summary_path
                display_success("Summary report generated.")
            else:
                display_error("Failed to generate summary report.")

            # Create ZIP file
            zip_path = zip_creator.create_zip_archive(source_dir=output_dir)
            if zip_path:
                st.session_state.zip_file_path = zip_path
                display_success("Feedback ZIP archive created.")
            else:
                display_warning("Failed to generate feedback ZIP archive (perhaps no feedback files were created?).")

            st.session_state.grading_complete = True
            display_success("All grading tasks finished!")
            if grading_errors:
                 st.warning("Some errors occurred during grading. Check details below:")
                 st.json({"grading_errors": grading_errors})


# --- Display Results Area ---
if st.session_state.grading_complete:
    st.markdown("---")
    st.header("📊 Grading Results")

    # Display Summary Report Content
    if st.session_state.summary_file_path and os.path.exists(st.session_state.summary_file_path):
        st.subheader("Grading Summary")
        try:
            with open(st.session_state.summary_file_path, 'r', encoding='utf-8') as f:
                summary_content = f.read()
            st.text_area("Summary Content", summary_content, height=300)

            # Summary Download Button
            with open(st.session_state.summary_file_path, "rb") as fp_summary:
                 st.download_button(
                    label="⬇️ Download Summary Report (.txt)",
                    data=fp_summary,
                    file_name=os.path.basename(st.session_state.summary_file_path),
                    mime="text/plain"
                )
        except Exception as e:
            display_error(f"Could not read or provide download for summary file: {e}")
    else:
        st.warning("Summary report file not available.")

    # Feedback ZIP Download Button
    if st.session_state.zip_file_path and os.path.exists(st.session_state.zip_file_path):
        st.subheader("Detailed Feedback Files")
        try:
            with open(st.session_state.zip_file_path, "rb") as fp_zip:
                st.download_button(
                    label="⬇️ Download All Student Feedback (.zip)",
                    data=fp_zip,
                    file_name=os.path.basename(st.session_state.zip_file_path),
                    mime="application/zip"
                )
        except Exception as e:
            display_error(f"Could not provide download for ZIP file: {e}")
    else:
        st.warning("Feedback ZIP archive not available.")

elif processing_started and not st.session_state.grading_complete:
    # If processing started but results aren't ready, show a message
    # (This helps if the process was interrupted or is still running)
    st.info("Processing files and grading answers. Please wait...")

else:
    # Initial state before button press
    st.info("Upload files and click 'Process Files & Start Grading' in the sidebar to begin.")