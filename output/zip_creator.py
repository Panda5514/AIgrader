# Uses the zipfile module to create the archive.

# Python

# output/zip_creator.py

import os
import zipfile
from typing import Optional
from utils import config, helpers

def create_zip_archive(
    source_dir: str = config.RESULTS_DIR,
    output_filename: str = f"student_feedback_{helpers.get_timestamp_string()}.zip",
    output_dir: str = config.RESULTS_DIR # Save zip in the same results dir
    ) -> Optional[str]:
    """
    Creates a ZIP archive containing all .txt files from the source directory.

    Args:
        source_dir: The directory containing the files to zip (e.g., individual feedback files).
        output_filename: The desired name for the output ZIP file.
        output_dir: The directory where the ZIP file will be saved.

    Returns:
        The full path to the created ZIP archive, or None on failure.
    """
    output_zip_path = os.path.join(output_dir, output_filename)

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        print(f"Creating ZIP archive: {output_zip_path} from directory: {source_dir}")
        # Find all .txt files in the source directory
        files_to_zip = [
            f for f in os.listdir(source_dir)
            if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith('.txt')
        ]

        if not files_to_zip:
            print(f"No .txt files found in '{source_dir}' to archive.")
            return None

        # Create the ZIP file
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in files_to_zip:
                file_path = os.path.join(source_dir, filename)
                # Add file to zip using its base name (avoids including source_dir path)
                zipf.write(file_path, arcname=filename)
                print(f"  Adding: {filename}")

        print(f"Successfully created ZIP archive: {output_zip_path}")
        return output_zip_path

    except FileNotFoundError:
        print(f"Error: Source directory '{source_dir}' not found.")
        return None
    except Exception as e:
        print(f"Error creating ZIP archive at {output_zip_path}: {e}")
        # Clean up potentially corrupted zip file if it was created
        if os.path.exists(output_zip_path):
            try:
                os.remove(output_zip_path)
            except Exception as remove_e:
                print(f"Warning: Could not remove incomplete ZIP file {output_zip_path}: {remove_e}")
        return None

if __name__ == '__main__':
    print("\n--- Testing ZIP Creator ---")
    # Assumes report_generator test ran and created some .txt files in data/results/

    # Ensure the results directory exists for the test
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    # Create a dummy txt file if none exist from previous test
    if not any(f.endswith('.txt') for f in os.listdir(config.RESULTS_DIR)):
        dummy_file_path = os.path.join(config.RESULTS_DIR, "dummy_report.txt")
        with open(dummy_file_path, "w") as f:
            f.write("This is a dummy report.")
        print(f"Created dummy file for testing: {dummy_file_path}")


    zip_file_path = create_zip_archive()

    if zip_file_path:
        print(f"\nZIP file created at: {zip_file_path}")
        print("Contents (verify manually or by unzipping):")
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zipf:
                zipf.printdir()
        except Exception as e:
            print(f"Error reading zip file contents: {e}")
    else:
        print("\nZIP file creation failed.")

    print("\nCheck the 'data/results' directory for the generated .zip file.")


