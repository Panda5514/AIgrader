# utils/helpers.py

import re
import datetime
import os

def sanitize_filename(filename: str) -> str:
    """Removes potentially problematic characters from a filename."""
    # Remove directory separators
    filename = filename.replace('/', '_').replace('\\', '_')
    # Remove characters that are problematic in filenames
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
    # Replace whitespace with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Reduce multiple underscores to one
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores/periods
    sanitized = sanitized.strip('._')
    # Limit length if necessary (optional)
    # max_len = 100
    # if len(sanitized) > max_len:
    #     name, ext = os.path.splitext(sanitized)
    #     sanitized = name[:max_len - len(ext) - 1] + "~" + ext

    if not sanitized:
        # If the name becomes empty, provide a default
        sanitized = "unnamed_file"
    return sanitized

def get_timestamp_string() -> str:
    """Returns a timestamp string suitable for filenames (YYYYMMDD_HHMMSS)."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == '__main__':
    print("--- Testing Helpers ---")
    test_names = [
        "Student Answer Sheet ?.pdf",
        "John Doe / Section A\\ ответы.docx",
        "  leading_and_trailing_spaces_ .txt",
        "file*with?special<chars>|.png",
        "",
        "../relative/path/attempt.jpg"
    ]
    for name in test_names:
        print(f"Original: '{name}' -> Sanitized: '{sanitize_filename(name)}'")

    print(f"\nCurrent Timestamp String: {get_timestamp_string()}")