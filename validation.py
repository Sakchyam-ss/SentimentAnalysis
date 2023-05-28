import re

# Function to validate Yelp business URL
def is_valid_yelp_url(url):
    # Check if URL is provided
    if not url:
        return False

    # Validate if the URL matches the expected Yelp format
    pattern = r'^https:\/\/www\.yelp\.com\/.*$'
    match = re.match(pattern, url)
    if not match:
        return False

    # Additional validation logic specific to Yelp URLs can be added here
    # (e.g., checking for specific subdomains or path patterns)

    return True

# Function to validate user text input
def is_valid_text(text):
    # Check if text is provided
    if not text:
        return False
    return True

# Function to validate uploaded file format
def is_valid_file_format(file_extension, expected_formats):
    return file_extension in expected_formats

# Example of expected file formats
EXPECTED_FILE_FORMATS = ["csv", "xls", "xlsx"]
