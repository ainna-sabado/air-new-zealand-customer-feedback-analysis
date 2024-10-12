import os
import nltk

# Define the path to the NLTK data directory
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")

# Check if the stopwords data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    # If not found, download the stopwords data
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', download_dir=nltk_data_path)

# Set the NLTK data path in your script
nltk.data.path.append(nltk_data_path)
