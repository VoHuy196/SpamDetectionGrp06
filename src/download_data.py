import os
import urllib.request
import zipfile
import pandas as pd

# Data URL and paths
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
CSV_PATH = os.path.join(DATA_DIR, "spam.csv")

def download_and_extract():
    # Create data directory if not exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    # Download ZIP file
    print(f"Downloading dataset from {DATA_URL}...")
    try:
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Successfully downloaded zip file.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        return

    # Extract ZIP
    print("Extracting zip file...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return

    # Convert tab-separated format to standard CSV
    raw_file_path = os.path.join(DATA_DIR, "SMSSpamCollection")
    if os.path.exists(raw_file_path):
        print("Converting dataset to CSV format...")
        try:
            # Read tab-separated file
            df = pd.read_csv(raw_file_path, sep='\t', header=None, names=['label', 'text'])
            # Save to CSV
            df.to_csv(CSV_PATH, index=False, encoding='utf-8')
            print(f"Dataset successfully saved to: {CSV_PATH}")
            print(f"Total rows: {len(df)}")
            print(df.head())
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    else:
        print("SMSSpamCollection file not found after extraction.")

    # Cleanup temp files
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    readme_path = os.path.join(DATA_DIR, "readme")
    if os.path.exists(readme_path):
        os.remove(readme_path)
    if os.path.exists(raw_file_path):
        os.remove(raw_file_path)
    print("Cleanup complete.")

if __name__ == "__main__":
    download_and_extract()
