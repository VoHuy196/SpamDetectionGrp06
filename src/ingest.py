import os
import pandas as pd
import urllib.request
import zipfile

def ingest(url: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"Data already exists at {dest_path}. Skipping download.")
        return pd.read_csv(dest_path)

    print(f"Downloading from {url} to {dest_path}")
    data_dir = os.path.dirname(dest_path)
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "temp.zip")
    extract_path = os.path.join(data_dir, "temp_extracted")
    
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    txt_file = os.path.join(extract_path, "SMSSpamCollection")
    df = pd.read_csv(txt_file, sep='	', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df.to_csv(dest_path, index=False)
    print(f"Ingested {len(df)} rows.")
    return df
