import kagglehub
from pathlib import Path

def get_data_path():
    """Downloads the TESS dataset via Kaggle API and returns the local path."""
    dataset_handle = "ejlok1/toronto-emotional-speech-set-tess"

    print("Checking for TESS dataset...")
    try:
        path = kagglehub.dataset_download(dataset_handle)
        print(f"Dataset is ready at: {path}")
        return Path(path)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to download dataset. {e}")
        return None

if __name__ == "__main__":
    get_data_path()