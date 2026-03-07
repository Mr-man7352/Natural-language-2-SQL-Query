"""
download.py — Downloads and extracts the WikiSQL dataset archive.
"""

import tarfile
import requests
from config import DATASET_URL, ARCHIVE_NAME


def download_and_extract() -> None:
    """Download the WikiSQL dataset archive and extract it locally."""
    print(f"Downloading dataset from {DATASET_URL} ...")
    response = requests.get(DATASET_URL, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(ARCHIVE_NAME, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(
                    f"\rProgress: {downloaded}/{total} bytes "
                    f"({100 * downloaded // total}%)",
                    end="",
                )

    print("\nDownload complete.")

    print("Extracting archive...")
    with tarfile.open(ARCHIVE_NAME, "r:bz2") as tar:
        tar.extractall()
    print("Extraction complete.")
