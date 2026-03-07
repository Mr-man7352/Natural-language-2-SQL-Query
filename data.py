import tarfile
import requests

def download_and_extract():
    """Download the WikiSQL dataset archive and extract it locally."""
    url = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open("data.tar.bz2", "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            print(f"\rDownloading: {downloaded}/{total} bytes ({100*downloaded//total}%)", end="")

    print("\nDone!")

    # Extract
    print("Extracting...")
    with tarfile.open("data.tar.bz2", "r:bz2") as tar:
        tar.extractall()



def main():
    """Entry point: trigger dataset download and extraction."""
    download_and_extract()


if __name__ == "__main__":
    main()
