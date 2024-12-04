import os
import requests
import tarfile
from tqdm import tqdm

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    t = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")

    with open(save_path, "wb") as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("WARNING: Downloaded file size does not match expected size!")


def extract_tgz(file_path, extract_dir):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        print(f"Extracted to: {extract_dir}")