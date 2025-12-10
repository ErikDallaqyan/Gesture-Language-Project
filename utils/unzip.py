import zipfile

def unzip_dataset(zip_path, extract_to="."):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
