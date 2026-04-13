import os
import tarfile

def download_and_extract():
    import shutil
    import requests
    
    # Get the directory where this utils.py file is located
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    
    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/DRKG/drkg.tar.gz"
    path = utils_dir  # Use the same folder as utils.py
    filename = "drkg.tar.gz"
    fn = os.path.join(path, filename)
    
    # Check if drkg.tsv already exists in the same folder
    if os.path.exists(os.path.join(utils_dir, "drkg", "drkg.tsv")):
        print("✓ DRKG data already downloaded")
        return
    
    opener, mode = tarfile.open, 'r:gz'
    os.makedirs(path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(path)
    
    while True:
        try:
            file = opener(filename, mode)
            try: 
                file.extractall()
                print("✓ Extraction complete")
            finally: 
                file.close()
            break
        except Exception:
            print(f"Downloading DRKG data from {url}...")
            f_remote = requests.get(url, stream=True)
            sz = f_remote.headers.get('content-length')
            assert f_remote.status_code == 200, 'fail to open {}'.format(url)
            with open(filename, 'wb') as writer:
                for chunk in f_remote.iter_content(chunk_size=1024*1024):
                    writer.write(chunk)
            print('Download finished. Unzipping the file...')
    
    os.chdir(cwd)
