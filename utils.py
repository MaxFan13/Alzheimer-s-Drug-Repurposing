import os
import tarfile

import requests


def download_and_extract():
    """Download and extract the DRKG dataset archive if not already present.

    Checks for an existing extracted directory before downloading to avoid
    redundant network requests. The archive is saved alongside this script
    and extracted in place.
    """
    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/DRKG/drkg.tar.gz"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    archive = os.path.join(script_dir, "drkg.tar.gz")

    if os.path.exists(os.path.join(script_dir, "drkg", "drkg.tsv")):
        return

    if not os.path.exists(archive):
        _download_file(url, archive)

    with tarfile.open(archive, 'r:gz') as tf:
        tf.extractall(script_dir)


def _download_file(url, dest):
    """Stream-download a file from url and write it to dest.

    Args:
        url: URL of the file to download.
        dest: Local file path to write the downloaded content to.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status code.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print("Download finished.")
