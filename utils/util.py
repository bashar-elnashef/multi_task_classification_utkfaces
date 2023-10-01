import os
import zipfile
import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import gdown


def categorize_age(age):
    """Create a function to categorize an age."""
    # Define the labels and corresponding age ranges
    new_ranges = ['1-3', '4-7', '8-12', '13-19', '20-35', '36-55', '56-75', '76-116']
    # new_label_classes = ['Baby', 'Child', 'Preteen', 'Teenager', 'Young Adult', 'Adult', 'Senior', 'Elderly']
    new_labels = [0, 1, 2, 3, 4, 5, 6, 7]

    for i, age_range in enumerate(new_ranges):
        start, end = map(int, age_range.split('-'))
        if start <= age <= end:
            return new_labels[i]
    return None  # Return None for ages that don't fall into any range

def download_utkfaces_dataset(url, destination=None, remove_source=True, download=False):
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.

    Returns:
        pathlib.Path to downloaded data.
    """
    # Setup path to data folder
    image_path = Path(destination)
    output = f"{destination}/UTKFaces.zip" 

    print(f'image_path.is_dir() = {image_path.is_dir()}')
    # If the image folder doesn't exist, download it and prepare it...

    if image_path.is_dir():
        download = not any(image_path.iterdir())
    else: 
        image_path.mkdir(parents=True, exist_ok=True)
        download = True
        
    if download:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        print(url)

        if url is not None:
            gdown.download(url, output, quiet=False)
            # Unzip data
            with zipfile.ZipFile(output, "r") as zip_ref:
                print(f"[INFO] Unzipping {output} data...")
                zip_ref.extractall(image_path)

            # Remove .zip file
            if remove_source:
                print(f"[INFO] Removing the source file {output} from folder...")
                os.remove(output)
        else:
            print("[INFO] Could not download the data, the given url are None.")
            return


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
