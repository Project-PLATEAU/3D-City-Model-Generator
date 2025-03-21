import json
import glob
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def get_files(directory, format='png'):
    """
    To get a list of file names in one directory, especially images
    :param directory: a path to the directory of the image files
    :return: a list of all the file names in that directory
    """
    if format == 'png':
        file_list = glob.glob(directory + "*.png")
    elif format == 'jpg':
        file_list = glob.glob(directory + "*.jpg")
    elif format == 'tif':
        file_list = glob.glob(directory + "*.tif")
    elif format == 'tiff':
        file_list = glob.glob(directory + "*.tiff")
    else:
        raise ValueError("dataset do not support")

    file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return file_list

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
