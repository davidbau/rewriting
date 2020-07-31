# A utility for saving a large number of images quickly without
# blocking a single thread to wait for each individual image to save.

import os
import PIL
from .workerpool import WorkerBase, WorkerPool
from . import pbar


def all_items_and_filenames(img_array, filename_pattern, index=()):
    for i, data in enumerate(img_array):
        inner_index = index + (i,)
        if PIL.Image.isImageType(data):
            yield data, (filename_pattern % inner_index)
        else:
            for img, name in all_items_and_filenames(data, filename_pattern,
                                                     inner_index):
                yield img, name


def expand_last_filename(img_array, filename_pattern):
    index, data = (), img_array
    while not PIL.Image.isImageType(data):
        index += (len(data) - 1,)
        data = data[len(data) - 1]
    return filename_pattern % index


def num_items(img_array):
    num = 1
    while not PIL.Image.isImageType(img_array):
        num *= len(img_array)
        img_array = img_array[-1]
    return num


def save_image_set(img_array, filename_pattern, sourcefile=None):
    '''
    Saves all the (PIL) images in the given array, using the
    given filename pattern (which should contain a `%d` to get
    the index number of the image).
    '''
    if sourcefile is not None:
        last_filename = expand_last_filename(img_array, filename_pattern)
        # Do nothing if the last file exists and is newer than the sourcefile
        if os.path.isfile(last_filename) and (os.path.getmtime(last_filename) >= os.path.getmtime(sourcefile)):
            pbar.descnext(None)
            return
    # Use multiple threads to write all the image files faster.
    pool = WorkerPool(worker=SaveImageWorker)
    for img, filename in pbar(
            all_items_and_filenames(img_array, filename_pattern),
            total=num_items(img_array)):
        pool.add(img, filename)
    pool.join()


class SaveImageWorker(WorkerBase):
    def work(self, img, filename, quality=99):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename, optimize=True, quality=quality)


class SaveImagePool(WorkerPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, worker=SaveImageWorker, **kwargs)
