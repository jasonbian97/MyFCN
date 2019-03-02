#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw,resizeto224


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_resize_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resizeto224(Image.open(dir + id + suffix))
        # yield get_square(im, pos)
        yield im
def to_resize_imgs_2bin(ids, dir, suffix):
    for id, pos in ids:
        im = resizeto224(Image.open(dir + id + suffix))
        im=im==255.
        im=im*1
        # yield get_square(im, pos)
        yield im

def get_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""

    imgs = to_resize_imgs(ids, dir_img, '.jpg')

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_resize_imgs_2bin(ids, dir_mask, '.jpg')


    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
