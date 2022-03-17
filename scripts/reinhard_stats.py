#Source: https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_normalization/reinhard_stats.py

import collections

import numpy as np

from histomicstk.preprocessing import color_conversion
from histomicstk.utils import sample_pixels

def reinhard_stats(slide_path, sample_fraction, magnification=None,
                   tissue_seg_mag=1.25):

    # generate a sampling of sample_pixels_rgb pixels from whole-slide image
    sample_pixels_rgb = sample_pixels(
        slide_path,
        sample_fraction=sample_fraction,
        magnification=magnification,
        tissue_seg_mag=tissue_seg_mag
    )

    # reshape the Nx3 pixel array into a 1 x N x 3 image for lab_mean_std
    sample_pixels_rgb = np.reshape(sample_pixels_rgb,
                                   (1, sample_pixels_rgb.shape[0], 3))

    # compute mean and stddev of sample pixels in Lab space
    Mu, Sigma = color_conversion.lab_mean_std(sample_pixels_rgb)

    # build named tuple for output
    ReinhardStats = collections.namedtuple('ReinhardStats', ['Mu', 'Sigma'])
    stats = ReinhardStats(Mu, Sigma)

    return stats