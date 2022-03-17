# Source: https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_normalization/reinhard.py

import numpy as np

from histomicstk.preprocessing import color_conversion

def reinhard(
        im_src, target_mu, target_sigma, src_mu=None, src_sigma=None,
        mask_out=None):
    # convert input image to LAB color space
    im_lab = color_conversion.rgb_to_lab(im_src)

    # mask out irrelevant tissue / whitespace / etc
    if mask_out is not None:
        mask_out = mask_out[..., None]
        im_lab = np.ma.masked_array(
            im_lab, mask=np.tile(mask_out, (1, 1, 3)))

    # calculate src_mu and src_sigma if either is not provided
    if (src_mu is None) or (src_sigma is None):
        src_mu = [im_lab[..., i].mean() for i in range(3)]
        src_sigma = [im_lab[..., i].std() for i in range(3)]

    # scale to unit variance
    for i in range(3):
        im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i]

    # rescale and recenter to match target statistics
    for i in range(3):
        im_lab[:, :, i] = im_lab[:, :, i] * target_sigma[i] + target_mu[i]

    # convert back to RGB colorspace
    im_normalized = color_conversion.lab_to_rgb(im_lab)
    im_normalized[im_normalized > 255] = 255
    im_normalized[im_normalized < 0] = 0

    # return masked values and reconstruct unmasked LAB image
    if mask_out is not None:
        im_normalized = im_normalized.data
        for i in range(3):
            original = im_src[:, :, i].copy()
            new = im_normalized[:, :, i].copy()
            original[np.not_equal(mask_out[:, :, 0], True)] = 0
            new[mask_out[:, :, 0]] = 0
            im_normalized[:, :, i] = new + original

    im_normalized = im_normalized.astype(np.uint8)

    return im_normalized