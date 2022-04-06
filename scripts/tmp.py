import numpy as np
import cv2 as cv
import Stain_Normalization.stain_utils as ut
from glob import glob


def lab_split(I):
    I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    I = I.astype(np.float32)
    I1, I2, I3 = cv.split(I)
    I1 /= 2.55
    I2 -= 128.0
    I3 -= 128.0
    return I1, I2, I3

def get_mean_std(I):
    I1, I2, I3 = lab_split(I)
    m1, sd1 = cv.meanStdDev(I1)
    m2, sd2 = cv.meanStdDev(I2)
    m3, sd3 = cv.meanStdDev(I3)
    means = m1, m2, m3
    stds = sd1, sd2, sd3
    return means, stds

def get_image_data(target):
    target = ut.standardize_brightness(target)
    means, stds = get_mean_std(target)
    
    return means, stds

def get_means_and_stds(data_paths):
    
    img_paths = glob(data_paths + '/train/images/*.png')

    img_data = []

    for index in range(0, len(img_paths)):
        img_path = img_paths[index]
        image = np.array(cv.imread(img_path, cv.IMREAD_COLOR))

        mean, stds = get_image_data(image)

        img_data.append({
            "path": img_path,
            "mean": mean[0] + mean[1] + mean[2],
            "stds": stds[0] + stds[1] + stds[2]
        })

    sorted_by_mean = sorted(img_data, key=lambda d: d['mean'])
    print(f"Min MEAN: {sorted_by_mean[0]}")
    print(f"Max MEAN: {sorted_by_mean[-1]}")
    difference = sorted_by_mean[-1]["mean"] - sorted_by_mean[0]["mean"]
    mid = sorted_by_mean[0]["mean"] + difference/2
    middle_element = {}
    el_index = 0
    for index in range(0, len(sorted_by_mean)):
        if sorted_by_mean[index]["mean"] > mid:
            middle_element = sorted_by_mean[index]
            el_index = index
            break
    print(f"Total length of array: {len(sorted_by_mean)}")
    print(f"Element index: {el_index}")
    print(middle_element)

    sorted_by_stds = sorted(img_data, key=lambda d: d['stds'])
    print(f"Min STD: {sorted_by_stds[0]}")
    print(f"Max STD: {sorted_by_stds[-1]}")
    difference = sorted_by_stds[-1]["stds"] - sorted_by_stds[0]["stds"]
    mid = sorted_by_stds[0]["stds"] + difference/2
    middle_element = {}
    el_index = 0
    for index in range(0, len(sorted_by_stds)):
        if sorted_by_stds[index]["stds"] > mid:
            middle_element = sorted_by_stds[index]
            el_index = index
            break
    print(f"Total length of array: {len(sorted_by_stds)}")
    print(f"Element index: {el_index}")
    print(middle_element)



