import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import glob
import poisson
from os import path

def stitch(img_left, img_righttrans, blending_mode):
    # Apply blending if specified
    if blending_mode == "linear_blending":
        img_wrap = linear_blending(img_left, img_righttrans)
    elif blending_mode == "poisson_blending":
        img_wrap = poisson_blending(img_left, img_righttrans)
    else:  # error
        raise ValueError(f"Invalid blending mode: {blending_mode}")

    return img_wrap

def poisson_blending(img_left, img_righttrans):
    h_trans, w_trans = img_righttrans.shape[:2]
    h_left, w_left = img_left.shape[:2]
    # Initialize masks to detect non-zero areas in both images
    mask_left = np.any(img_left > 0, axis=2).astype(np.float32) # return True or False
    mask_trans = np.any(img_righttrans > 0, axis=2).astype(np.float32)

    # Calculate overlap mask
    overlap_mask = np.zeros((h_trans, w_trans), dtype="int")
    for i in range(h_left):
        for j in range(w_left):
            if mask_left[i, j] == False and mask_trans[i, j] == True:
                overlap_mask[i, j] = 1

    overlap_mask = overlap_mask.astype(np.uint8)
    # Shrink overlap region edges by 2 pixels
    kernel = np.ones((5, 5), np.uint8)  # Define a kernel of size 5x5 (to erode 2 pixels)
    shrunk_mask = cv2.erode(overlap_mask, kernel, iterations=1)  # Shrink the mask
    overlap_mask = shrunk_mask  # Update the overlap mask

    # Create target_mask
    target_mask = np.zeros((h_trans, w_trans), dtype=np.uint8)  # Same size as img_left, all black
    target_mask[overlap_mask == 1] = 255  # Set overlap region to white

    # # Create target_object
    # target_object = np.zeros_like(img_left)  # Same size as img_left, all black
    # for i in range(h_left):
    #     for j in range(w_left):
    #         if overlap_mask[i, j] == 1:
    #             target_object[i, j] = img_trans[i, j]  # Copy img_trans values in the overlap region
    target_object = img_righttrans
    
    img_wrap = np.copy(img_righttrans)
    img_wrap[:h_left, :w_left] = np.where(img_left > 0, img_left, img_wrap[:h_left, :w_left]) 

    background = img_wrap

    target_mask_path = path.join(path.dirname(folder), f"{name}_target_mask.png")
    target_object_path = path.join(path.dirname(folder), f"{name}_target_object.png")
    background_path = path.join(path.dirname(folder), f"{name}_background.png")
    cv2.imwrite(target_mask_path, target_mask)
    cv2.imwrite(target_object_path, target_object)
    cv2.imwrite(background_path, background)

    img_wrap = poisson.poisson_edit(target_object, background, target_mask)

    return img_wrap

def linear_blending(img_left, img_trans):
    h_trans, w_trans = img_trans.shape[:2]
    h_left, w_left = img_left.shape[:2]
    # Initialize masks to detect non-zero areas in both images
    mask_left = np.any(img_left > 0, axis=2).astype(np.float32) # return True or False
    mask_trans = np.any(img_trans > 0, axis=2).astype(np.float32)

    # Calculate overlap mask
    overlap_mask = np.zeros((h_trans, w_trans), dtype="int")
    for i in range(h_left):
        for j in range(w_left):
            if mask_left[i, j] == True and mask_trans[i, j] == True:
                overlap_mask[i, j] = 1

    # Create an alpha mask that linearly transitions in the overlapping region
    alpha_mask = np.zeros((h_trans, w_trans), dtype=np.float32)
    for i in range(h_trans):  # Loop through each row
        minIdx, maxIdx = -1, -1
        for j in range(w_trans):  # Find the overlap range for each row
            if overlap_mask[i, j] == 1 and minIdx == -1:
                minIdx = j
            if overlap_mask[i, j] == 1:
                maxIdx = j

        # Skip rows without overlap or with only one overlapping pixel
        if minIdx == -1 or minIdx == maxIdx:
            continue

        # Create a gradient from 1 to 0 over the overlap region
        for j in range(minIdx, maxIdx + 1):
            alpha_mask[i, j] = 1 - (j - minIdx) / (maxIdx - minIdx)

    img_wrap = np.copy(img_trans)
    img_wrap[:h_left, :w_left] = np.where(img_left > 0, img_left, img_wrap[:h_left, :w_left])

    # Blend the images using the alpha mask
    for i in range(h_trans):
        for j in range(w_trans):
            if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                img_wrap[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_trans[i, j]

    return img_wrap

if __name__ == "__main__":

    name = "hill"
    folder = "output_data/"

    # img_left = cv2.imread(f"{folder}{name}_left.jpg")
    # img_righttrans = cv2.imread(f"{folder}{name}_righttrans.jpg")

    # Load the arrays
    img_left = np.load(f"{folder}homo/{name}_left1.npy")
    img_righttrans = np.load(f"{folder}homo/{name}_righttrans.npy")

    # "linear_blending", "poisson_blending"
    blending_mode = "poisson_blending"

    # Stitch the current `img_wrap` with the next image in the sequence
    img_wrap = stitch(img_left, img_righttrans, blending_mode)

    saveFilePath = f"output_data/{name}_{blending_mode}.jpg"
    cv2.imwrite(saveFilePath, img_wrap)

