import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import glob

# Stitcher, 
def stitch(img_left, img_right, blending_mode, ratio, num_iter=1000, threshold=3.0):
    # Ensure the output directory exists
    os.makedirs("output", exist_ok=True)

    print(f"step1: get key points of {base_name} each image...")
    kp_left, kp_right, des_left, des_right = detect_and_describe(img_left, img_right)

    print(f"step2: {i + 1}match key points of {base_name} each images with ratio {ratio}...")
    left_match_points, right_match_points = match_keypoints(kp_left, kp_right, des_left, des_right, ratio)
    print(f"The number of {base_name} matching points:", len(left_match_points))

    # Construct the save path
    print(f"step3: draw key points match of {base_name} each images...")
    save_path = f"output/{base_name}_{i + 1}keyPoint_match_ratio={ratio}.jpg"
    draw_matches(left_match_points, right_match_points, img_left, img_right, save_path=save_path)

    print(f"step4: use RANSAC to get best homography matric of {base_name} images...")
    H = RANSAC_homography(left_match_points, right_match_points, num_iter, threshold)

    print("step5: warp and blend of two images...")
    img_wrap = warp(img_left, img_right, H, blending_mode, base_name) 
    
    return img_wrap

def detect_and_describe(img_left, img_right):
    gray1 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None) # cv.drawKeypoints(image, keypoints, outImage[, color[, flags]])
    kp2, des2 = sift.detectAndCompute(gray2, None) 
    return kp1, kp2, des1, des2

def draw_matches(left_match_points, right_match_points, img_left, img_right, save_path=None):
    # Create a new image by placing img_right to the right of img_left
    height = max(img_left.shape[0], img_right.shape[0])
    width = img_left.shape[1] + img_right.shape[1]
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[:img_left.shape[0], :img_left.shape[1]] = img_left
    result[:img_right.shape[0], img_left.shape[1]:] = img_right

    # Plot matches
    plt.figure(figsize=(10, 5))
    plt.imshow(result[:, :, ::-1])  # Convert BGR to RGB for Matplotlib

    for pt1, pt2 in zip(left_match_points, right_match_points):
        # Shift the right image points to match the concatenated image coordinates
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0] + img_left.shape[1]), int(pt2[1])
        
        # Draw circles at each keypoint
        color = np.random.rand(3)  # Generate a random color for each match
        plt.scatter([x1, x2], [y1, y2], color=color, s=10)
        
        # Draw line connecting the matches
        plt.plot([x1, x2], [y1, y2], color=color)

    plt.axis('off')
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()


def match_keypoints(kp_left, kp_right, des_left, des_right, ratio):
    left_match_points = []
    right_match_points = []

    for i, d_left in enumerate(des_left):
        # Set initial distances to large values
        min_dist = np.inf
        sec_min_dist = np.inf
        best_match_idx = -1

        for j, d_right in enumerate(des_right):
            # Calculate Euclidean distance
            dist = np.linalg.norm(d_left - d_right)
            
            # Update best and second-best matches
            if dist < min_dist:
                sec_min_dist = min_dist
                min_dist = dist
                best_match_idx = j
            elif dist < sec_min_dist:
                sec_min_dist = dist
        
        # Apply ratio test to keep only good matches
        if min_dist < sec_min_dist * ratio:
            left_match_points.append(kp_left[i].pt)  # Get (x, y) coordinates of keypoint
            right_match_points.append(kp_right[best_match_idx].pt)

    return left_match_points, right_match_points

def RANSAC_homography(left_match_points, right_match_points, num_iter, threshold):
    max_inliers = 0
    best_H = None

    for _ in range(num_iter):
        # Step 1: Randomly select 4 point pairs
        sample_indices = random.sample(range(len(left_match_points)), 4)
        sample_left_points = np.array([left_match_points[i] for i in sample_indices])
        sample_right_points = np.array([right_match_points[i] for i in sample_indices])

        # Step 2: Compute homography from these 4 point pairs
        H = get_homography(sample_left_points, sample_right_points)

        # Step 3: Count inliers by applying the homography to all left_match_points
        inliers = 0
        for i in range(len(right_match_points)):
            # Convert the point to homogeneous coordinates and apply homography
            right_point = np.array([right_match_points[i][0], right_match_points[i][1], 1])
            transformed_point = H @ right_point
            transformed_point /= transformed_point[2]  # Normalize to convert back to 2D

            # Calculate Euclidean distance between transformed left point and actual right point
            left_point = np.array(left_match_points[i])
            distance = np.linalg.norm(transformed_point[:2] - left_point)

            # Count as inlier if distance is within the threshold
            if distance < threshold:
                inliers += 1

        # Step 4: Keep track of the best homography with the most inliers
        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H

    print("Maximum Inliers points Found:", max_inliers)
    return best_H

def get_homography(points_in_img_left, points_in_img_right):
    # In this function, we use points of img1 and img2 to derive the relation between (u, v) and (X, Y)
    # img_left = H @ img_right
    P = []
    for i in range(len(points_in_img_right)):
        X, Y = points_in_img_right[i, 0], points_in_img_right[i, 1]
        u, v = points_in_img_left[i, 0], points_in_img_left[i, 1]
        P.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        P.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    
    P = np.array(P)
    # Use SVD to solve the Ph = 0
    _, _, V = np.linalg.svd(P)
    # Take the last row of V (smallest sigular value that is closest ans of Ph = 0)                                  
    H = V[-1].reshape(3, 3)                                     
    return H / H[2, 2]

def warp(img_left, img_right, H, blending_mode, fileName1):
    # Retrieve dimensions
    h_left, w_left = img_left.shape[:2] 
    h_right, w_right = img_right.shape[:2] # 0 x down height, 1 y right width, 2 z depth
    # Calculate inverse homography for backward warping
    H_inverse = np.linalg.inv(H)

    # print("point2 (x, y): ", int(h_right), ", ",int(w_right) )

    # Transform points to find the new width, 0 x right width, 1 y down height
    point1 = H @ np.array([w_right, 0, 1]) # right top
    point2 = H @ np.array([w_right, h_right, 1]) #right bottom
    point3 = H @ np.array([0, 0, 1]) # left top
    point4 = H @ np.array([0, h_right, 1]) # left bottom
    point1 /= point1[2] 
    point2 /= point2[2]
    point3 /= point3[2]
    point4 /= point4[2]

    # print(f"point1 (x, y): {point1[0]}, {point1[1]}")
    # print(f"point2 (x, y): {point2[0]}, {point2[1]}")
    # print(f"point3 (x, y): {point3[0]}, {point3[1]}") # 0 x right width, y down height
    w_max = int(max(point1[0], point2[0], int(w_left)))
    w_min = int(min(point3[0], point4[0], 0))
    h_max = int(max(point2[1], point4[1], int(h_left)))
    h_min = int(min(point1[1], point3[1], 0)) 

    h_newMax = int(h_max - h_min)
    w_newMax = int(w_max - w_min)
    # Initialize the transformed image with dimensions large enough to hold both images
    img_trans = np.zeros((h_newMax, w_newMax, 3), dtype=img_left.dtype)

    # print("img_trans.shape[0] : ", img_trans.shape[0]) # 0 x down height
    # print("img_trans.shape[1] : ", img_trans.shape[1]) # 1 y right width

    # Warp the right image onto the transformed image, and shift it
    for i in range(h_min, h_max): # 0 x down height -> i
        for j in range(w_min, w_max): # 1 y right width -> j, shift -h_min
            # Map (j, i) in the output image back to coordinates in img_right using H_inverse
            coor = np.array([j, i, 1]) # 0 j -> x right width, i -> y down height
            coor_right = H_inverse @ coor
            coor_right /= coor_right[2]  # Normalize to 2D coordinates
            
            x, y = int(round(coor_right[0])), int(round(coor_right[1]))  # 0 x for width, 1 y for height
            
            # Check if the mapped coordinates (x, y) are within img_right's bounds
            if 0 <= x < w_right and 0 <= y < h_right:
                img_trans[i - h_min, j - w_min] = img_right[y, x]  # 0 down height, 1 right width

    # shift left image and make it same size with img_trans
    img_left1 = np.zeros_like(img_trans)
    img_left1[-h_min:h_left - h_min, -w_min:w_left - w_min] = img_left  # Shift img_left down by -h_min


    saveFilePath = f"output/warp_{fileName1}_left.jpg"
    cv2.imwrite(saveFilePath, img_left1)
    saveFilePath = f"output/warp_{fileName1}_righttrans.jpg"
    cv2.imwrite(saveFilePath, img_trans)

    # Apply blending if specified
    if blending_mode == "linear_blending":
        img_wrap = linear_blending(img_left1, img_trans)
    else:  # no blending
        img_wrap = np.zeros_like(img_trans)
        img_wrap[-h_min:h_left - h_min, :w_left] = img_left  # Shift img_left down by -h_min
        img_wrap = np.where(img_trans > 0, img_trans, img_wrap)  # Overlay img_trans over img_wrap

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
    # fileNameList = [('S')] 
    # fileNameList = [('TV'), ('S'), ('hill')]
    # fileNameList = [('hill')]
    # fileNameList = [('TV')]
    fileNameList = [('building')]
    # fileNameList = [('guiter')]


    # "noblending" , "linear_blending"
    blending_mode = "linear_blending"
    ratio = 0.5
    num_iter = 1000
    threshold=1
    # src_path = "data/"
    src_path = "my_data/"

for base_name in fileNameList:
    # Find all images that match the pattern, e.g., building1.jpg, building2.jpg, etc.
    image_files = sorted(glob.glob(f"{src_path}{base_name}*.jpg"))
    n = len(image_files)  # Total number of matching images

    if n < 2:
        print(f"Error: Not enough images found for {base_name} to stitch.")
        continue

    # Initialize the first image as the base for stitching
    img_wrap = cv2.imread(image_files[0])
    if img_wrap is None:
        print(f"Error: Could not load {image_files[0]}")
        continue

    # Stitch sequential images
    for i in range(1, n):
        img_right = cv2.imread(image_files[i])
        if img_right is None:
            print(f"Error: Could not load {image_files[i]}")
            break

        # Stitch the current `img_wrap` with the next image in the sequence
        img_wrap = stitch(img_wrap, img_right, blending_mode, ratio, num_iter, threshold)
        # Save each intermediate result
        saveFilePath = f"output/warp_{base_name}_{blending_mode}_ratio={ratio}_{i + 1}imgs.jpg"
        cv2.imwrite(saveFilePath, img_wrap)
    # Plot and save the stitched image
    plt.figure(13)
    plt.title("warp_img")
    plt.imshow(img_wrap[:,:,::-1].astype(int))  # Rearranges the channels from BGR to RGB

    saveFilePath = f"output/warp_{base_name}_{blending_mode}_ratio={ratio}.jpg"
    # saveFilePath = f"output/warp_{imgs}_{blending_mode}_ratio={ratio}_threshold={threshold}.jpg"
    cv2.imwrite(saveFilePath, img_wrap)