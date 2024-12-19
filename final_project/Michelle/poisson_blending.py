import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def poisson_blend(source, target, mask, offset):
    """
    Perform Poisson blending of source image into the target image.

    Parameters:
        source (numpy.ndarray): Source image.
        target (numpy.ndarray): Target image.
        mask (numpy.ndarray): Binary mask indicating the region to blend.
        offset (tuple): (x_offset, y_offset) of the source image in the target image.

    Returns:
        numpy.ndarray: Blended image.
    """
    h, w = mask.shape
    y_offset, x_offset = offset

    # Indices of the pixels in the mask
    mask_indices = np.where(mask.flatten() > 0)[0]
    num_pixels = len(mask_indices)

    # Create a mapping from pixel index in the mask to equation index
    index_map = -1 * np.ones_like(mask, dtype=int)
    index_map[mask > 0] = np.arange(num_pixels)

    # Sparse matrix for the Poisson equation
    A = lil_matrix((num_pixels, num_pixels))
    b = np.zeros(num_pixels)

    # Fill the sparse matrix A and the vector b
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                eq_index = index_map[y, x]
                A[eq_index, eq_index] = 4

                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] > 0:
                            A[eq_index, index_map[ny, nx]] = -1
                        else:
                            b[eq_index] += target[y_offset + ny, x_offset + nx]

                b[eq_index] += (
                    4 * source[y, x]
                    - source[y - 1, x] - source[y + 1, x]
                    - source[y, x - 1] - source[y, x + 1]
                )

    # Solve the sparse linear system
    blended_region = spsolve(A.tocsr(), b)

    # Insert the blended region back into the target image
    result = target.copy()
    blended_region = blended_region.clip(0, 255).astype(np.uint8)
    result[y_offset : y_offset + h, x_offset : x_offset + w][mask > 0] = blended_region

    return result

# Load images
source = cv2.imread("source.jpg", cv2.IMREAD_COLOR)
target = cv2.imread("target.jpg", cv2.IMREAD_COLOR)
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# Ensure mask is binary
mask = (mask > 128).astype(np.uint8)

# Define the offset (where to place the source in the target image)
offset = (50, 100)  # Example values

# Perform Poisson blending
blended = poisson_blend(source, target, mask, offset)

# Save and display the result
cv2.imwrite("blended_result.jpg", blended)
cv2.imshow("Blended Image", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
