"""Poisson image editing.

"""

import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve

from os import path

def laplacian_matrix(background_high, background_width):
    """Generate the Poisson matrix.  
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Au = b 
    divide the object image  into [row1, row2, ... row[object_high]] as array b
    """
    mat_D = scipy.sparse.lil_matrix((background_width, background_width))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * background_high).tolil()
    
    mat_A.setdiag(-1, 1 * background_width)
    mat_A.setdiag(-1, -1 * background_width)
    
    return mat_A


def poisson_edit(target_object, background, target_mask):
    """The poisson blending function. 

    Refer to: 
    Perez et. al., "Poisson Image Editing", 2003.
    """

    background_high, background_width = background.shape[:-1]
    
    mat_A = laplacian_matrix(background_high, background_width)
    laplacian = mat_A.tocsc() # convert lil_matrix to csc_matrix

    # set the region outside the mask to identity    
    for y in range(1, background_high - 1):
        for x in range(1, background_width - 1):
            if target_mask[y, x] == 0:
                k = x + y * background_width
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + background_width] = 0
                mat_A[k, k - background_width] = 0

    mat_A = mat_A.tocsc()
    mask_flat = target_mask.flatten()    
    result_image = np.zeros_like(background)
    for channel in range(target_object.shape[2]):
        object_flat = target_object[0:background_high, 0:background_width, channel].flatten()
        background_flat = background[0:background_high, 0:background_width, channel].flatten()        

        # calculate matrix b by 4 * pixel(i,j) - pixel(i-1, j) - pixel(i+1, j) - pixel(i, j-1) - pixel(i, j+1) of target object
        mat_b = laplacian.dot(object_flat)

        # outside the mask: (frame of object > frame of mask)
        mat_b[mask_flat==0] = background_flat[mask_flat==0]

        u = spsolve(mat_A, mat_b)
        u = u.reshape((background_high, background_width))
        u[u > 255] = 255
        u[u < 0] = 0
        u = u.astype('uint8')

        result_image[0:background_high, 0:background_width, channel] = u

    return result_image

def main():    
    scr_dir = './input_data'
    out_dir = './output_data'
    i = 5
    target_object = cv2.imread(path.join(scr_dir, f"{i}target_object.png")) 
    background = cv2.imread(path.join(scr_dir, f"{i}background.jpg"))    
    target_mask = cv2.imread(path.join(scr_dir, f"{i}target_mask.png"), 
                      cv2.IMREAD_GRAYSCALE) 
    
    result = poisson_edit(target_object, background, target_mask)

    cv2.imwrite(path.join(out_dir, f"{i}possion.png"), result)

    # new_blend = np.zeros_like(background)
    # new_blend[target_mask == 0] = background[target_mask == 0]
    # cv2.imwrite(path.join(out_dir, f"{i}mat_a.png"), new_blend)

if __name__ == '__main__':
    main()