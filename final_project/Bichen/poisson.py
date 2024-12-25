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

    y_max, x_max = background.shape[:-1]
    y_min, x_min = 0, 0

    background_width = x_max - x_min
    background_high = y_max - y_min
    
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
    for channel in range(target_object.shape[2]):
        object_flat = target_object[y_min:y_max, x_min:x_max, channel].flatten()
        background_flat = background[y_min:y_max, x_min:x_max, channel].flatten()        

        # calculate matrix b by 4 * pixel(i,j) - pixel(i-1, j) - pixel(i+1, j) - pixel(i, j-1) - pixel(i, j+1)
        mat_b = laplacian.dot(object_flat)

        # outside the mask:
        mat_b[mask_flat==0] = background_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b)
        #print(x.shape)
        x = x.reshape((background_high, background_width))
        #print(x.shape)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        print(x.shape)

        background[y_min:y_max, x_min:x_max, channel] = x

    return background

def main():    
    scr_dir = './input_data'
    out_dir = scr_dir
    i = 2
    target_object = cv2.imread(path.join(scr_dir, f"{i}target_object.png")) 
    background = cv2.imread(path.join(scr_dir, f"{i}background.jpg"))    
    target_mask = cv2.imread(path.join(scr_dir, f"{i}target_mask.png"), 
                      cv2.IMREAD_GRAYSCALE) 

    # Check unique values in the mask
    unique_values = np.unique(target_mask)
    print(f"Unique values in the target mask: {unique_values}")
    
    result = poisson_edit(target_object, background, target_mask)

    cv2.imwrite(path.join(out_dir, f"{i}possion.png"), result)
    

if __name__ == '__main__':
    main()