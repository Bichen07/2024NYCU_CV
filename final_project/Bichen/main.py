import cv2

from create_mask import MaskPainter
from move_mask import MaskMover
from poisson import poisson_edit

#import argparse
import getopt
import sys
from os import path


def usage():
    print("Usage: python main.py [options] \n\n\
    Options: \n\
    \t-h\tPrint a brief help message and exits..\n\
    \t-s\t(Required) Specify a source image.\n\
    \t-t\t(Required) Specify a target image.\n\
    \t-m\t(Optional) Specify a mask image with the object in white and other part in black, ignore this option if you plan to draw it later.")


if __name__ == '__main__':

    scr_dir = './input_data'
    out_dir = scr_dir
    i = 1
    source = cv2.imread(path.join(scr_dir, f"object_cut{i}.jpg")) 
    image_path = cv2.imread(path.join(scr_dir, f"background{i}.jpg"))    
    mask_path = cv2.imread(path.join(scr_dir, f"mask{i}.png"), 
                      cv2.IMREAD_GRAYSCALE) 

    # adjust mask position for target image
    print('Please move the object to desired location to apparate.\n')
    mm = MaskMover(image_path, mask_path)
    offset_x, offset_y, target_mask_path = mm.move_mask()            

    # blend
    print('Blending ...')
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE) 
    offset = offset_x, offset_y

    poisson_blend_result = poisson_edit(source, image_path, target_mask, offset)
    
    cv2.imwrite(path.join(out_dir, 'target_result.png'), poisson_blend_result)
    
    print('Done.\n')