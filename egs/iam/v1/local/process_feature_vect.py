#!/usr/bin/env python

import argparse
import os
import sys
import scipy.io as sio
import numpy as np
from scipy import misc

parser = argparse.ArgumentParser(description="""Generates and saves the feature vectors""")
parser.add_argument('database_path', type=str, help='path to downloaded data')
parser.add_argument('dir', type=str, help='output directory')
parser.add_argument('--dataset', type=str, default='trainset', 
                    choices=['trainset', 'testset', 'validationset1', 'validationset2'])
parser.add_argument('--out-ark', type=str, default='-', help='where to write the output feature file')
parser.add_argument('--scale-size', type=int, default=40, help='size to scale the height of all images')

args = parser.parse_args()

#Image dimensions
C = 1

def write_kaldi_matrix(file_handle, matrix, key):
    #file_handle.write("[ ")
    file_handle.write(key + " [ ")

    num_rows = len(matrix)
    if num_rows == 0:
        raise Exception("Matrix is empty")
    num_cols = len(matrix[0])

    for row_index in range(len(matrix)):
        if num_cols != len(matrix[row_index]):
            raise Exception("All the rows of a matrix are expected to "
                            "have the same length")
        file_handle.write(" ".join(map(lambda x: str(x), matrix[row_index])))
        if row_index != num_rows - 1:
            file_handle.write("\n")
    file_handle.write(" ]\n")

def zeropad(x, length):
    s = str(x)
    while len(s) < length:
        s = '0' + s
    return s

def get_scaled_image(im):
    scale_size = args.scale_size
    sx = im.shape[1]
    sy = im.shape[0]
    scale = (1.0 * scale_size) / sy
    im = misc.imresize(im, scale)
    
    return im

### main ###
data_list_path = os.path.join(args.database_path,
                              'largeWriterIndependentTextLineRecognitionTask',
                              args.dataset + '.txt')

if args.out_ark == '-':
    out_fh = sys.stdout
else:
    out_fh = open(args.out_ark,'wb')

image_ID = 1
with open(data_list_path) as f:
    for line in f:
        key = zeropad(image_ID, 8)
        line_vect = line.split('-')
        file = os.path.join(args.database_path,
                            'lines',
                            line_vect[0],
                            line_vect[0] + '-' + line_vect[1],
                            line.strip() + '.png')
        im = misc.imread(file)
        im_scale = get_scaled_image(im)
        data = np.transpose(im_scale, (1, 0))
        data = np.divide(data, 255.0)
        write_kaldi_matrix(out_fh, data, key)
        image_ID = image_ID + 1

