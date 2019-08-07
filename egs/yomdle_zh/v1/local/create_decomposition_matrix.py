#!/usr/bin/env python3

# Copyright      2019  Chun Chieh Chang

""" This script reads in the Chinese decompositions and outputs
    a bag of features matrix for the Chinese characters.
    This matrix will be used for a fixed affine layer in the
    neural network. The bag of features is created using the lexicon
    and nonsilence phones in the lang directory
    eg. local/create_decomposition_matrix.py \
            data/lang/phones.txt \
            data/local/dict/cj5-cc.txt
"""
import argparse
import os
import sys
import math
import numpy as np
from scipy import misc

parser = argparse.ArgumentParser(description="""Creates a bag of features with the lexicon and nonsilence phones""")
parser.add_argument('phones_path', type=str, help='Path to phones')
parser.add_argument('decomp_path', type=str, help='Path to the decomposition')
parser.add_argument('out_dir', type=str, default='decomp.mat', help='output file')
args = parser.parse_args()

def write_kaldi_matrix(f, matrix):
    """This function writes the matrix stored as a list of lists
    into 'output_file' in kaldi matrix text format.
    """
    f.write("[ ")
    num_rows = len(matrix)
    if num_rows == 0:
        raise Exception("Matrix is empty")
    num_cols = len(matrix[0])

    for row_index in range(len(matrix)):
        if num_cols != len(matrix[row_index]):
            raise Exception("All the rows of a matrix are expected to "
                            "have the same length")
        f.write(" ".join(map(lambda x: str(x), matrix[row_index])))
        if row_index != num_rows - 1:
            f.write("\n")
    f.write(" ] \n")

#Stores decompositions
decomp_dict = {}
#Stores unique graphemes from BPE
graphemes = {}
grapheme_count = {}
num_graphemes = 0
#Stores grapheme sequences to avoid repeats
#and disambiguate
#Uses space as delimiter between keystrokes
decomp_dict_disambig = {}
bag_of_features_list = []
num_disambig = 0
with open(args.decomp_path, 'r', encoding='utf8') as f:
    for line in f:
        line = line.strip()
        character = line.split()[-1]
        keystrokes = line.split()[:-1]
        if not ''.join(keystrokes).startswith('yyy') and not ''.join(keystrokes).startswith('z'):
            #Input method
            for key in keystrokes:
                if key not in graphemes:
                    graphemes[key] = num_graphemes
                    num_graphemes = num_graphemes + 1
                    grapheme_count[graphemes[key]] = 0
                grapheme_count[graphemes[key]] = grapheme_count[graphemes[key]] + 1
            decomp_dict[character] = [graphemes[x] for x in keystrokes]
            #Handle cases where characters have same decomposition
            bag_of_features = [value for (index,value) in enumerate(keystrokes)]
            bag_of_features_list.append(set(bag_of_features))
            decomp_dict_disambig[character] = bag_of_features_list.count(set(bag_of_features)) - 1
            if bag_of_features_list.count(set(bag_of_features)) > num_disambig:
                num_disambig = bag_of_features_list.count(set(bag_of_features))


phone_dict = {}
num_phones = 0
num_pdfclass = 0
max_pdfclass = 0
total_pdf = 0
prev_phone = ''
with open(args.phones_path, encoding='utf8') as f:
    for line in f:
        line = line.strip()
        phone = line.split()[0]
        num = line.split()[1]
        if phone not in phone_dict and phone not in decomp_dict:
            phone_dict[phone] = [num_phones]
            num_phones = num_phones + 1
        if prev_phone == phone:
            num_pdfclass = num_pdfclass + 1
            if num_pdfclass > max_pdfclass:
                max_pdfclass = num_pdfclass
        else:
            num_pdfclass = 0
        total_pdf = total_pdf + 1
        prev_phone = phone

# plus one for the bias term for the fixed affine layer
#affine_decomp = np.zeros([num_phones + num_graphemes + num_disambig + max_pdfclass + 2, total_pdf], dtype=float)
affine_decomp = np.random.rand(num_phones + num_graphemes + num_disambig + max_pdfclass + 2, total_pdf)
affine_decomp = affine_decomp * 0.001
prev_phone = ''
num_pdfclass = 0
with open(args.phones_path, encoding='utf8') as f:
    for line in f:
        line = line.strip()
        phone = line.split()[0]
        num = line.split()[1]
        if phone in phone_dict:
            affine_decomp[phone_dict[phone], int(num)] = 1.0
        elif phone in decomp_dict:
            for index in decomp_dict[phone]:
                affine_decomp[index + num_phones, int(num)] = 1.0 / math.sqrt(grapheme_count[index])
            affine_decomp[num_phones + num_graphemes + decomp_dict_disambig[phone], int(num)] = 1.0
        if prev_phone == phone:
            num_pdfclass = num_pdfclass + 1
        else:
            prev_phone = phone
            num_pdfclass = 0
        affine_decomp[num_phones + num_graphemes + num_disambig + num_pdfclass, int(num)] = 1.0

#print(phone_dict)
#print(affine_decomp.shape)
#print(affine_decomp.T)
#print(len(np.unique(affine_decomp.T, axis=0)))
#print(len(np.unique(affine_decomp, axis=0)))

decomp_dim_path = os.path.join(args.out_dir, 'decomp.dim')
decomp_mat_path = os.path.join(args.out_dir, 'decomp.mat')
dim_fh = open(decomp_dim_path, 'w', encoding='utf8')
mat_fh = sys.stdout
np.save(os.path.join(args.out_dir, 'decomp'), np.transpose(affine_decomp))
dim_fh.write(str(affine_decomp.shape[0]) + ' ' + str(affine_decomp.shape[1]) + '\n')
write_kaldi_matrix(mat_fh, np.transpose(affine_decomp))
