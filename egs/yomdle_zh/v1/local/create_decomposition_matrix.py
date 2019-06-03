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
            #Normal input method
            for key in keystrokes:
                if key not in graphemes:
                    graphemes[key] = num_graphemes
                    num_graphemes = num_graphemes + 1
            decomp_dict[character] = [graphemes[x] for x in keystrokes]
            #Handle cases where characters have same decomposition
            bag_of_features = [value+'_b' if index==0 else value+'_e' if index==len(keystrokes)-1 else value+'_m' for (index,value) in enumerate(keystrokes)]
            bag_of_features_list.append(set(bag_of_features))
            decomp_dict_disambig[character] = bag_of_features_list.count(set(bag_of_features)) - 1
            if bag_of_features_list.count(set(bag_of_features)) > num_disambig:
                num_disambig = bag_of_features_list.count(set(bag_of_features))


phone_dict = {}
num_phones = 0
num_pdfclass = 1
max_pdfclass = 1
total_characters = 0
prev_phone = ''
with open(args.phones_path, encoding='utf8') as f:
    for line in f:
        line = line.strip()
        phone = line.split()[0]
        num = line.split()[1]
        if phone not in decomp_dict:
            phone_dict[num] = [num_phones]
            num_phones = num_phones + 1
        if prev_phone == phone:
            num_pdfclass = num_pdfclass + 1
            if num_pdfclass > max_pdfclass:
                max_pdfclass = num_pdfclass
        else:
            num_pdfclass = 1
        prev_phone = phone
        total_characters = total_characters + 1

# plus one for the bias term for the fixed affine layer
affine_decomp = np.zeros([num_phones + 3*num_graphemes + num_disambig + max_pdfclass + 1, total_characters], dtype=float)
#affine_decomp = np.ones([num_phones + 3*num_graphemes + num_disambig + 1, total_characters], dtype=float)
#affine_decomp = affine_decomp * 0.01
prev_phone = ''
num_pdfclass = 0
with open(args.phones_path, encoding='utf8') as f:
    for line in f:
        line = line.strip()
        phone = line.split()[0]
        num = line.split()[1]
        indices = []
        if num in phone_dict:
            indices = phone_dict[num]
        elif phone in decomp_dict:
            #Adds position dependency
            indices = [(3*x + num_phones) for x in decomp_dict[phone]]
            indices = [(x + 1) for x in indices]
            indices[0] = indices[0] - 1
            indices[-1] = indices[-1] + 1
            #disambiguate
            indices.append(num_phones + 3*num_graphemes + decomp_dict_disambig[phone])
            if prev_phone == phone:
                num_pdfclass = num_pdfclass + 1
            else:
                prev_phone = phone
                num_pdfclass = 0
            indices.append(num_phones + 3*num_graphemes + num_disambig + num_pdfclass)
        for index in indices:
            affine_decomp[index, int(num)] = 1.0

decomp_dim_path = os.path.join(args.out_dir, 'decomp.dim')
decomp_mat_path = os.path.join(args.out_dir, 'decomp.mat')
dim_fh = open(decomp_dim_path, 'w', encoding='utf8')
mat_fh = sys.stdout
dim_fh.write(str(affine_decomp.shape[0]) + ' ' + str(affine_decomp.shape[1]) + '\n')
write_kaldi_matrix(mat_fh, np.transpose(affine_decomp))
