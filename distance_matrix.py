import os
import numpy as np
import toolsIO as io
from params import DISTANCES_FOLDER, DECADES

import argparse
parser = argparse.ArgumentParser(description='Compute distances between words for every decades.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['A','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
args = parser.parse_args()
pos, repr_mode, distance = args.pos, args.repr, args.dist

model_name = repr_mode + '_' + distance
word_list, word2ind = io.getTargets(pos,repr_mode)
contexts = io.getContexts()
matdist = io.getMatrixDist(distance)

if not os.path.exists(f'{DISTANCES_FOLDER}/{model_name}/'):
    os.makedirs(f'{DISTANCES_FOLDER}/{model_name}/')

for d in DECADES:
    print(f'[INFO] Starting computing distances in {d} for POS tag "{pos}".')
    matrix = io.getRepr(repr_mode,int(d),pos,wl=word_list,cl=contexts)
    distance_matrix = matdist(matrix).round(8)
    np.save( arr=distance_matrix, file=f'{DISTANCES_FOLDER}/{model_name}/{d}_{pos}.npy' )
    print(f'[INFO] Finished computations and storage for decade {d}.')

print(f'[INFO] Saved distances in "{DISTANCES_FOLDER}/{model_name}/".')
