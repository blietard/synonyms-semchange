import os
import numpy as np
import argparse
from params import NEIGHBORS_FOLDER, DECADES, DISTANCES_FOLDER

parser = argparse.ArgumentParser(description='Compute the neighbordhoods for every decades.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['ADJ','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
args = parser.parse_args()
pos, repr_mode, distance = args.pos, args.repr, args.dist

model_name = repr_mode + '_' + distance

if not os.path.exists(f'{NEIGHBORS_FOLDER}/{model_name}/'):
    os.mkdir(f'{NEIGHBORS_FOLDER}/{model_name}/')

for decade in DECADES:
    print(f'[INFO] Starting computing neighborhoods for POS tag "{pos}" and decade {decade}.')
    distmat = np.load(f'{DISTANCES_FOLDER}/{model_name}/{decade}_{pos}.npy')
    nn_mat = distmat.argsort(axis=1)[:,1:]
    np.save( arr=nn_mat, file=f'{NEIGHBORS_FOLDER}/{model_name}/{decade}_{pos}.npy')
    print(f'[INFO] Finished for POS tag "{pos}" and decade {decade}.')

print(f'[INFO] Saved neighborhoods in "{NEIGHBORS_FOLDER}/{model_name}/".')
