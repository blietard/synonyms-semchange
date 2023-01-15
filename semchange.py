import os
import numpy as np
import pandas as pd
import argparse

import toolsIO as io
from params import NEIGHBORS_FOLDER, DECADES, INFO_WORDS_FOLDER, K_NEIGHBORS
parser = argparse.ArgumentParser(description='Compute the semantic change for every decades.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['A','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
parser.add_argument('--diachrdist', '-d', type=str, metavar='Diachronic dist mode', default='neighbors', nargs='?',
                    help='mode corresponding to the SC metric', choices=['neighbors','contexts'])
args = parser.parse_args()
pos, repr_mode, distance , sc_mode = args.pos, args.repr, args.dist, args.diachrdist

model_name = repr_mode + '_' + distance
word_list, word2ind = io.getTargets(pos,repr_mode)
contexts = io.getContexts()

semChange = pd.DataFrame(columns=DECADES,dtype='float16',index = word_list )
semChange.index.name = 'words'
origin_decade = DECADES[0]
semChange[origin_decade] = np.zeros(len(word_list))


if sc_mode == 'neighbors':
    nn_origin_array = np.load(f'{NEIGHBORS_FOLDER}/{model_name}/{origin_decade}_{pos}.npy')
    for decade in DECADES[1:]:
        nn_array = np.load(f'{NEIGHBORS_FOLDER}/{model_name}/{decade}_{pos}.npy')
        intersect = np.array([len(np.intersect1d(nn_array[i][:K_NEIGHBORS],nn_origin_array[i][:K_NEIGHBORS])) for i in range(nn_array.shape[0])])
        union = np.array([len(np.union1d(nn_array[i][:K_NEIGHBORS],nn_origin_array[i][:K_NEIGHBORS])) for i in range(nn_array.shape[0])])
        target_change = 1 - intersect/union
        #target_change = 1 - intersect/K_NEIGHBORS
        print(decade,'Average change : ', target_change.mean().round(4))
        print(decade,'Std Dev. of change: ', target_change.std().round(4))
        semChange[ decade ] = target_change.copy()

    if not os.path.exists(f'{INFO_WORDS_FOLDER}/{model_name}/'):
        os.mkdir(f'{INFO_WORDS_FOLDER}/{model_name}/')

    semChange.to_csv(f'{INFO_WORDS_FOLDER}/{model_name}/{pos}_semchange.csv', sep='\t',index=True)

if sc_mode == 'contexts':
    raise NotImplementedError()
