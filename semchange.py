import os
import numpy as np
import pandas as pd
import argparse

import toolsIO as io
from params import NEIGHBORS_FOLDER, DECADES, INFO_WORDS_FOLDER, K_NEIGHBORS
parser = argparse.ArgumentParser(description='Compute the semantic change for every decades.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['ADJ','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
parser.add_argument('--diachrdist', '-d', type=str, metavar='Diachronic dist mode', default='neighbors', nargs='?',
                    help='mode corresponding to the SC metric', choices=['neighbors', 'procrustes'])
args = parser.parse_args()
pos, repr_mode, distance , sc_mode = args.pos, args.repr, args.dist, args.diachrdist

model_name = repr_mode + '_' + distance
word_list, word2ind = io.getTargets(pos,repr_mode)
d_func, _ = io.getDist(distance)
semChange = pd.DataFrame(columns=DECADES,dtype='float16',index = word_list )
semChange.index.name = 'words'
origin_decade = DECADES[0]
semChange[origin_decade] = np.zeros(len(word_list))


def standardize(arr):
    return ( arr-arr.mean(0)  )/(arr.std(0) )

def unitcenter(arr):
    # the unit-center normalisation from
    # https://github.com/Garrafao/LSCDetection/blob/master/modules/embeddings.py
    arr_c = arr.copy()
    norms = np.sqrt(np.sum(arr_c**2, axis=1))
    norms[norms == 0] = 1
    arr_c /= norms[:, np.newaxis]
    avg = np.mean(arr_c, axis=0)
    arr_c -= avg
    return arr_c

def centerunit(arr):
    # same as unitcenter but operation order is reversed
    arr_c = arr.copy()
    avg = np.mean(arr_c, axis=0)
    arr_c -= avg
    norms = np.sqrt(np.sum(arr_c**2, axis=1))
    norms[norms == 0] = 1
    arr_c /= norms[:, np.newaxis]
    return arr_c


def OrthogProcrustAlign(arr1,arr2, standard=False, backward=False, std_func=standardize):
    '''
    Return Orthogonal Procrustes alignment matrix of arr1 and arr2.
    `standard` set to True if arr1 and arr2 are already standardized. Default is False.
    '''
    if standard:
        A = arr1
        B = arr2
    else:
        A = std_func(arr1)
        B = std_func(arr2)

    temp = B.T @ A
    U, e, Vt = np.linalg.svd(temp,)
    if backward:
        W = Vt.T @ U.T
    else:
        W = U @ Vt
    return W



if sc_mode == 'neighbors':
    nn_origin_array = np.load(f'{NEIGHBORS_FOLDER}/{model_name}/{origin_decade}_{pos}.npy')
    for decade in DECADES[1:]:
        nn_array = np.load(f'{NEIGHBORS_FOLDER}/{model_name}/{decade}_{pos}.npy')
        intersect = np.array([len(np.intersect1d(nn_array[i][:K_NEIGHBORS],nn_origin_array[i][:K_NEIGHBORS])) for i in range(nn_array.shape[0])])
        union = np.array([len(np.union1d(nn_array[i][:K_NEIGHBORS],nn_origin_array[i][:K_NEIGHBORS])) for i in range(nn_array.shape[0])])
        target_change = 1 - intersect/union
        print(decade,'Average change : ', target_change.mean().round(4))
        print(decade,'Std Dev. of change: ', target_change.std().round(4))
        semChange[ decade ] = target_change.copy()

    if not os.path.exists(f'{INFO_WORDS_FOLDER}/{model_name}/'):
        os.mkdir(f'{INFO_WORDS_FOLDER}/{model_name}/')

    semChange.to_csv(f'{INFO_WORDS_FOLDER}/{model_name}/{pos}_semchange.csv', sep='\t',index=True)

if sc_mode == 'procrustes':
    OriginMat = standardize(io.getRepr( repr_mode, DECADES[0], pos ))
    for decade in DECADES[1:]:
        TargetMat = standardize(io.getRepr( repr_mode, decade, pos ))
        R = OrthogProcrustAlign(OriginMat, TargetMat, standard=True)
        TargetMat_align = TargetMat.dot(R)
        target_change = np.array([ d_func(v_O, v_T) for v_O, v_T in zip(OriginMat,TargetMat_align) ])
        print(decade,'Average change : ', target_change.mean().round(4))
        print(decade,'Std Dev. of change: ', target_change.std().round(4))
        semChange[ decade ] = target_change.copy()

    if not os.path.exists(f'{INFO_WORDS_FOLDER}/{model_name}/'):
        os.mkdir(f'{INFO_WORDS_FOLDER}/{model_name}/')
    semChange.to_csv(f'{INFO_WORDS_FOLDER}/{model_name}/{pos}_semchange_procrustes.csv', sep='\t',index=True)
