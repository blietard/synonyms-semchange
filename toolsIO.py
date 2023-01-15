import numpy as np
import pandas as pd
import pickle
from mangoes.base import create_representation, CountBasedRepresentation
from mangoes.weighting import ShiftedPPMI
from mangoes.vocabulary import Vocabulary
from scipy.spatial.distance import cosine,euclidean
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from collections import defaultdict
from params import ALPHA, SHIFT ,TEMP_FOLDER, MATRIX_FOLDER, SGNS_FOLDER, WORDS_FOLDER, INFO_WORDS_FOLDER, NEIGHBORS_FOLDER, DISTANCES_FOLDER, CSV_FOLDER


def getRepr(repr_mode,decade,pos,wl=None,cl=None):
    if repr_mode.lower() == 'sgns':
        return np.load(f'{SGNS_FOLDER}/{decade}_{pos}-w.npy')
    L_mat = sp.load_npz(MATRIX_FOLDER+f'/cooc-matrix_{decade}_{pos}_L.npz')
    R_mat = sp.load_npz(MATRIX_FOLDER+f'/cooc-matrix_{decade}_{pos}_R.npz')
    if repr_mode.lower() == 'doubnorm':
        return normalize(normalize(sp.hstack([L_mat,R_mat]), axis=0,norm='l1'), axis=1,norm='l1').toarray()
    if repr_mode.lower() == 'sppmi':
        matrix = L_mat + R_mat
        matrix = CountBasedRepresentation(Vocabulary(wl), Vocabulary(cl), matrix)
        matrix = create_representation(matrix, weighting=ShiftedPPMI(alpha=ALPHA,shift=SHIFT))
        matrix.save(TEMP_FOLDER)
        with np.load(f'{TEMP_FOLDER}/matrix.npz') as loaded:
            matrix = sp.csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
        return normalize(matrix, axis=1,norm='l2').toarray()
    raise ValueError('Unkown representation mode. Use "sgns", "sppmi" or "doubNorm".')

def getSynPairs(selection_mode,model_name,pos):
    with open(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/fernald_{pos}_{model_name}_{selection_mode.upper()}.pickle','rb') as f:
        syn_pairs = pickle.load(f)
    return syn_pairs

def getTargets(pos,repr_mode='doubNorm'):
    if pos not in ['A','N','V']:
        raise ValueError(f'Invalid POS tag value ({pos}). Must be A, N or V.')
    if repr_mode.lower() == 'sgns':
        with open(f'{WORDS_FOLDER}/{pos}_list_sgns.pickle','rb') as f:
            word_list = pickle.load(f)
    else:
        with open(f'{WORDS_FOLDER}/{pos}_list.pickle','rb') as f:
            word_list = pickle.load(f)
    word2ind = {word : i for i,word in enumerate(word_list)}
    return word_list, word2ind

def getContexts():
    with open(f'{WORDS_FOLDER}/contexts_list.txt','r',encoding='utf-8') as f:
        contexts = f.read().split('\n')
    contexts = [context for context in contexts if context]
    return contexts

def getDist(distance):
    if distance.lower() == 'cosine':
        dist = cosine
        distance_name = 'cosine'
    elif distance.lower() == 'euclid':
        dist = euclidean
        distance_name = 'euclidean'
    else:
        raise ValueError('Unknown distance. Use "cosine" or "euclid".')
    return dist, distance_name

def getMatrixDist(distance):
    if distance.lower() == 'cosine':
        matdist = cosine_distances
    elif distance.lower() == 'euclid':
        matdist = euclidean_distances
    else:
        raise ValueError('Unknown distance. Use "cosine" or "euclid".')
    return matdist


def getSemChange(pos,model_name):
    return pd.read_csv(f'{INFO_WORDS_FOLDER}/{model_name}/{pos}_semchange.csv',sep='\t',index_col='words')

def getFreq(pos,model_name):
    return pd.read_csv(f'{INFO_WORDS_FOLDER}/{model_name}/{pos}_frequency.csv',sep='\t',index_col='words')

def getWordLabels(pos,model_name):
    return pd.read_csv(f'{INFO_WORDS_FOLDER}/{model_name}/{pos}_wordlabels.csv',sep='\t',index_col='words')

def getNeighbors(pos,model_name,decade):
    return np.load(f'{NEIGHBORS_FOLDER}/{model_name}/{decade}_{pos}.npy')

def getDistanceMatrix(pos,model_name,decade):
    return np.load(f'{DISTANCES_FOLDER}/{model_name}/{decade}_{pos}.npy')

def getWordNetSyns(pos):
    with open(f'{WORDS_FOLDER}/wordnet/synsets_by_ID_{pos}.pkl','rb') as f:
        d = pickle.load(f)
    synsets = d.values()
    synpairs = []
    for synset in synsets:
        for w1 in synset:
            for w2 in synset:
                if w1 != w2:
                    synpairs.append((w1,w2))
    return set(synpairs)

def getWordNetSyns_asDict(pos):
    with open(f'{WORDS_FOLDER}/wordnet/synsets_by_ID_{pos}.pkl','rb') as f:
        d = pickle.load(f)
    synsets = d.values()
    synlists_dict = defaultdict(set)
    for synset in synsets:
        for w1 in synset:
            for w2 in synset:
                if w1 != w2:
                    synlists_dict[w1] = synlists_dict[w1] | {w2}
                    synlists_dict[w2] = synlists_dict[w2] | {w1}   
    return synlists_dict

def getWordNetPolysemy(pos):
    return pd.read_csv(f'{WORDS_FOLDER}/wordnet/words_{pos}.csv',sep='\t',index_col='words')

def getSynpairsInfos(pos,model_name):
    return pd.read_csv(f'{CSV_FOLDER}/{model_name}/{pos}_synpairs_analysis.csv',sep='\t',index_col='pairIdx',na_filter=False)

def getWordNetSynpairsInfos(pos,model_name):
    return pd.read_csv(f'{CSV_FOLDER}/{model_name}/{pos}_WNsynpairs_analysis.csv',sep='\t',index_col='pairIdx',na_filter=False)