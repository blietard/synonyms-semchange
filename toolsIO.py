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
from params import TEMP_FOLDER, MATRIX_FOLDER, SGNS_FOLDER, WORDS_FOLDER, INFO_WORDS_FOLDER, NEIGHBORS_FOLDER, DISTANCES_FOLDER, CSV_FOLDER


def getRepr(repr_mode,decade,pos):
    if repr_mode.lower() == 'sgns':
        return np.load(f'{SGNS_FOLDER}/{decade}_{pos}.npy')
    raise ValueError('Unkown representation mode. Use "sgns".')

def getSynPairs(selection_mode,model_name,pos):
    with open(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/fernald_{pos}_{model_name}_{selection_mode.upper()}.pickle','rb') as f:
        syn_pairs = pickle.load(f)
    return syn_pairs

def getTargets(pos,repr_mode='doubNorm'):
    if pos not in ['ADJ','N','V']:
        raise ValueError(f'Invalid POS tag value ({pos}). Must be ADJ, N or V.')
    with open(f'{WORDS_FOLDER}/{pos}_list.pkl','rb') as f:
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


def getSemChange(pos,model_name,procrustes=False):
    if procrustes:
        return pd.read_csv(f'{INFO_WORDS_FOLDER}/{model_name}/{pos}_semchange_procrustes.csv',sep='\t',index_col='words')
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


def getOpenWordNetSyns(pos):
    with open(f'{WORDS_FOLDER}/openwordnet/synsets_by_ID_{pos}.pkl','rb') as f:
        d = pickle.load(f)
    synsets = d.values()
    synpairs = []
    for synset in synsets:
        for w1 in synset:
            for w2 in synset:
                if w1 != w2:
                    synpairs.append((w1,w2))
    return set(synpairs)

def getOpenWordNetSyns_asDict(pos):
    with open(f'{WORDS_FOLDER}/openwordnet/synsets_by_ID_{pos}.pkl','rb') as f:
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

def getOpenWordNetPolysemy(pos):
    return pd.read_csv(f'{WORDS_FOLDER}/openwordnet/words_{pos}.csv',sep='\t',index_col='words')


def getSynpairsInfos(pos,model_name):
    return pd.read_csv(f'{CSV_FOLDER}/{model_name}/{pos}_synpairs_analysis.csv',sep='\t',index_col='pairIdx',na_filter=False)

def getWordNetSynpairsInfos(pos,model_name):
    return pd.read_csv(f'{CSV_FOLDER}/{model_name}/{pos}_WNsynpairs_analysis.csv',sep='\t',index_col='pairIdx',na_filter=False)