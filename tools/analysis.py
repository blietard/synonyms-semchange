from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from tools import utils


def nearest_neighbors(k, word, matrix, word2idx, idx2word, dist_params=None):
    neigh = NearestNeighbors(n_neighbors=6, metric=utils.dist, metric_params=dist_params)
    neigh.fit(matrix)
    w = matrix[ word2idx[word] ]
    neighbors_inds = neigh.kneighbors(w, n_neighbors=6, return_distance=False)
    return [idx2word[i] for i in neighbors_inds[0][1:]]



class Analyser:
    '''
    Class to conduct analysis of results.
    '''
    def __init__(self, matrix1, matrix2, vocabulary):
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.index2word = dict()
        self.word2index = dict()
        for i,word in enumerate(list(vocabulary.words)):
            self.word2index[word]=i
            self.index2word[i]=word

    def get_nearest_neighbors(self, k, word, dist_params=None):
        n1 = nearest_neighbors(k, word, self.matrix1, self.word2index, self.index2word, dist_params=dist_params)
        n2 = nearest_neighbors(k, word, self.matrix2, self.word2index, self.index2word, dist_params=dist_params)
        return (n1, n2)

    def visual_ranking_errors(self, targets, gold_scores, distances):
        true_order = list(np.array(targets)[np.argsort(gold_scores)])
        pred_order = list(np.array(targets)[np.argsort(distances)] )
        true_ranks = np.empty(len(targets), dtype='int16')
        pred_ranks = np.empty(len(targets), dtype='int16')
        for i, target in enumerate( targets ):
            true_ranks[i] = true_order.index(target)
            pred_ranks[i] = pred_order.index(target)  

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.plot( true_ranks, true_ranks, label='objective', ls='-' , c='black', alpha=0.1)
        ax.scatter(true_ranks, pred_ranks)
        ax.set_xticks(true_ranks)
        ax.set_xticklabels(targets, rotation=90)
        plt.legend()
        plt.show()
    
    def visual_distances_errors(self, targets, gold_scores, distances):
        order = np.argsort(gold_scores)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.plot(gold_scores[order], distances[order])
        ax.set_xticks(gold_scores[order])
        ax.set_xticklabels(np.array(targets)[order], rotation=90)
        #ax.set_xscale('log')
        plt.show()