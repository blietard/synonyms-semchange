"""
Module for reading datasets
"""
from collections import defaultdict
from scipy.stats import spearmanr
from tools.utils import dist
import mangoes
import numpy as np

class Reader():
    '''
    Base class for any dataset reader
    '''

    def __init__(self, path: str):
        self.path = path
        self.targets = None

    def load_2periods_corpora(self, corpus1_path, corpus2_path, verbose=True):
        '''
        Return mangoes.corpus.Corpus object for a pair of corpora.
        '''
        if verbose:
            print('[INFO] Building corpus 1...')
        corpus1 = mangoes.Corpus(corpus1_path)
        if verbose:
            print(f"[INFO] Corpus 1: {corpus1.nb_sentences} sentences \
                \t{len(corpus1._words_count)} words")
            print('[INFO] Building corpus 2...')
        corpus2 = mangoes.Corpus(corpus2_path)
        if verbose:
            print(f"[INFO] Corpus 2: {corpus2.nb_sentences} sentences \
                \t{len(corpus2._words_count)} words")
        return (corpus1, corpus2)

    def read_targets(self, language: str, out=True):
        raise NotImplementedError("read_targets method was \
                                  not implemented for this reader.")

    def get_spearman_score(self, matrix1, matrix2, word2idx_dict1, word2idx_dict2,
                       targets, gold_scores, out=True, nan_replace=1.0):
        distances = np.empty(len(targets))
        for i, word in enumerate(targets):
            idx1 = word2idx_dict1[word]
            idx2 = word2idx_dict2[word]
            d = dist(matrix1[idx1], matrix2[idx2])
            if np.isnan(d):
                print(f'[WARNING] Null vector encountered for word "{word}".')
                distances[i] = nan_replace
            else:
                distances[i] = d

        rho, p = spearmanr(distances, gold_scores)
        if out:
            return (rho.round(5), p.round(4))
        else:
            print(f'Spearman\'s rho: {rho.round(5)} \tp-value: {p.round(4)}')

    def get_distances(self, matrix1, matrix2, word2idx_dict1, word2idx_dict2,
                targets, nan_replace=1.0):
        distances = np.empty(len(targets))
        for i, word in enumerate(targets):
            idx1 = word2idx_dict1[word]
            idx2 = word2idx_dict2[word]
            d = dist(matrix1[idx1], matrix2[idx2])
            if np.isnan(d):
                print(f'[WARNING] Null vector encountered for word "{word}".')
                distances[i] = nan_replace
            else:
                distances[i] = d
        return distances

class SemEvalReader(Reader):
    '''
    Reader for SemEval2020 Task 1 (subtask 2) data.

    Parameters:
    > 'path': str
        Path to find the data repository.
    '''

    def __init__(self, path: str):
        super().__init__(path)
        self.targets = defaultdict(list)
        self.gold_scores = defaultdict(np.array)

    def load_corpora(self, language: str, subcorpus=None, verbose=True):
        '''
        Return mangoes.corpus.Corpus object for both corpora
        of the selected language. If subcorpus is provided,
        will read `self.path/language/corpus{X}/{subcorpus}/`
        '''
        if subcorpus is None:
            path_end = ''
        else:
            path_end = subcorpus+'/'

        return super().load_2periods_corpora(f'{self.path}/{language}/corpus1/'+path_end,
                                             f'{self.path}/{language}/corpus2/'+path_end,
                                             verbose)

    def read_targets(self, language: str, out=True):
        '''
        Return the list of target words and the numpy array of corresponding gold_scores.
        '''
        with open(f'{self.path}/{language}/truth/graded.txt', 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        lines.pop(-1)
        targets = [line.split()[0] for line in lines]
        gold_scores = np.array([line.split()[1] for line in lines], dtype='float64')
        self.targets[language] = targets.copy()
        self.gold_scores[language] = gold_scores.copy()
        if out:
            return (targets, gold_scores)

    def spearman_score(self, matrix1, matrix2, word2idx_dict1, word2idx_dict2,
                       language, out=True, nan_replace=1.0):
        targets = self.targets[language]
        gold_scores = self.gold_scores[language]
        score = super().get_spearman_score(matrix1, matrix2, word2idx_dict1,
                                       word2idx_dict2, targets, gold_scores,
                                       out, nan_replace)
        if out:
            return score

    def predict(self, matrix1, matrix2, word2idx_dict1, word2idx_dict2,
                language, nan_replace=1.0):
        targets = self.targets[language]
        return super().get_distances(matrix1, matrix2, word2idx_dict1,
                        word2idx_dict2, targets, nan_replace)
