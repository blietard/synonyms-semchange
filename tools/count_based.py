import scipy
import numpy as np
import mangoes
from sklearn.utils.extmath import randomized_svd
from tools.utils import standardize


#============COUNT==============


def create_count_matrix(corpus : mangoes.corpus.Corpus, vocabulary : mangoes.vocabulary.Vocabulary, window_size : int):
    return mangoes.counting.count_cooccurrence(corpus, vocabulary, context = mangoes.context.Window(vocabulary=vocabulary,size=window_size))


def creates_count_matrices_pair(corpus1 : mangoes.corpus.Corpus, corpus2: mangoes.corpus.Corpus, shared_vocabulary: mangoes.vocabulary.Vocabulary, window_size=10, verbose=True):
    '''
    Return co-occurences matrices for both corpora given their shared vocabulary.

    Output:
    > matrix1, the co-occurence matrix of the first corpus
    > matrix2, the co-occurence matrix of the second corpus
    '''
    
    if verbose:
        print('[INFO] Computing count matrix for corpus 1...')
    matrix1 = create_count_matrix(corpus1, shared_vocabulary, window_size)
    if verbose:
        print('[INFO] Success!')
        print('[INFO] Computing count matrix for corpus 2...')
    matrix2 = create_count_matrix(corpus2, shared_vocabulary, window_size)
    if verbose:
        print('[INFO] Success!')
    return (matrix1,matrix2)

def load_count_matrices(storage_folder: str, matrix_name: str, vocabulary, word2index):
    c1 = mangoes.base.CountBasedRepresentation.load(f'{storage_folder}/count1/{matrix_name}')    
    c2 = mangoes.base.CountBasedRepresentation.load(f'{storage_folder}/count2/{matrix_name}')
    if c1.words != c2.words:
        raise ValueError('Vocabularies of the 2 matrices are not matching!')
    if c1.words != vocabulary:
        print('[WARNING] Vocabulary have been changed when loading count matrices.')
        #recreate word2index
        word2index = dict()
        for i, word in enumerate(list(c1.words.words)):
            word2index[word]=i
    return (c1, c2, c1.words, word2index)

#============PPMI==============

def create_ppmi_matrix(counts_matrix, alpha, k):
    return mangoes.create_representation(counts_matrix, weighting=mangoes.weighting.ShiftedPPMI(alpha=alpha,shift=k))

def create_ppmi_matrices_pair(counts_matrix1, counts_matrix2, alpha, k, storage_folder : str, verbose=True):
    if verbose:
        print(f'[INFO] Computing PPMI matrices with alpha={alpha} and k={k}.')
        print('[INFO] Computing PPMI matrix for Corpus 1...')

    ppmi1 = create_ppmi_matrix(counts_matrix1, alpha, k)
    if verbose:
        print('[INFO] Success!')
        print('[INFO] Computing PPMI matrix for Corpus 2...')
    ppmi2 = create_ppmi_matrix(counts_matrix2, alpha, k)
    if verbose:
        print('[INFO] Success!')
    ppmi1.save(storage_folder+'/ppmi1')
    ppmi2.save(storage_folder+'/ppmi2')
    if verbose:
        print(f'[INFO] Matrices stored in {storage_folder}/.')

def load_ppmi_matrices_as_csr(storage_folder: str, vocabulary=None,matrix_name=None):
    check_voc = True
    if matrix_name is None:
        check_voc = False
        matrix_name='matrix'
    else:
        vocab_name = matrix_name+'_words'
    with np.load(f'{storage_folder}/ppmi1/{matrix_name}.npz') as loaded:
        ppmi1_matrix = scipy.sparse.csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
    with np.load(f'{storage_folder}/ppmi2/{matrix_name}.npz') as loaded:
        ppmi2_matrix = scipy.sparse.csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
    
    if check_voc:
        v = mangoes.vocabulary.Vocabulary.load(f'{storage_folder}/ppmi1/',vocab_name)
        #recreate word2index
        word2index = dict()
        for i, word in enumerate(list(v.words)):
            word2index[word]=i
        return (ppmi1_matrix, ppmi2_matrix, v, word2index)
    return (ppmi1_matrix,ppmi2_matrix)

#============SVD==============

def compute_SVD_representation(matrix,dim=100,gamma=1.0,random_state=None,n_oversamples=10, n_iter=5):
    '''
    Compute U*Sigma^gamma representations from the truncated SVD of matrix.
    '''
    u, s, v = randomized_svd(matrix, n_components=dim, n_oversamples=n_oversamples ,n_iter=n_iter, transpose=False,random_state=random_state)
    if gamma == 0.0:
        matrix_reduced = u
    elif gamma == 1.0:
        matrix_reduced = s * u
    else:
        matrix_reduced = np.power(s, gamma) * u
    return matrix_reduced

def create_svd_matrices_pair(matrix1,matrix2, storage_folder : str, standardise=True, dim=100,gamma=1.0,random_state=None,n_iter=5, n_oversamples=10, verbose=True, matrix_name=None):
    '''
    If `standardise` is True, gamma is ignored as cancelled by the standardisation process.
    '''
    if matrix_name is None:
        matrix_name = 'matrix'
    if verbose:
        print(f'[INFO] Computing {"standardised "*standardise}SVD matrices with gamma={gamma} and d={dim}.')
        print('[INFO] Computing SVD matrix for Corpus 1...')
    if standardise:
        svd1 = standardize(compute_SVD_representation(matrix1,dim,0,random_state=random_state,n_iter=n_iter))
        if verbose:
            print('[INFO] Success!')
            print('[INFO] Computing SVD matrix for Corpus 2...')
        svd2 = standardize(compute_SVD_representation(matrix2,dim,0,random_state=random_state,n_iter=n_iter))
        if verbose:
            print('[INFO] Success!')
    else:
        svd1 = compute_SVD_representation(matrix1,dim,gamma,random_state=random_state,n_iter=n_iter, n_oversamples=n_oversamples)
        if verbose:
            print('[INFO] Success!')
            print('[INFO] Computing SVD matrix for Corpus 2...')
        svd2 = compute_SVD_representation(matrix2,dim,gamma,random_state=random_state,n_iter=n_iter, n_oversamples=n_oversamples)
        if verbose:
            print('[INFO] Success!')
    np.save(storage_folder+'/svd1/'+matrix_name,svd1)
    np.save(storage_folder+'/svd2/'+matrix_name,svd2)
    if verbose:
        print(f'[INFO] Matrices stored in {storage_folder}/.')

def load_svd_matrices(storage_folder: str, is_txt=False, matrix_name=None):
    check_voc = True
    if matrix_name is None:
        check_voc = False
        matrix_name='matrix'
    else:
        vocab_name = matrix_name+'_words'
    if is_txt:
        matrix_array = np.loadtxt(f'{storage_folder}/svd1/{matrix_name}.txt', dtype=object, comments=None, delimiter=' ', skiprows=1, encoding='utf-8')
        svd1 = matrix_array[:,1:].astype(np.float)
        matrix_array = np.loadtxt(f'{storage_folder}/svd2/{matrix_name}.txt', dtype=object, comments=None, delimiter=' ', skiprows=1, encoding='utf-8')
        svd2 = matrix_array[:,1:].astype(np.float)
    else:
        svd1 =  np.load(f'{storage_folder}/svd1/{matrix_name}.npy')
        svd2 =  np.load(f'{storage_folder}/svd2/{matrix_name}.npy')
    return (svd1, svd2)
