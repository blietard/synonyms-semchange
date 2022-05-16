import scipy
import numpy as np
import mangoes
import os
from tools.readers import SemEvalReader
from tools.utils import *
from tools.count_based import *

#============Full Pipeline==============
def static_SemEval_pipeline(data_folder: str, storage_folder: str, language: str, 
                    window_size: int, ppmi_alpha, ppmi_k, 
                    svd_dim: int,svd_niter: int, OP_func,
                    rng_seed):

    print('[START] Collecting data')
    reader = SemEvalReader(data_folder)
    targets, gold_scores = reader.read_targets(language)
    corpus1, corpus2 = reader.load_corpora(language,'lemma')
    folder=  storage_folder + '/{}/'.format(language)
    vocabulary, vocabs_len = shared_vocabulary(corpus1, corpus2, out_vocab_len=True)
    print(f'[INFO] Number of types: {vocabs_len[0]} in C1, {vocabs_len[1]} in C2, {vocabs_len[2]} shared')
    print('[INFO] Counting occurences...')
    counts_matrix1, counts_matrix2 = creates_count_matrices_pair(corpus1,corpus2,vocabulary,window_size)
    print('[INFO] Done.')
    word2index = dict()
    idx2word = dict()
    for i, word in enumerate(list(vocabulary.words)):
        word2index[word]=i
        idx2word[i] = word
    print('[INFO] Computing and scoring PPMI')
    create_ppmi_matrices_pair(counts_matrix1, counts_matrix2, ppmi_alpha, ppmi_k, storage_folder=folder+'ppmi', verbose=True)
    del counts_matrix1, counts_matrix2
    ppmi1, ppmi2 = load_ppmi_matrices_as_csr(folder+'ppmi')
    ppmi_score = reader.spearman_score(ppmi1,ppmi2,word2index,word2index,language,out=True)
    print(f'[SCORE PPMI] Spearman\'s rho: {ppmi_score[0]} \tp-value: {ppmi_score[1]}')
    print('[INFO] Computing SVD')
    create_svd_matrices_pair(ppmi1,ppmi2,folder+'svd',standardise=False, dim=svd_dim,gamma=0.0,random_state=rng_seed,n_iter=svd_niter, verbose=True)
    del ppmi1, ppmi2
    svd1, svd2 = load_svd_matrices(folder+'svd')
    print('[INFO] Aligning SVD with OP')
    svd1_std = OP_func(svd1)
    svd2_std = OP_func(svd2)
    del svd1, svd2
    W_align_SVD = OrthogProcrustAlign(svd1_std,svd2_std,standard=True, backward=True)
    svd1_std_aligned = svd1_std.dot(W_align_SVD)
    print('[INFO] Scoring SVD')
    svd_score = reader.spearman_score(svd1_std_aligned,svd2_std,word2index,word2index,language,out=True)
    del svd1_std,svd1_std_aligned,svd2_std
    print(f'[SCORE SVD] Spearman\'s rho: {svd_score[0]} \tp-value: {svd_score[1]}')
    print('[END] End of pipeline.')
    return (ppmi_score,svd_score)
    
#============Partial Pipelines==============
def prepare_SemEval_data(data_folder: str, storage_folder: str, language: str):
    print('[START] Collecting data')
    reader = SemEvalReader(data_folder)
    folder=  storage_folder + '/{}/'.format(language)
    corpus1, corpus2 = reader.load_corpora(language,'lemma')
    vocabulary, vocabs_len = shared_vocabulary(corpus1, corpus2, out_vocab_len=True)
    print(f'[INFO] Number of types: {vocabs_len[0]} in C1, {vocabs_len[1]} in C2, {vocabs_len[2]} shared')
    return (reader, folder, corpus1, corpus2, vocabulary)
    
def count_SemEval_occurences(corpus1,corpus2,vocabulary,window_size:int, folder: str, matrix_name=''):
    print('[INFO] Counting occurences...')
    counts_matrix1, counts_matrix2 = creates_count_matrices_pair(corpus1,corpus2,vocabulary,window_size)
    print('[INFO] Done.')
    if not matrix_name:
        matrix_name = 'count_ws'+str(window_size)
    if not os.path.exists(folder+'count/count1/'+matrix_name):
        os.mkdir(folder+'count/count1/'+matrix_name)
    if not os.path.exists(folder+'count/count2/'+matrix_name):
        os.mkdir(folder+'count/count2/'+matrix_name)
    counts_matrix1.save(folder+'count/count1/'+matrix_name)
    counts_matrix2.save(folder+'count/count2/'+matrix_name)
    
    return (counts_matrix1,counts_matrix2)

def compute_score_PPMI(counts_matrix1,counts_matrix2,ppmi_alpha,ppmi_k,word2index,folder,reader,language):
    print('[INFO] Computing and scoring PPMI')
    create_ppmi_matrices_pair(counts_matrix1, counts_matrix2, ppmi_alpha, ppmi_k, storage_folder=folder+'ppmi', verbose=True, )
    
    ppmi1, ppmi2 = load_ppmi_matrices_as_csr(folder+'ppmi')
    ppmi_score = reader.spearman_score(ppmi1,ppmi2,word2index,word2index,language,out=True)
    print(f'[SCORE PPMI] Spearman\'s rho: {ppmi_score[0]} \tp-value: {ppmi_score[1]}')
    return (ppmi1,ppmi2,ppmi_score)

def rename_and_clean_PPMIs(folder, matrix_name, move_to=''):
    os.rename(folder+'/ppmi1/matrix.npz',folder+'/ppmi1/'+move_to+matrix_name+'.npz')
    os.rename(folder+'/ppmi1/contexts_words.txt',folder+'/ppmi1/'+move_to+matrix_name+'_words.txt')
    os.remove(folder+'/ppmi1/words.txt')
    os.remove(folder+'/ppmi1/.metadata')
    os.rename(folder+'/ppmi2/matrix.npz',folder+'/ppmi2/'+move_to+matrix_name+'.npz')
    os.rename(folder+'/ppmi2/contexts_words.txt',folder+'/ppmi2/'+move_to+matrix_name+'_words.txt')
    os.remove(folder+'/ppmi2/words.txt')
    os.remove(folder+'/ppmi2/.metadata')

def compute_score_SVD(ppmi1,ppmi2,svd_dim,rng_seed,svd_niter , word2index, folder, reader, language, op_func, matrix_name, n_oversamples=10):
    create_svd_matrices_pair(ppmi1,ppmi2,folder+'svd',standardise=False, dim=svd_dim,gamma=0.0,random_state=rng_seed,n_iter=svd_niter, verbose=True, matrix_name=matrix_name, n_oversamples=n_oversamples)
    svd1, svd2 = load_svd_matrices(folder+'svd',matrix_name=matrix_name)
    print('[INFO] Aligning SVD with OP')
    svd1_std = op_func(svd1)
    svd2_std = op_func(svd2)
    del svd1, svd2
    W_align_SVD = OrthogProcrustAlign(svd1_std,svd2_std,standard=True, backward=True)
    svd1_std_aligned = svd1_std.dot(W_align_SVD)
    print('[INFO] Scoring SVD')
    svd_score = reader.spearman_score(svd1_std_aligned,svd2_std,word2index,word2index,language,out=True)
    del svd1_std,svd1_std_aligned,svd2_std
    print(f'[SCORE SVD] Spearman\'s rho: {svd_score[0]} \tp-value: {svd_score[1]}')
    return svd_score
