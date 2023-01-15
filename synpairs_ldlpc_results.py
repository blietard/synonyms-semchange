from tqdm import tqdm
import scipy.stats as stats
import numpy as np
import toolsIO as io
from params import DECADES

import argparse
parser = argparse.ArgumentParser(description='Compute the proportion of LD/LPC.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['A','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
parser.add_argument('--select', '-s', type=str, default='avg', nargs='?',
                    help='synonyms SC selection threshold', choices=['avg','avgstd'])
parser.add_argument('--decision', '-d', type=str, default='avgstd', nargs='?',
                    help='synonyms differentiation decision threshold', choices=['avg','avgstd'])
args = parser.parse_args()
pos, repr_mode, distance, selection_mode , decision_mode = args.pos, args.repr, args.dist, args.select, args.decision



model_name = repr_mode + '_' + distance
syn_pairs = io.getSynPairs(selection_mode,model_name,pos)
word_list, word2ind = io.getTargets(pos,repr_mode)
distances_origin = io.getDistanceMatrix( pos, model_name, DECADES[0] )
distances_end = io.getDistanceMatrix( pos, model_name, DECADES[-1] )

syn_distances_origin = []
syn_distances_end = []
for i,(s1,s2) in tqdm(enumerate(syn_pairs), desc="Synonyms", total=len(syn_pairs)):
    ind1 = word2ind[s1]
    ind2 = word2ind[s2]
    syn_distances_origin.append( distances_origin[ind1,ind2] )
    syn_distances_end.append( distances_end[ind1,ind2] )
syn_distances_origin = np.array(syn_distances_origin)
syn_distances_end = np.array(syn_distances_end)

div_syns = syn_distances_end-syn_distances_origin
div_pop = distances_end-distances_origin


diverge_prop_pop = (distances_end-distances_origin > 0).mean()
diverge_prop_syns = (syn_distances_end-syn_distances_origin > 0).sum()/len(syn_pairs)
print('======= SPATIAL BEHAVIOR =======')
print(f'General divergence rate : {diverge_prop_pop.round(3)*100}%')
print(f'Synonyms divergence rate : {diverge_prop_syns.round(3)*100}%')
print(' Difference of divergence pvalue:', str(stats.ttest_ind( (distances_end-distances_origin).flatten() , syn_distances_end - syn_distances_origin, alternative='two-sided').pvalue))


print('Computing results...')


if decision_mode.lower()=='avg':
    threshold = div_pop.mean()
elif decision_mode.lower()=='avgstd':
    threshold = div_pop.mean() + div_pop.std()
else:
    raise ValueError('Unknown decision mode. use "avg", "avgstd".')

ld_prop = ( div_syns > threshold ).sum()/len(div_syns)
print('\n======= RESULTS =======')
print('\n------ PROPORTIONS ------')
print(f'Law of differentiation: {ld_prop.round(3)*100}% \t Law of parallel change: {(1-ld_prop).round(3)*100}%')

result_binom_test = stats.binomtest( k =  ( div_syns > threshold ).sum(), n = len(syn_distances_end), alternative='greater' )
print('\n------ TEST LD > LPC ------')
print('Test: \tH0: "LPC/LD are even." \tH1: "LD more frequent than LPC."')
print(f'p-value:\t {result_binom_test.pvalue}\nCI of LD:\t{result_binom_test.proportion_ci()}')

result_binom_test = stats.binomtest( k =  ( div_syns > threshold ).sum(), n = len(syn_distances_end), alternative='less' )
print('\n------ TEST LD < LPC ------')
print('Test: \tH0: "LPC/LD are even." \tH1: "LPC more frequent than LD."')
print(f'p-value:\t {result_binom_test.pvalue}\nCI of LD:\t{result_binom_test.proportion_ci()}')