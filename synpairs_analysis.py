import pandas as pd 
import numpy as np
from params import WORDS_FOLDER, DECADES, CSV_FOLDER, MIN_FREQ, NEIGHBORS_FOLDER, SND_K_RANGE
import toolsIO as io 
import os
import pickle
from tqdm import tqdm
import argparse
import Levenshtein as lev
from nltk.corpus import wordnet31 as wn



def k_jaccard_distance(u,v,k):
    intersect = len(np.intersect1d(u[:k],v[:k]))
    union = len(np.union1d(u[:k],v[:k]))
    return 1 - intersect/union

wn_pos_mapper = {'ADJ':wn.ADJ,'N':wn.NOUN,'V':wn.VERB,'ALL':None}

parser = argparse.ArgumentParser(description='Select pairs of synonyms.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['ADJ','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
args = parser.parse_args()
pos, repr_mode, distance = args.pos, args.repr, args.dist

model_name = repr_mode + '_' + distance
word_list, word2ind = io.getTargets(pos,repr_mode)
contexts = io.getContexts()
words_labels = io.getWordLabels(pos,model_name)
semchanges = io.getSemChange(pos, model_name)
semchanges_OP = io.getSemChange(pos, model_name, procrustes=True)
origin_neighbors = io.getNeighbors(pos, model_name, DECADES[0])
end_neighbors = io.getNeighbors(pos, model_name, DECADES[-1])
wnpos = wn_pos_mapper[pos]


with open(f'{WORDS_FOLDER}/source_synonyms.pkl','rb') as f:
    syns_dict = pickle.load(f)[pos]

if not os.path.exists(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/'):
    os.makedirs(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/')


candidate_syns_pairs = []
full_number_of_pairs = 0
for entry, syns in syns_dict.items():
    for syn in syns:
        full_number_of_pairs += 1
        if (entry in word_list) and (syn in word_list):
            candidate_syns_pairs.append( (entry,syn, len(syns)) )

print('[STATS:] Number of pairs : ',len(candidate_syns_pairs))

df = pd.DataFrame(columns=['entry','syn','pressure_entry'],data=candidate_syns_pairs)

wn_distances = []
wn_senses_entry_pos = []
wn_pressure_entry_pos = []
wn_senses_syn_pos = []
wn_senses_entry = []
wn_pressure_entry = []
wn_senses_syn = []
wn_hypernyms = []
wn_antonyms = []
wn_holonyms = []

for e, s in tqdm(df[['entry','syn']].values, desc='WN pairs info'):    
    synsets_e_all = wn.synsets(e, pos=None)
    synsets_s_all = wn.synsets(s, pos=None)
    wn_senses_entry.append(len(synsets_e_all))
    wn_senses_syn.append(len(synsets_s_all))
    wn_pressure_entry.append(sum([len(synset.lemmas())-1 for synset in synsets_e_all]))
    
    synsets_e = wn.synsets(e, pos=wnpos)
    synsets_s = wn.synsets(s, pos=wnpos)
    wn_senses_entry_pos.append(len(synsets_e))
    wn_senses_syn_pos.append(len(synsets_s))
    wn_pressure_entry_pos.append(sum([len(synset.lemmas())-1 for synset in synsets_e]))
    
    min_d = np.inf
    are_hypernyms = False
    are_antonyms = False
    are_holonyms = False
    for e_set in synsets_e:
        if min_d == 0:
            break
        for s_set in synsets_s:
            lowest_hyps = e_set.lowest_common_hypernyms(s_set)
            for hyp in lowest_hyps:
                if hyp == s_set or hyp == e_set:
                    are_hypernyms = True
            holonyms = e_set.member_holonyms()
            holonyms += e_set.substance_holonyms()
            holonyms += e_set.part_holonyms()
            meronyms = e_set.member_meronyms()
            meronyms += e_set.substance_meronyms()
            meronyms += e_set.part_meronyms()
            if s_set in holonyms or s_set in meronyms:
                are_holonyms = True
            
            d = e_set.shortest_path_distance(s_set)
            if d == 0:
                min_d = 0
                break
            if d is not None:
                if  d < min_d:
                    min_d = d
    
        lemmas = e_set.lemmas()
        for lemma in lemmas:
            if are_antonyms:
                break
            for antonym in lemma.antonyms():
                if antonym.name()==s:
                    are_antonyms = True
                    break
    
    
    wn_distances.append(min_d)
    if min_d == 0:
        are_hypernyms = False
        are_holonyms = False
    wn_hypernyms.append(are_hypernyms)
    wn_antonyms.append(are_antonyms)
    wn_holonyms.append(are_holonyms)
    
    

df['WN_distance'] = wn_distances
df['syns_WordNet'] = df.WN_distance == 0
df['diff_WordNet'] = ~ df.syns_WordNet
df['WN_hypernymy'] = wn_hypernyms
df['WN_antonymy'] = wn_antonyms
df['WN_holonymy'] = wn_holonyms

df['WNpressure_entry'] = wn_pressure_entry
df['WNsenses_entry'] = wn_senses_entry
df['WNsenses_syn'] = wn_senses_syn
df['WNpressure_POS_entry'] = wn_pressure_entry_pos
df['WNsenses_POS_entry'] = wn_senses_entry_pos
df['WNsenses_POS_syn'] = wn_senses_syn_pos


# wn_synpairs = io.getWordNetSyns(pos)
# df['syns_WordNet'] = [pair in wn_synpairs for pair in list(df[['entry','syn']].itertuples(index=False,name=None))]
# df['diff_WordNet'] = ~df.syns_WordNet

# own_synpairs = io.getOpenWordNetSyns(pos)
# df['syns_OpenWordNet'] = [pair in own_synpairs for pair in list(df[['entry','syn']].itertuples(index=False,name=None))]
# df['diff_OpenWordNet'] = ~df.syns_OpenWordNet

# WNsyndict = io.getWordNetSyns_asDict(pos)
# def areSynsExtended(w1,w2,areSynsWN,k=10,debug=False):
#     if areSynsWN:
#         if debug:
#             return (True,'')
#         else:
#             return True
#     w1_synsWN = {synWN for synWN in WNsyndict[w1] if synWN in word_list}
#     w2_synsWN = {synWN for synWN in WNsyndict[w2] if synWN in word_list}

#     # Search in neighborhood of w1
#     end_neigh_w1 = np.array(word_list)[ end_neighbors[word2ind[w1]] ][:k]
#     if w2 in end_neigh_w1:
#         for wordX in w1_synsWN:
#             end_neigh_X = np.array(word_list)[ end_neighbors[word2ind[wordX]] ][:k]
#             if w2 in end_neigh_X:
#                 if debug:
#                     return (True,wordX)
#                 else:
#                     return True

#     # Search in neighborhood of w2
#     end_neigh_w2 = np.array(word_list)[ end_neighbors[word2ind[w2]] ][:k]
#     if w1 in end_neigh_w2:
#         for wordX in w2_synsWN:
#             end_neigh_X = np.array(word_list)[ end_neighbors[word2ind[wordX]] ][:k]
#             if w1 in end_neigh_X:
#                 if debug:
#                     return (True,wordX)
#                 else:
#                     return True
    
#     if debug:
#         return (False,'')
#     else:
#         return False
# syns_WNExt = []
# WNExt_words = []
# for entry,syn,areSynsWN in df[['entry','syn','syns_WordNet']].values:    
#     syns_ext_bool, wordX = areSynsExtended(entry,syn,areSynsWN,k=20,debug=True)
#     syns_WNExt.append(syns_ext_bool)
#     WNExt_words.append(wordX)
# df['syns_WNExt'] = syns_WNExt
# df['WNExt_word'] = WNExt_words
# df['diff_WNExt'] = ~df.syns_WNExt

# wn_polysemy = io.getWordNetPolysemy(pos)
# nb_senses1 = []
# nb_senses2 = []
# pressure_wordnet = list()
# for pair in list(df[['entry','syn']].itertuples(index=False,name=None)):
#     try:
#         nb_senses1.append( wn_polysemy['nb_senses'].loc[pair[0]] )
#         pressure_wordnet.append( wn_polysemy['nb_syns'].loc[pair[0]] )
#     except KeyError:
#         nb_senses1.append(0)
#         pressure_wordnet.append(0)
#     try:
#         nb_senses2.append( wn_polysemy['nb_senses'].loc[pair[1]] )
#     except KeyError:
#         nb_senses2.append(0)
# df['WNpressure_entry'] = pressure_wordnet
# df['WNsenses_entry'] = nb_senses1
# df['WNsenses_syn'] = nb_senses2

# own_polysemy = io.getOpenWordNetPolysemy(pos)
# nb_senses1 = []
# nb_senses2 = []
# pressure_openwordnet = list()
# for pair in list(df[['entry','syn']].itertuples(index=False,name=None)):
#     try:
#         nb_senses1.append( own_polysemy['nb_senses'].loc[pair[0]] )
#         pressure_openwordnet.append( own_polysemy['nb_syns'].loc[pair[0]] )
#     except KeyError:
#         nb_senses1.append(0)
#         pressure_openwordnet.append(0)
#     try:
#         nb_senses2.append( own_polysemy['nb_senses'].loc[pair[1]] )
#     except KeyError:
#         nb_senses2.append(0)
# df['OWNpressure_entry'] = pressure_openwordnet
# df['OWNsenses_entry'] = nb_senses1
# df['OWNsenses_syn'] = nb_senses2


df['DD_N_entry'] = semchanges.loc[df.entry][str(DECADES[-1])].values
df['DD_N_syn'] = semchanges.loc[df.syn][str(DECADES[-1])].values
df['DDG_N_entry'] = words_labels.loc[df.entry]['DDG_N'].values
df['DDG_N_syn'] = words_labels.loc[df.syn]['DDG_N'].values

df['DDG_N_avg_entry'] = words_labels.loc[df.entry]['DDG_N_avg'].values
df['DDG_N_avg_syn'] = words_labels.loc[df.syn]['DDG_N_avg'].values
df['DDG_N_avg_one'] = df.DDG_N_avg_entry | df.DDG_N_avg_syn
df['DDG_N_avg_both'] = df.DDG_N_avg_entry & df.DDG_N_avg_syn

df['DDG_N_avgstd_entry'] = words_labels.loc[df.entry]['DDG_N_avgstd'].values
df['DDG_N_avgstd_syn'] = words_labels.loc[df.syn]['DDG_N_avgstd'].values
df['DDG_N_avgstd_one'] = df.DDG_N_avgstd_entry | df.DDG_N_avgstd_syn
df['DDG_N_avgstd_both'] = df.DDG_N_avgstd_entry & df.DDG_N_avgstd_syn


df['DD_OP_entry'] = semchanges_OP.loc[df.entry][str(DECADES[-1])].values
df['DD_OP_syn'] = semchanges_OP.loc[df.syn][str(DECADES[-1])].values
df['DDG_OP_entry'] = words_labels.loc[df.entry]['DDG_OP'].values
df['DDG_OP_syn'] = words_labels.loc[df.syn]['DDG_OP'].values

df['DDG_OP_avg_entry'] = words_labels.loc[df.entry]['DDG_OP_avg'].values
df['DDG_OP_avg_syn'] = words_labels.loc[df.syn]['DDG_OP_avg'].values
df['DDG_OP_avg_one'] = df.DDG_OP_avg_entry | df.DDG_OP_avg_syn
df['DDG_OP_avg_both'] = df.DDG_OP_avg_entry & df.DDG_OP_avg_syn

df['DDG_OP_avgstd_entry'] = words_labels.loc[df.entry]['DDG_OP_avgstd'].values
df['DDG_OP_avgstd_syn'] = words_labels.loc[df.syn]['DDG_OP_avgstd'].values
df['DDG_OP_avgstd_one'] = df.DDG_OP_avgstd_entry | df.DDG_OP_avgstd_syn
df['DDG_OP_avgstd_both'] = df.DDG_OP_avgstd_entry & df.DDG_OP_avgstd_syn


# Minimal Frequency for kept pairs : 
df['FO_entry'] = words_labels.loc[df.entry]['Origin_Freq'].values
df['FO_syn'] = words_labels.loc[df.syn]['Origin_Freq'].values
df['FE_entry'] = words_labels.loc[df.entry]['End_Freq'].values
df['FE_syn'] = words_labels.loc[df.syn]['End_Freq'].values
df['EnoughFreq'] = (df.FO_entry >= MIN_FREQ) & (df.FO_syn >= MIN_FREQ) & (df.FE_entry >= MIN_FREQ) & (df.FE_syn >= MIN_FREQ)

df['FGO_entry'] = words_labels.loc[df.entry]['FGO'].values
df['FGO_syn'] = words_labels.loc[df.syn]['FGO'].values
df['FGE_entry'] = words_labels.loc[df.entry]['FGE'].values
df['FGE_syn'] = words_labels.loc[df.syn]['FGE'].values
df['FGEv_entry'] = words_labels.loc[df.entry]['FreqEvol'].values
df['FGEv_syn'] = words_labels.loc[df.syn]['FreqEvol'].values

distances_origin = io.getDistanceMatrix( pos, model_name, DECADES[0] )
distances_end = io.getDistanceMatrix( pos, model_name, DECADES[-1] )
n_synpairs = len(candidate_syns_pairs)
syns_distances_origin = []
syns_distances_end = []
syns_lev_distances = []
syns_neigh_distances_origin = []
syns_neigh_distances_end = []
for i,(s1,s2,_) in tqdm(enumerate(candidate_syns_pairs), desc="Computing distances", total=n_synpairs):
    ind1 = word2ind[s1]
    ind2 = word2ind[s2]
    syns_lev_distances.append(lev.distance(s1, s2))
    syns_distances_origin.append( distances_origin[ind1,ind2] )
    syns_distances_end.append( distances_end[ind1,ind2] )
    syns_neigh_distances_origin.append( [k_jaccard_distance(origin_neighbors[ind1], origin_neighbors[ind2], k) for k in SND_K_RANGE ] )
    syns_neigh_distances_end.append( [k_jaccard_distance(end_neighbors[ind1], end_neighbors[ind2], k) for k in SND_K_RANGE ] )
df['Lev'] = np.array(syns_lev_distances)
df['SDO'] = np.array(syns_distances_origin)
df['SDE'] = np.array(syns_distances_end)
df['Div'] = df['SDE'] - df['SDO']
syns_neigh_distances_origin = np.array(syns_neigh_distances_origin)
syns_neigh_distances_end = np.array(syns_neigh_distances_end)
df[['SNDO_'+str(k) for k in SND_K_RANGE]] = syns_neigh_distances_origin
df[['SNDE_'+str(k) for k in SND_K_RANGE]] = syns_neigh_distances_end
df[['NDiv_'+str(k) for k in SND_K_RANGE]] = syns_neigh_distances_end - syns_neigh_distances_origin
df['SDGO'] = 'mid'
df.loc[ df.SDO > (df.SDO.mean()+df.SDO.std()), 'SDGO' ] = 'far'
df.loc[ df.SDO < (df.SDO.mean()-df.SDO.std()), 'SDGO' ] = 'close'
df['SDGE'] = 'mid'
df.loc[ df.SDE > (df.SDE.mean()+df.SDE.std()), 'SDGE' ] = 'far'
df.loc[ df.SDE < (df.SDE.mean()-df.SDE.std()), 'SDGE' ] = 'close'


div_pop = distances_end - distances_origin
close_div_pop = div_pop[distances_origin < distances_origin.mean()] # Keep only closer than average pairs
nnz_close_div_pop = close_div_pop[np.nonzero(close_div_pop)] # Remove pairs with the same word

df['DivG_avg'] = 'syns'
df.loc[df['Div'] >= div_pop.mean(),'DivG_avg'] = 'diff'
df['DivG_avgstd'] = 'syns'
df.loc[df['Div'] >= div_pop.mean()+div_pop.std(),'DivG_avgstd'] = 'diff'

df['DivG_closeavg'] = 'syns'
df.loc[df['Div'] >= nnz_close_div_pop.mean(),'DivG_closeavg'] = 'diff'
df['DivG_closeavgstd'] = 'syns'
df.loc[df['Div'] >= nnz_close_div_pop.mean()+nnz_close_div_pop.std(),'DivG_closeavgstd'] = 'diff'




# XK control pairs
rng_controls = np.random.default_rng(0)
control_word1 = []
control_word2 = []
control_word1_DD = []
control_word2_DD = []
controls_SDO = []
controls_SDE = []
nb_doubles = 0
summed_DD = np.add.outer( semchanges[str(DECADES[-1])].values, semchanges[str(DECADES[-1])].values)
for i,(entry,syn,syns_SDO,syns_SDE,s1_DD,s2_DD) in tqdm(df[['entry','syn','SDO','SDE','DD_N_entry','DD_N_syn']].iterrows(), desc="Finding controls", total=n_synpairs):
    syns_summed_DD = s1_DD+s2_DD
    candidate_i, candidate_j = np.where( (distances_origin <= syns_SDO)&(summed_DD<=syns_summed_DD) )

    if (len(candidate_i) == 0):
        raise Exception(f'Can\'t find a control pair for ({entry},{syn}) (pair {i}).')        
    else:
        # if np.array_equal(candidate_i,candidate_j):
        #     nb_doubles += 1
        #     print("Warning: found pair is a double. Synair/SDO/sumDD: ",(entry,syn),syns_SDO,syns_summed_DD)
        #     pair_to_pick = rng_controls.integers(len(candidate_i))
        #     i_c1, i_c2 = candidate_i[pair_to_pick], candidate_j[pair_to_pick]
        # else:
        while True:
            pair_to_pick = rng_controls.integers(len(candidate_i))
            i_c1, i_c2 = candidate_i[pair_to_pick], candidate_j[pair_to_pick]
            if i_c1 != i_c2:
                break
            
        c1 = word_list[i_c1]
        c2 =  word_list[i_c2]
        control_word1.append( c1 )
        control_word2.append( c2 )
        control_word1_DD.append( semchanges.loc[c1][str(DECADES[-1])] )
        control_word2_DD.append( semchanges.loc[c2][str(DECADES[-1])] )
        controls_SDO.append(distances_origin[i_c1,i_c2])
        controls_SDE.append(distances_end[i_c1,i_c2])
df['control_1'] = np.array(control_word1)
df['control_2'] = np.array(control_word2)
df['control_DD_1'] = np.array(control_word1_DD)
df['control_DD_2'] = np.array(control_word2_DD)
df['control_SDO'] = np.array(controls_SDO)
df['control_SDE'] = np.array(controls_SDE)

df['XKcontrols'] = 'diff'
df.loc[df['SDE'] - df['control_SDE'] < 0,'XKcontrols'] = 'syns'




pop_neigh_distances_origin = list()
pop_neigh_distances_end = list()
close_i1, close_i2 = np.where( (distances_origin < distances_origin.mean()) & (distances_origin != 0) )
sample_size=100000
sample = np.random.choice(len(close_i1), size=sample_size, replace=False)
close_i1_sample = close_i1[sample]
close_i2_sample = close_i2[sample]
for ind1, ind2 in tqdm(zip(close_i1_sample,close_i2_sample), desc="Sampling distances", total=len(sample)):
    pop_neigh_distances_origin.append( [k_jaccard_distance(origin_neighbors[ind1], origin_neighbors[ind2], k) for k in SND_K_RANGE ] )
    pop_neigh_distances_end.append( [k_jaccard_distance(end_neighbors[ind1], end_neighbors[ind2], k) for k in SND_K_RANGE ] )

neigh_div_pop = np.array(pop_neigh_distances_end) - np.array(pop_neigh_distances_origin)

df[['SNDO_'+str(k) for k in SND_K_RANGE]] = syns_neigh_distances_origin
df[['SNDE_'+str(k) for k in SND_K_RANGE]] = syns_neigh_distances_end
df[['NDiv_'+str(k) for k in SND_K_RANGE]] = syns_neigh_distances_end - syns_neigh_distances_origin

df[['NDivG_avg_'+str(k) for k in SND_K_RANGE]] = ['syns']*len(SND_K_RANGE)
df[['NDivG_avgstd_'+str(k) for k in SND_K_RANGE]] = ['syns']*len(SND_K_RANGE)
for i,k in enumerate(SND_K_RANGE):
    df.loc[df['NDiv_'+str(k)] >= neigh_div_pop.mean(axis=0)[i],'NDivG_avg_'+str(k)] = 'diff'
    df.loc[df['NDiv_'+str(k)] >= neigh_div_pop.mean(axis=0)[i]+neigh_div_pop.std(axis=0)[i],'NDivG_avgstd_'+str(k)] = 'diff'



df['absFGEv_entry'] = df.FGEv_entry.abs()
df['absFGEv_syn'] = df.FGEv_syn.abs()
df['normLev'] = df.Lev /((df.entry.str.len()+df.syn.str.len())/2)
df['FGDO'] = (df.FGO_entry - df.FGO_syn)
df['FGDE'] = (df.FGE_entry - df.FGE_syn)
df['FGEvD'] = df['FGEv_entry'] - df['FGEv_syn']
df['DeltaDD_N'] = (df.DD_N_entry - df.DD_N_syn).abs()
df['DeltaDD_OP'] = (df.DD_OP_entry - df.DD_OP_syn).abs()

print('Syns (raw) :', sum(df.syns_WordNet))
print('Syns (raw) %:', round(sum(df.syns_WordNet)/len(df)*100,1))
print('Hypernyms (all):', sum(df.WN_hypernymy))
print('Hypernyms (all) %:', round(sum(df.WN_hypernymy)/len(df)*100,1))
print('Hypernyms (direct):', sum(df.WN_distance==1))
print('Hypernyms (direct) %:', round(sum(df.WN_distance==1)/len(df)*100,1))
print('Hypernyms (2):', sum( df.WN_hypernymy&(df.WN_distance==2) ))
print('Hypernyms (2) %:', round(sum(df.WN_hypernymy&(df.WN_distance==2))/len(df)*100,1))
print('Hypernyms (3):', sum(df.WN_hypernymy&(df.WN_distance==3)))
print('Hypernyms (3) %:', round(sum(df.WN_hypernymy&(df.WN_distance==3))/len(df)*100,1))
print('Syns (with hyp.):', sum(df.WN_hypernymy)+sum(df.syns_WordNet))
print('Syns (with hyp.) %:', round((sum(df.WN_hypernymy)+sum(df.syns_WordNet))/len(df)*100,1))
print('Antonyms :', sum(df.WN_antonymy))
print('Antonyms %:', round(sum(df.WN_antonymy)/len(df)*100,1))
print('Holonyms :', sum(df.WN_holonymy))
print('Holonyms %:', round(sum(df.WN_holonymy)/len(df)*100,1))
print('--------------------------')
print('Nb of bad control pairs (double):', nb_doubles)

df.index.name = 'pairIdx'
df.to_csv(f'{CSV_FOLDER}/{model_name}/{pos}_synpairs_analysis.csv', sep='\t',index=True)
