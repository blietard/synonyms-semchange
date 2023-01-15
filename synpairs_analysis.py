import pandas as pd 
import numpy as np
from params import WORDS_FOLDER, DECADES, CSV_FOLDER, MINIMUM_FREQ
import toolsIO as io 
import os
import pickle
from tqdm import tqdm
import argparse
import Levenshtein as lev

parser = argparse.ArgumentParser(description='Select pairs of synonyms.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['A','N','V'])
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

with open(f'{WORDS_FOLDER}/fernald_synonyms.pickle','rb') as f:
    syns_dict = pickle.load(f)[pos]

if not os.path.exists(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/'):
    os.makedirs(f'{WORDS_FOLDER}/synonym_pairs/{model_name}/')


candidate_syns_pairs = []
for entry, syns in syns_dict.items():
    if entry in word_list:
        for syn in syns:
            if syn in word_list:
                candidate_syns_pairs.append( (entry,syn, len(syns)) )

print('[STATS:] Initial number of candidate pairs : ',len(candidate_syns_pairs))

df = pd.DataFrame(columns=['entry','syn','pressure_entry'],data=candidate_syns_pairs)


wn_synpairs = io.getWordNetSyns(pos)
df['syns_WordNet'] = [pair in wn_synpairs for pair in list(df[['entry','syn']].itertuples(index=False,name=None))]

wn_polysemy = io.getWordNetPolysemy(pos)
nb_senses1 = []
nb_senses2 = []
pressure_wordnet = list()
for pair in list(df[['entry','syn']].itertuples(index=False,name=None)):
    try:
        nb_senses1.append( wn_polysemy['nb_senses'].loc[pair[0]] )
        pressure_wordnet.append( wn_polysemy['nb_syns'].loc[pair[0]] )
    except KeyError:
        nb_senses1.append(0)
        pressure_wordnet.append(0)
    try:
        nb_senses2.append( wn_polysemy['nb_senses'].loc[pair[1]] )
    except KeyError:
        nb_senses2.append(0)
df['WNpressure_entry'] = pressure_wordnet
df['WNsenses_entry'] = nb_senses1
df['WNsenses_syn'] = nb_senses2

end_neighbors = io.getNeighbors(pos,model_name,DECADES[-1])
WNsyndict = io.getWordNetSyns_asDict(pos)
def areSynsExtended(w1,w2,areSynsWN,k=10,debug=False):
    if areSynsWN:
        if debug:
            return (True,'')
        else:
            return True
    w1_synsWN = {synWN for synWN in WNsyndict[w1] if synWN in word_list}
    w2_synsWN = {synWN for synWN in WNsyndict[w2] if synWN in word_list}

    # Search in neighborhood of w1
    end_neigh_w1 = np.array(word_list)[ end_neighbors[word2ind[w1]] ][:k]
    if w2 in end_neigh_w1:
        for wordX in w1_synsWN:
            end_neigh_X = np.array(word_list)[ end_neighbors[word2ind[wordX]] ][:k]
            if w2 in end_neigh_X:
                if debug:
                    return (True,wordX)
                else:
                    return True

    # Search in neighborhood of w2
    end_neigh_w2 = np.array(word_list)[ end_neighbors[word2ind[w2]] ][:k]
    if w1 in end_neigh_w2:
        for wordX in w2_synsWN:
            end_neigh_X = np.array(word_list)[ end_neighbors[word2ind[wordX]] ][:k]
            if w1 in end_neigh_X:
                if debug:
                    return (True,wordX)
                else:
                    return True
    
    if debug:
        return (False,'')
    else:
        return False
syns_WNExt = []
WNExt_words = []
for entry,syn,areSynsWN in df[['entry','syn','syns_WordNet']].values:    
    syns_ext_bool, wordX = areSynsExtended(entry,syn,areSynsWN,k=20,debug=True)
    syns_WNExt.append(syns_ext_bool)
    WNExt_words.append(wordX)
df['syns_WNExt'] = syns_WNExt
df['WNExt_word'] = WNExt_words


df['DiachrD_entry'] = semchanges.loc[df.entry][str(DECADES[-1])].values
df['DiachrD_syn'] = semchanges.loc[df.syn][str(DECADES[-1])].values
df['DiachrDG_entry'] = words_labels.loc[df.entry]['SCgroup'].values
df['DiachrDG_syn'] = words_labels.loc[df.syn]['SCgroup'].values
df['DiachrDG_pair_avg'] = None
df.loc[df.DiachrDG_entry.isin(['avg','avgstd']) , 'DiachrDG_pair_avg'] = 'entry'
df.loc[df.DiachrDG_syn.isin(['avg','avgstd']) , 'DiachrDG_pair_avg'] = 'syn'
df.loc[ (df.DiachrDG_entry.isin(['avg','avgstd']) & df.DiachrDG_syn.isin(['avg','avgstd'])) , 'DiachrDG_pair_avg'] = 'both'

df['DiachrDG_pair_avgstd'] = None
df.loc[df.DiachrDG_entry == 'avgstd' , 'DiachrDG_pair_avgstd'] = 'entry'
df.loc[df.DiachrDG_syn == 'avgstd' , 'DiachrDG_pair_avgstd'] = 'syn'
df.loc[ (df.DiachrDG_entry =='avgstd') & (df.DiachrDG_syn == 'avgstd') , 'DiachrDG_pair_avgstd'] = 'both'

# Minimal Frequency for kept pairs : 
df['FO_entry'] = words_labels.loc[df.entry]['Origin_Freq'].values
df['FO_syn'] = words_labels.loc[df.syn]['Origin_Freq'].values
df['FE_entry'] = words_labels.loc[df.entry]['End_Freq'].values
df['FE_syn'] = words_labels.loc[df.syn]['End_Freq'].values
df['EnoughFreq'] = (df.FO_entry >= MINIMUM_FREQ) & (df.FO_syn >= MINIMUM_FREQ) & (df.FE_entry >= MINIMUM_FREQ) & (df.FE_syn >= MINIMUM_FREQ)

df['FGO_entry'] = words_labels.loc[df.entry]['FGO'].values
df['FGO_syn'] = words_labels.loc[df.syn]['FGO'].values
df['FGE_entry'] = words_labels.loc[df.entry]['FGE'].values
df['FGE_syn'] = words_labels.loc[df.syn]['FGE'].values
df['FEv_entry'] = words_labels.loc[df.entry]['FreqEvol'].values
df['FEv_syn'] = words_labels.loc[df.syn]['FreqEvol'].values

distances_origin = io.getDistanceMatrix( pos, model_name, DECADES[0] )
distances_end = io.getDistanceMatrix( pos, model_name, DECADES[-1] )
n_synpairs = len(candidate_syns_pairs)
syns_distances_origin = []
syns_distances_end = []
syns_lev_distances = []
for i,(s1,s2,_) in tqdm(enumerate(candidate_syns_pairs), desc="Synonyms", total=n_synpairs):
    ind1 = word2ind[s1]
    ind2 = word2ind[s2]
    syns_lev_distances.append(lev.distance(s1, s2))
    syns_distances_origin.append( distances_origin[ind1,ind2] )
    syns_distances_end.append( distances_end[ind1,ind2] )
df['Lev'] = np.array(syns_lev_distances)
df['SDO'] = np.array(syns_distances_origin)
df['SDE'] = np.array(syns_distances_end)
df['Div'] = df['SDE'] - df['SDO']

df['SDGO'] = 'mid'
df.loc[ df.SDO > (df.SDO.mean()+df.SDO.std()), 'SDGO' ] = 'far'
df.loc[ df.SDO < (df.SDO.mean()-df.SDO.std()), 'SDGO' ] = 'close'
df['SDGE'] = 'mid'
df.loc[ df.SDE > (df.SDE.mean()+df.SDE.std()), 'SDGE' ] = 'far'
df.loc[ df.SDE < (df.SDE.mean()-df.SDE.std()), 'SDGE' ] = 'close'


div_pop = distances_end - distances_origin

close_div_pop = div_pop[distances_origin < distances_origin.mean()] # Keep only closer than average pairs
nnz_close_div_pop = close_div_pop[np.nonzero(close_div_pop)] # Remove pairs with the same word

df['Dec_avg'] = 'LPC'
df.loc[df['Div'] > nnz_close_div_pop.mean(),'Dec_avg'] = 'LD'
df['Dec_avgstd'] = 'LPC'
df.loc[df['Div'] > nnz_close_div_pop.mean()+nnz_close_div_pop.std(),'Dec_avgstd'] = 'LD'


df.index.name = 'pairIdx'
df.to_csv(f'{CSV_FOLDER}/{model_name}/{pos}_synpairs_analysis.csv', sep='\t',index=True)
