import pandas as pd 
import numpy as np
from params import DECADES, CSV_FOLDER
import toolsIO as io 
from tqdm import tqdm
import argparse

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

print('Retrieving synpairs...')
wn_synpairs = io.getWordNetSyns(pos)
wn_synpairs = [ (w1,w2) for w1,w2 in wn_synpairs if (w1 in word_list and w2 in word_list) ]
print('Analysing synpairs...')

df = pd.DataFrame(columns=['syn1','syn2'],data=wn_synpairs).sort_values(['syn1','syn2']).reset_index()

df['DiachrD1'] = semchanges.loc[df.syn1][str(DECADES[-1])].values
df['DiachrD2'] = semchanges.loc[df.syn2][str(DECADES[-1])].values
df['DiachrDG1'] = words_labels.loc[df.syn1]['SCgroup'].values
df['DiachrDG2'] = words_labels.loc[df.syn2]['SCgroup'].values
df['DiachrDG_avg'] = None
df.loc[df.DiachrDG1.isin(['avg','avgstd']) , 'DiachrDG_avg'] = 's1'
df.loc[df.DiachrDG2.isin(['avg','avgstd']) , 'DiachrDG_avg'] = 's2'
df.loc[ (df.DiachrDG1.isin(['avg','avgstd']) & df.DiachrDG2.isin(['avg','avgstd'])) , 'DiachrDG_avg'] = 'both'

df['DiachrDG_avgstd'] = None
df.loc[df.DiachrDG1 == 'avgstd' , 'DiachrDG_avgstd'] = 's1'
df.loc[df.DiachrDG2 == 'avgstd' , 'DiachrDG_avgstd'] = 's2'
df.loc[ (df.DiachrDG1 =='avgstd') & (df.DiachrDG2 == 'avgstd') , 'DiachrDG_avgstd'] = 'both'

# Minimal Frequency for kept pairs : 
df['FO1'] = words_labels.loc[df.syn1]['Origin_Freq'].values
df['FO2'] = words_labels.loc[df.syn2]['Origin_Freq'].values
df['FE1'] = words_labels.loc[df.syn1]['End_Freq'].values
df['FE2'] = words_labels.loc[df.syn2]['End_Freq'].values
df['EnoughFreq'] = (df.FO1 >= 5) & (df.FO2 >= 5) & (df.FE1 >= 5) & (df.FE2 >= 5)

df['FGO1'] = words_labels.loc[df.syn1]['FGO'].values
df['FGO2'] = words_labels.loc[df.syn2]['FGO'].values
df['FGE1'] = words_labels.loc[df.syn1]['FGE'].values
df['FGE2'] = words_labels.loc[df.syn2]['FGE'].values
df['FEvol1'] = words_labels.loc[df.syn1]['FreqEvol'].values
df['FEvol2'] = words_labels.loc[df.syn2]['FreqEvol'].values


wn_polysemy = io.getWordNetPolysemy(pos)
nb_senses1 = list()
nb_senses2 = list()
pressure1_wordnet = list()
pressure2_wordnet = list()
for pair in list(df[['syn1','syn2']].itertuples(index=False,name=None)):
    try:
        nb_senses1.append( wn_polysemy['nb_senses'].loc[pair[0]] )
        pressure1_wordnet.append( wn_polysemy['nb_syns'].loc[pair[0]] )
    except KeyError:
        nb_senses1.append(0)
        pressure2_wordnet.append(0)
    try:
        nb_senses2.append( wn_polysemy['nb_senses'].loc[pair[1]] )
        pressure2_wordnet.append( wn_polysemy['nb_syns'].loc[pair[1]] )
    except KeyError:
        nb_senses2.append(0)
        pressure2_wordnet.append(0)
df['WNpressure_1'] = pressure1_wordnet
df['WNpressure_2'] = pressure2_wordnet
df['WNsenses_1'] = nb_senses1
df['WNsenses_2'] = nb_senses2

distances_origin = io.getDistanceMatrix( pos, model_name, DECADES[0] )
distances_end = io.getDistanceMatrix( pos, model_name, DECADES[-1] )
n_synpairs = len(wn_synpairs)
syns_distances_origin = []
syns_distances_end = []
for i,(s1,s2) in tqdm(enumerate(wn_synpairs), desc="Synonyms", total=n_synpairs):
    ind1 = word2ind[s1]
    ind2 = word2ind[s2]
    syns_distances_origin.append( distances_origin[ind1,ind2] )
    syns_distances_end.append( distances_end[ind1,ind2] )
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
df['Dec_avg'] = 'LPC'
df.loc[df['Div'] > div_pop.mean(),'Dec_avg'] = 'LD'
df['Dec_avgstd'] = 'LPC'
df.loc[df['Div'] > div_pop.mean()+div_pop.std(),'Dec_avgstd'] = 'LD'


df.index.name = 'pairIdx'
df.to_csv(f'{CSV_FOLDER}/{model_name}/{pos}_WNsynpairs_analysis.csv', sep='\t',index=True)
