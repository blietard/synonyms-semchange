from tqdm import tqdm
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import toolsIO as io

from params import CSV_FOLDER, COLORMAP, MARKERMAP, DECADES

import argparse
parser = argparse.ArgumentParser(description='Create plots from pairs of synonyms and csv for close pairs.')
parser.add_argument('pos', metavar='PoS', type=str, nargs='?',
                    help='the Part-of-Speech tag', choices=['A','N','V'])
parser.add_argument('repr', metavar='Repr', type=str, nargs='?',
                    help='the Word Representation model', choices=['sgns','sppmi','doubNorm'])
parser.add_argument('dist', metavar='Dist', type=str, nargs='?',
                    help='the Synchronic Distance', choices=['cosine','euclid'])
parser.add_argument('--select', '-s', type=str, default='avg', nargs='?',
                    help='synonyms SC selection threshold', choices=['avg','avgstd'])
parser.add_argument('--closeness', '-c', type=str, default='stdavg_s', nargs='?',
                    help='synonyms closeness threshold', choices=['stdavg_s','2stdavg_p','avg_s','avg_p'])
args = parser.parse_args()
pos, repr_mode, distance, selection_mode , closeness = args.pos, args.repr, args.dist, args.select, args.closeness

model_name = repr_mode + '_' + distance
syn_pairs = io.getSynPairs(selection_mode,model_name,pos)
word_list, word2ind = io.getTargets(pos,repr_mode)
contexts = io.getContexts()
distances_origin = io.getDistanceMatrix( pos, model_name, DECADES[0] )
distances_end = io.getDistanceMatrix( pos, model_name, DECADES[-1] )
semchanges = io.getSemChange(pos,model_name)

bin_count = 50
n_points = 50
max_std_times = 3
min_std_times = 1
top_k_div = 50


if distance == 'cosine':
    max_dist = 2
    bin_count *= 2

n_synpairs = len(syn_pairs)
s1_list = np.empty(n_synpairs,dtype='object')
s2_list = np.empty(n_synpairs,dtype='object')
syns_distances_origin = []
syns_distances_end = []
sc1_list = np.empty(n_synpairs,dtype='object')
sc2_list = np.empty(n_synpairs,dtype='object')
for i,(s1,s2) in tqdm(enumerate(syn_pairs), desc="Synonyms", total=n_synpairs):
    ind1 = word2ind[s1]
    ind2 = word2ind[s2]
    s1_list[i] = s1
    s2_list[i] = s2
    sc1_list[i] = semchanges[str(DECADES[-1])][s1]
    sc2_list[i] = semchanges[str(DECADES[-1])][s2]
    syns_distances_origin.append( distances_origin[ind1,ind2] )
    syns_distances_end.append( distances_end[ind1,ind2] )
syns_distances_origin = np.array(syns_distances_origin)
syns_distances_end = np.array(syns_distances_end)

distances_origin_pop = distances_origin.flatten()
distances_end_pop = distances_end.flatten()


if distance == "euclid":
    max_dist = max(max(max(syns_distances_origin),max(syns_distances_end)),max(max(distances_origin_pop),max(distances_end_pop))) +1
    bin_count *= 2

if closeness.lower() == 'avg_p':
    nu = distances_origin_pop.mean()
elif closeness.lower() == 'stdavg_p':
    nu = distances_origin_pop.mean() - distances_origin_pop.std()
elif closeness.lower() == '2stdavg_p':
    nu = distances_origin_pop.mean() - 2*distances_origin_pop.std()
elif closeness.lower() == 'avg_s':
    nu = syns_distances_origin.mean()
elif closeness.lower() == 'stdavg_s':
    nu = syns_distances_origin.mean() - syns_distances_origin.std()
else:
    raise ValueError('The closeness threshold <threshold> must be avg_P, stdavg_P, 2stdavg_P (for population), avg_S, stdavg_S (for synonyms).')
close_mask = syns_distances_origin < nu

div_syns = syns_distances_end-syns_distances_origin
div_pop = distances_end_pop - distances_origin_pop

n_close = close_mask.sum()
print(f'Closeness threshold value: {nu.round(4)}')
print(f'Number of close synonym pairs: {n_close} / {len(syns_distances_origin)}')

print('===== CSV GENERATION =====')

if not os.path.exists(f'{CSV_FOLDER}/{model_name}/'):
    os.mkdir(f'{CSV_FOLDER}/{model_name}/')

columns = ('Npair','word1','word2','DDist1','DDist2','SDist1890','SDist1990','Div')
table = np.transpose( np.vstack( ( np.arange(n_synpairs)[close_mask],
                        np.array(syn_pairs)[close_mask].T,
                        sc1_list[close_mask], 
                        sc2_list[close_mask], 
                        syns_distances_origin[close_mask], 
                        syns_distances_end[close_mask], 
                        div_syns[close_mask]
                        ) ) )
df = pd.DataFrame(table,columns=columns).set_index('Npair')
df['avg'],df['avgstd'] = ( df.Div > div_pop.mean(), df.Div > div_pop.mean() + div_pop.std() )

df[ df['avg'] != True ].sort_values('SDist1990').to_csv(f'{CSV_FOLDER}/{model_name}/{pos}_{selection_mode}_{closeness}_converging_syns.csv')
df[ df['avg'] & (df['avgstd'] != True) ].sort_values('SDist1990').to_csv(f'{CSV_FOLDER}/{model_name}/{pos}_{selection_mode}_{closeness}_stable_syns.csv')
df[ df['avgstd'] ].sort_values('Div',ascending=False).head(top_k_div).sort_values('SDist1990',ascending=False).to_csv(f'{CSV_FOLDER}/{model_name}/{pos}_{selection_mode}_{closeness}_diverging_syns.csv')

print('CSV stored for converging, stable and diverging close syns pairs.')

print('===== PLOTS GENERATION =====')
# ======================================
print('Thresholding curve.')

thresholds = np.linspace(
                start = div_pop.mean() - min_std_times*div_pop.std(),
                stop = div_pop.mean() + max_std_times*div_pop.std(),
                num = n_points
                )
print(f'thresholds: {n_points} points between {thresholds.min()} and {thresholds.max()}')
ld_props = np.vstack( [div_syns > t for t in thresholds] ).sum(axis=1)/len(div_syns)*100
lpc_props = 100 - ld_props

ld_props_close = np.vstack( [div_syns[close_mask] > t for t in thresholds] ).sum(axis=1)/n_close*100
lpc_props_close = 100 - ld_props_close

fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
ax = fig.add_subplot(111)

ax.plot(thresholds,ld_props,label='LD (all pairs)', color = COLORMAP['s'], marker=MARKERMAP['s'])
ax.plot(thresholds,ld_props_close,label='LD (close pairs)', color = COLORMAP['c'], marker=MARKERMAP['c'])
ax.set_ylim(0, 100)

ax.axvline(div_pop.mean(),label='$\\tau=0$',color='black',linestyle='--',alpha=0.4)
ax.axvline(div_pop.mean()+div_pop.std(),label='$\\tau=$ 1 std',color='red',linestyle='--',alpha = 0.4)

ax.yaxis.grid()
ax.set_title(f'Proportions curves for POS {pos}, selection {selection_mode},\n model {model_name} and closeness {closeness}.')
ax.set_xlabel('Threshold: $\\overline{\\Delta(pop)}+\\tau$')
ax.set_ylabel('% of Differentiation')
ax.legend(loc='center left')
fig.savefig(f'./img/{model_name}_{pos}_{selection_mode}_thresholding_closepairs_{closeness}.png',transparent=False,dpi=100)

# ======================================
print('Distance repartition scatterplot.')

# Subsampling for plot generation
if len(distances_end_pop) > 100000:
    sample_pop = np.random.randint(low=0,high=len(distances_end_pop),size=100000) # choose 100'000 randomly
else:
    sample_pop = list(range(len(distances_end_pop))) # all pairs

fig = plt.figure(figsize=(10,10))
fig.set_facecolor('white')
ax = fig.add_subplot(111)

ax.scatter(distances_origin_pop[sample_pop],distances_end_pop[sample_pop],label='pop pairs',alpha=0.6, color = COLORMAP['p'])
ax.scatter(syns_distances_origin,syns_distances_end,label='syn. pairs', color = COLORMAP['s'])
ax.scatter(syns_distances_origin[close_mask],syns_distances_end[close_mask],label='close syn. pairs', color = COLORMAP['c'])

ax.set_xlim(0,max_dist+0.05)
ax.set_ylim(0,max_dist+0.05)

ax.yaxis.grid()
ax.xaxis.grid()
ax.legend()
ax.set_title(f'Scatterplot of word pairs distances for POS {pos}, selection {selection_mode},\n model {model_name} and closeness threshold {closeness}.')
ax.set_xlabel('Synchronic distance S-Dist in 1890')
ax.set_ylabel('Synchronic distance S-Dist in 1990')
fig.savefig(f'./img/{model_name}_{pos}_{selection_mode}_repartition_closepairs_{closeness}.png',transparent=False,dpi=100)


fig = plt.figure(figsize=(10,10))
fig.set_facecolor('white')
ax = fig.add_subplot(111)
ax.scatter(distances_origin_pop[sample_pop],div_pop[sample_pop],label='pop pairs',alpha=0.6, color = COLORMAP['p'])
ax.scatter(syns_distances_origin,div_syns,label='syn. pairs', color = COLORMAP['s'])
ax.scatter(syns_distances_origin[close_mask],div_syns[close_mask],label='close syn. pairs', color = COLORMAP['c'])
ax.set_xlim(0,max_dist+0.05)
ax.set_ylim(-max_dist/2,max_dist/2+0.05)
ax.yaxis.grid()
ax.xaxis.grid()
ax.legend()
ax.set_title(f'Scatterplot of word pairs distances for POS {pos}, selection {selection_mode},\n model {model_name} and closeness threshold {closeness}.')
ax.set_xlabel('Synchronic distance S-Dist in 1890')
ax.set_ylabel('Divergence (S-Dist_1990 - S-Dist_1890)')
fig.savefig(f'./img/{model_name}_{pos}_{selection_mode}_repartition_divergence_closepairs_{closeness}.png',transparent=False,dpi=100)


# ======================================
print('Distances and difference histogram.')

fig = plt.figure(figsize=(12,6))
fig.set_facecolor('white')
ax = fig.subplot_mosaic([['1890','1990'],['D','D']])
ax['1890'].hist(syns_distances_origin, bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label = 'all syn. pairs', color = COLORMAP['s'])
ax['1890'].hist(syns_distances_origin[close_mask], bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))), label = 'close pairs', color = COLORMAP['c'])
ax['1890'].set_title('1890')
ax['1990'].hist(syns_distances_end, bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , color = COLORMAP['s'])
ax['1990'].hist(syns_distances_end[close_mask], bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))), color = COLORMAP['c'])
ax['1990'].set_title('1990')
ax['D'].hist(div_syns, bins=(np.arange(bin_count+1)*(max_dist/bin_count) - (max_dist/2)), label = 'all syn. pairs', color = COLORMAP['s'])
ax['D'].hist(div_syns[close_mask], bins=(np.arange(bin_count+1)*(max_dist/bin_count) - (max_dist/2)), label = f'close pairs ({n_close}/{len(syn_pairs)})', color = COLORMAP['c'])
ymin,ymax = ax['D'].get_ylim()
ax['D'].vlines(div_pop.mean(),ymin=ymin, ymax=ymax ,label='Average difference',colors='black',linestyles='--',alpha=0.7)
ax['D'].vlines(div_pop.mean()+div_pop.std(), ymin=ymin, ymax=ymax,label='Average difference + 1 std',colors='red',linestyles='--',alpha = 0.7)
ax['D'].set_title('Difference')
ax['D'].legend()
ax['1890'].set_ylabel('Nb of pairs')
ax['1890'].set_xlabel('Synchronic distance S-Dist')
ax['1990'].set_xlabel('Synchronic distance S-Dist')
ax['D'].set_ylabel('Nb of pairs')
ax['D'].set_xlabel('Divergence (S-Dist_1990 - S-Dist_1890)')
fig.suptitle(f'Distribution of synonyms for POS {pos}, selection {selection_mode},\n model {model_name} and closeness threshold {closeness}.')
fig.tight_layout()
fig.savefig(f'./img/{model_name}_{pos}_{selection_mode}_histograms_closepairs_{closeness}.png',transparent=False,dpi=100)



fig = plt.figure(figsize=(12,6))
fig.set_facecolor('white')
ax = fig.subplot_mosaic([['1890','1990'],['D','D']])
ax['1890'].hist(distances_origin_pop[sample_pop], bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label='Population', color = COLORMAP['p'], alpha=0.8, density=True)
ax['1890'].hist(syns_distances_origin, bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label='Syn. pairs', color = COLORMAP['s'], alpha=0.8, density=True)
ax['1890'].set_title('1890')
ax['1990'].hist(distances_end_pop[sample_pop], bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label='Population', color = COLORMAP['p'], alpha=0.8, density=True)
ax['1990'].hist(syns_distances_end, bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label='Syn. pairs', color = COLORMAP['s'], alpha=0.8, density=True)
ax['1990'].set_title('1990')
ax['D'].hist(div_pop[sample_pop], bins=(np.arange(bin_count+1)*(max_dist/bin_count) - (max_dist/2)) , label='Population', color = COLORMAP['p'], alpha=0.8, density=True)
ax['D'].hist(div_syns, bins=(np.arange(bin_count+1)*(max_dist/bin_count) - (max_dist/2))  , label='Syn. pairs', color = COLORMAP['s'], alpha=0.8, density=True)
ax['D'].set_title('Difference')
ax['D'].legend()
ax['1890'].set_ylabel('Density')
ax['1890'].set_xlabel('Synchronic distance S-Dist')
ax['1990'].set_xlabel('Synchronic distance S-Dist')
ax['D'].set_ylabel('Density')
ax['D'].set_xlabel('Divergence (S-Dist_1990 - S-Dist_1890)')
fig.suptitle(f'Distribution of synonyms with {distance}')
fig.tight_layout()
fig.savefig(f'./img/{model_name}_{pos}_{selection_mode}_histograms.png',transparent=False,dpi=100)

fig = plt.figure(figsize=(12,6))
fig.set_facecolor('white')
ax = fig.subplot_mosaic([['1890','1990'],['D','D']])
ax['1890'].hist(distances_origin_pop[sample_pop], bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label='Population', color = COLORMAP['p'], alpha=0.8, density=True)
ax['1890'].hist(syns_distances_origin, bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label=f'Syn. pairs ({len(syn_pairs)})', color = COLORMAP['s'], alpha=0.8, density=True)
ax['1890'].hist(syns_distances_origin[close_mask], bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label=f'Close syn pairs ({n_close})', color = COLORMAP['c'], alpha=0.5, density=True)
ax['1890'].set_title('1890')
ax['1990'].hist(distances_end_pop[sample_pop], bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label='Population', color = COLORMAP['p'], alpha=0.8, density=True)
ax['1990'].hist(syns_distances_end, bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label='Syn. pairs', color = COLORMAP['s'], alpha=0.8, density=True)
ax['1990'].hist(syns_distances_end[close_mask], bins=(np.arange(bin_count//2+1)*(max_dist/(bin_count//2))) , label='Close syn. pairs', color = COLORMAP['c'], alpha=0.5, density=True)
ax['1990'].set_title('1990')
ax['D'].hist(div_pop[sample_pop], bins=(np.arange(bin_count+1)*(max_dist/bin_count) - (max_dist/2)) , label='Population', color = COLORMAP['p'], alpha=0.8, density=True)
ax['D'].hist(div_syns, bins=(np.arange(bin_count+1)*(max_dist/bin_count) - (max_dist/2))  , label=f'Syn. pairs ({len(syn_pairs)})', color = COLORMAP['s'], alpha=0.8, density=True)
ax['D'].hist(div_syns[close_mask], bins=(np.arange(bin_count+1)*(max_dist/bin_count) - (max_dist/2))  , label=f'Close syn. pairs ({n_close})', color = COLORMAP['c'], alpha=0.6, density=True)
ax['D'].set_title('Difference')
ax['D'].legend()
ax['1890'].set_ylabel('Density')
ax['1890'].set_xlabel('Synchronic distance S-Dist')
ax['1990'].set_xlabel('Synchronic distance S-Dist')
ax['D'].set_ylabel('Density')
ax['D'].set_xlabel('Divergence (S-Dist_1990 - S-Dist_1890)')
fig.suptitle(f'Distribution of synonyms with {distance}')
fig.tight_layout()
fig.savefig(f'./img/{model_name}_{pos}_{selection_mode}_histograms_closepairs_{closeness}_MIXED.png',transparent=False,dpi=100)
