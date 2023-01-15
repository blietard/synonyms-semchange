from tqdm import tqdm
import os

import numpy as np

import matplotlib.pyplot as plt
import toolsIO as io

from params import DECADES, CSV_FOLDER, COLORMAP, MARKERMAP

import argparse
parser = argparse.ArgumentParser(description='Create plots for timeseries.')
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
semchanges = io.getSemChange(pos,model_name)
origin, later_decades = DECADES[0], DECADES[1:]
n_synpairs = len(syn_pairs)

if not os.path.exists(f'{CSV_FOLDER}/{model_name}/'):
    os.mkdir(f'{CSV_FOLDER}/{model_name}/')

distances = np.empty((len(DECADES),n_synpairs),dtype='float16')
sc_tensor = np.empty((len(DECADES),n_synpairs,2),dtype='float16')

pop_distances = np.empty((len(DECADES),len(word_list)**2),dtype='float16')


for ind_d, d in enumerate(DECADES):
    distances_matrix = io.getDistanceMatrix(pos, model_name, d)
    sc_table = semchanges[str(d)]

    for ind_pair,(s1,s2) in tqdm(enumerate(syn_pairs), desc=f"{d} Synonyms", total=len(syn_pairs)):
        ind1 = word2ind[s1]
        ind2 = word2ind[s2]
        sc_tensor[ind_d][ind_pair] = semchanges[str(d)][[s1,s2]]
        distances[ind_d][ind_pair] = distances_matrix[ind1,ind2]

    pop_distances[ind_d] = distances_matrix.flatten()

if closeness.lower() == 'avg_p': # ind_d=0 because 1890
    nu = pop_distances[0].mean()
elif closeness.lower() == 'stdavg_p':
    nu = pop_distances[0].mean() - pop_distances[0].std()
elif closeness.lower() == '2stdavg_p':
    nu = pop_distances[0].mean() - 2*pop_distances[0].std()
elif closeness.lower() == 'avg_s':
    nu = distances[0].mean()
elif closeness.lower() == 'stdavg_s':
    nu = distances[0].mean() - distances[0].std()
else:
    raise ValueError('The closeness threshold <threshold> must be avg_P, stdavg_P, 2stdavg_P (for population), avg_S, stdavg_S (for synonyms).')
close_mask = distances[0] < nu #close in 1890
n_close = close_mask.sum()

if pop_distances.shape[1] > 200000:
    print('Sub-sampled pop for convenience.')
    sample = np.random.randint(low=0,high=pop_distances.shape[1],size=200000)
    pop_distances = pop_distances[:,sample]

div_syns = distances-distances[0]
div_close = distances[:,close_mask]-distances[0,close_mask]
div_pop = pop_distances - pop_distances[0]

labels = {'s':f'All syns ({n_synpairs})','c':f'Close syns ({n_close})','p':'Population'}
div_map = {'s':div_syns,'c':div_close,'p':div_pop}
dist_map = {'s':distances,'c':distances[:,close_mask],'p':pop_distances}
sc_map = {'s':sc_tensor.reshape((len(DECADES),n_synpairs*2)),'c':sc_tensor[:,close_mask,:].reshape((len(DECADES),n_close*2)),'p':semchanges.values.T}

print('===== PLOTS GENERATION =====')
# ======================================
print('Divergence curve.')

fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
ax = fig.add_subplot(111)

ax.set_ylim(-0.5,0.5)
for group in ['s','c','p']:
    means = div_map[group].mean(axis=1)
    errors = div_map[group].std(axis=1)
    ax.plot(DECADES,means,label=labels[group], color=COLORMAP[group], marker=MARKERMAP[group])
    # ax.plot(DECADES,means-errors,alpha=0.2,ls='--',color=c)
    # ax.plot(DECADES,means+errors,alpha=0.2,ls='--',color=c)
    ax.fill_between(DECADES,means-errors,means+errors,alpha=0.1,color=COLORMAP[group])

ax.yaxis.grid()
ax.set_xticks(DECADES)
ax.set_title(f'Pair divergence curves for POS {pos}, selection {selection_mode},\n model {model_name} and closeness {closeness}.')
ax.set_xlabel('Decade')
ax.set_ylabel('Divergence (S-Dist_T - S-Dist_1890 ) ')
ax.legend()
fig.savefig(f'./img/{model_name}_{pos}_{selection_mode}_Timeseries_Div_{closeness}.png',transparent=False,dpi=100)


# ======================================
print('Distance curve')

fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
ax = fig.add_subplot(111)

ax.set_ylim(0,1.25)
for group in ['s','c','p']:
    means = dist_map[group].mean(axis=1)
    errors = dist_map[group].std(axis=1)
    ax.plot(DECADES,means,label=labels[group], color=COLORMAP[group], marker=MARKERMAP[group])
    # ax.plot(DECADES,means-errors,alpha=0.2,ls='--',color=c)
    # ax.plot(DECADES,means+errors,alpha=0.2,ls='--',color=c)
    ax.fill_between(DECADES,means-errors,means+errors,alpha=0.1,color=COLORMAP[group])

ax.yaxis.grid()
ax.set_xticks(DECADES)
ax.set_title(f'Pair S-Dist curves for POS {pos}, selection {selection_mode},\n model {model_name} and closeness {closeness}.')
ax.set_xlabel('Decade')
ax.set_ylabel('Synchronic distance S-Dist between word pairs')
ax.legend()
fig.savefig(f'./img/{model_name}_{pos}_{selection_mode}_Timeseries_Distance_{closeness}.png',transparent=False,dpi=100)

# ======================================
print('SemChange curve')

fig = plt.figure(figsize=(12,8))
fig.set_facecolor('white')
ax = fig.add_subplot(111)


ax.set_ylim(0,1.1)
for group in ['s','c','p']:
    means = sc_map[group].mean(axis=1)
    errors = sc_map[group].std(axis=1)
    ax.plot(DECADES,means,label=labels[group], color=COLORMAP[group], marker=MARKERMAP[group])
    ax.fill_between(DECADES,means-errors,means+errors,alpha=0.1,color=COLORMAP[group])

ax.set_xticks(DECADES)
ax.yaxis.grid()
ax.set_title(f'Diachronic distance curves of words for POS {pos}, selection {selection_mode},\n model {model_name} and closeness {closeness}.')
ax.set_xlabel('Decade')
ax.set_ylabel('Diachronic distance D-Dist from $T_1$ = 1890s')
ax.legend()
fig.savefig(f'./img/{model_name}_{pos}_{selection_mode}_Timeseries_SemChange_{closeness}.png',transparent=False,dpi=100)
