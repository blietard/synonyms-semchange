from numpy import arange

SELECTED_TAGS = ['N','ADJ','V']

MIN_FREQ = 3 # minimum frequency per decade
MIN_LENGTH = 3 # minimum word length

K_NEIGHBORS = 100 
NB_FREQ_GROUPS = 5

SOURCE_SYNDICT = '/home/blietard/PhD/synonyms/fernald.txt'
WORDNET_DB_FOLDER  = '/home/blietard/Downloads/wn3.1.dict/dict/'
OPENWORDNET_DB_FOLDER  = '/home/blietard/Downloads/english-wordnet-2022/dict/'

TEMP_FOLDER = './temp/'
MATRIX_FOLDER = '/home/blietard/syns_data/cooc_matrices_GNgrams_V2/'
HAMILTON_SGNS_FOLDER = '/home/blietard/PhD/HamiltonSGNS/'
SGNS_FOLDER = '/home/blietard/PhD/synonyms/sgns/'
WORDS_FOLDER = './words/'
COHA_FREQ_FILE = '1_pos_y_cs_n_academicUse.txt'
INFO_WORDS_FOLDER = './words_info/'
NEIGHBORS_FOLDER = './nearest_neighbors/'
DISTANCES_FOLDER = './distances/'
CSV_FOLDER = './out_csv/'
IMG_FOLDER = './img/'

STARTYEAR = 1890 #included ; minimum 1800
ENDYEAR = 2000 #excluded ; maximum 2000

DECADES = arange(start=STARTYEAR,stop=ENDYEAR,step=10)
DECADES_INDS = (DECADES/10-180).astype('int16')

COLORMAP = {'s':'tab:blue','c':'tab:orange','p':'black'}
MARKERMAP = {'s':'o','c':'s','p':'^'}
