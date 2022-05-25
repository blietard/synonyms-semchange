import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from collections import defaultdict
import requests
import pickle
import os
import time
import gzip


data_folder = './data/'
matrices_folder = './matrices/'
inout_folder = './'


def regroup_by_decade(occurences):
    grouped_occ = defaultdict(int)
    for triplet in occurences:
        year,count,vol = triplet.split(',')
        year = int(year)
        if year >= 1890:
            if year < 2000:
                decade = int(np.floor(year/10)*10)
                grouped_occ[decade] += int(count)
    return grouped_occ

class CooConstructor:
    def __init__(self,folder=''):
        self.folder = folder
        #rows
        if os.path.exists(folder+'rows'):
            with open(folder+'rows','rb') as f:
                self.row_inds = pickle.load(f)
        else:
            self.row_inds = []
        #values
        if os.path.exists(folder+'values'):
            with open(folder+'values','rb') as f:
                self.values = pickle.load(f)
        else:
            self.values = []
        # columns indices
        if os.path.exists(folder+'columns'):
            with open(folder+'columns','rb') as f:
                self.col_inds = pickle.load(f)
        else:
            self.col_inds = []
        
    def add_entry(self,row,col,val):
        self.row_inds.append(row)
        self.col_inds.append(col)
        self.values.append(val)
        

    def store(self):
        with open(self.folder+'rows','wb') as f:
            pickle.dump(self.row_inds, f)
        with open(self.folder+'values','wb') as f:
            pickle.dump(self.values, f)
        with open(self.folder+'columns','wb') as f:
            pickle.dump(self.col_inds, f)
        
    def to_coo(self,shape):
        return coo_matrix( (self.values, (self.row_inds,self.col_inds) ) , shape)


if __name__ == "__main__":

    decades = np.array(range(1890,2000,10))

    for decade in decades:
        try :
            os.mkdir(matrices_folder+str(decade)+'/')
        except FileExistsError:
            pass

        for pos in {'N','A','V'}:
            try:
                os.mkdir(matrices_folder+str(decade)+'/'+pos+'/')
            except FileExistsError:
                pass
            try:
                os.mkdir(matrices_folder+str(decade)+'/'+pos+'/L/')
            except FileExistsError:
                pass
            try:
                os.mkdir(matrices_folder+str(decade)+'/'+pos+'/R/')
            except FileExistsError:
                pass
            

    targets_df = pd.read_csv(inout_folder + 'candidates_target_list.csv',delimiter='\t')
    targets = dict(targets_df.groupby( ['word-cs','PoS'] ).groups.keys())
    for i,word in enumerate(targets.keys()):
        targets[word] = ( targets[word], i )
    n_rows = len(targets)
    print("Number of targets :\t", n_rows)
    print('sample :\t  {"human" :',targets['human'],'}')

    with open(inout_folder + 'contexts_list.txt','r',encoding='utf-8') as f:
        contexts = f.read().split('\n')
    contexts = [context for context in contexts if context] # remove potential empty line
    contexts = dict([(context,i) for i,context in enumerate(contexts)]) # turn to dict with indices
    n_cols = len(contexts)
    print('Number of contexts :\t', n_cols)
    print('sample :\t {"human" :',contexts['human'],'}')

    file_numbers = list(range(589))
    if os.path.exists(inout_folder + 'Gngrams_processor.log'):
        with open( inout_folder + 'Gngrams_processor.log','r') as log:
            loglines = [line for line in log.read().split('\n') if line]
        last_processed = int(loglines[-1].split()[-1])
        file_numbers = list(range(last_processed+1,589))
        print(f'Resuming at file {last_processed+1}.')

    for fnumber in file_numbers:
        start_time = time.time()
        fnumber_as_str = str(1000+fnumber)[1:] #to fill with 0s
        fname = '2-00' + fnumber_as_str + '-of-00589.gz'
        if os.path.exists(data_folder +fname):
            print('[INFO:] Re-using file '+fname+'.')
        else:
            print('[INFO:] Downloading file '+fname+' ...')
            response = requests.get('http://storage.googleapis.com/books/ngrams/books/20200217/eng/'+fname)
            with open(data_folder+fname,'wb') as gz:
                gz.write(response.content)
            print('[INFO:] Downloaded and stored.')
        print('[INFO:] Processing file '+fname+' ...')
        i = 0
        L_matrices = dict()
        R_matrices = dict()
        for decade in decades:
            L_matrices[decade]= { PoS : CooConstructor(folder=matrices_folder+str(decade)+'/'+PoS+'/L/') for PoS in {'N','A','V'} }
            R_matrices[decade]= { PoS : CooConstructor(folder=matrices_folder+str(decade)+'/'+PoS+'/R/') for PoS in {'N','A','V'} }

        with gzip.open(data_folder +fname,'rt') as gz:
            while True:
                line = gz.readline()
                if line == '':
                    break
                i += 1
                print(f'line : {i}',end='\r')
                line = line[:-1].split('\t')
                w1,w2 = line.pop(0).split()
                w1 = w1.split('_')[0].lower()
                w2 = w2.split('_')[0].lower()
                if w1 in contexts:
                    if w2 in targets:
                        # Left context : c,w
                        col = contexts[w1]
                        pos, row = targets[w2]
                        decade_counts = regroup_by_decade(occurences=line)
                        for decade,count in decade_counts.items():
                            L_matrices[decade][pos].add_entry( row,col,count )

                if w1 in targets:
                    if w2 in contexts :
                        # Right context : w,c
                        pos, row = targets[w1]
                        col = contexts[w2]
                        decade_counts = regroup_by_decade(occurences=line)
                        for decade,count in decade_counts.items():
                            R_matrices[decade][pos].add_entry( row,col,count )
        end_time = time.time()
        print(f'\n[INFO:] Processed file {fname} in {end_time - start_time}s')
        print('[INFO:] Creating data back-up...')
        for decade in decades:
            for PoS in {'N','A','V'}:
                L_matrices[decade][pos].store()
                R_matrices[decade][pos].store()
        if os.path.exists('./Gngrams_processor.log'):
            with open('./Gngrams_processor.log','r') as log:
                logtxt = log.read()
        else:
            logtxt = ''
        logtxt += 'Finished processing file '+str(fnumber) + '\n'
        with open('./Gngrams_processor.log','w') as log:
            log.write(logtxt)
        print('[INFO:] Backup done !')
        os.remove(data_folder +fname)
        print('[INFO:] Removed file '+fname+'.')