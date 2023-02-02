import pickle
from tqdm import tqdm
from collections import defaultdict
from params import SOURCE_SYNDICT, WORDS_FOLDER

with open(SOURCE_SYNDICT,'r') as f:
    text = f.read()
lines = text.split('\n')

def extract_next_entry_section(lines,start,end):
    '''
    Search an entry section in the list of lines `lines` between indices `start` and `end` (included).
    Return the line number at the end of the entry section, and the entry section itself.
    '''
    searching_entry = True
    current_nline = start
    # Searching the first entry
    while searching_entry and current_nline<end:
        line = lines[current_nline]
        if line == '       *       *       *       *       *':
            searching_entry = False
        current_nline +=1
    entry_start = current_nline
    # Searching the next entry
    searching_next_entry = True
    while searching_next_entry and current_nline<end:
        line = lines[current_nline]
        if line == '       *       *       *       *       *':
            searching_next_entry = False
        current_nline +=1
    entry_end = current_nline-1
    return (entry_end,lines[entry_start:entry_end])
    
def get_synonyms(entry_section):
   '''
   Return the list of synonymous pairs found in an entry section.
   A pair is (ENTRY, synonymous word).
   `entry_section` must be a list of text lines.
   '''
   syns = list()
   
   headline = entry_section[1].split(',')

   if len(headline) == 1:
      entry = headline[0][:-1].lower()
   elif len(headline) > 1:
      entry = ''.join( [ head_part.strip() for head_part in headline]).lower()
   else:
      raise ValueError('No entry found.')

   if ' ' in entry:
      print(entry_section[1])
      raise ValueError('Entry is not a single word.')

   syn_list_start = 0
   while entry_section[syn_list_start]!='Synonyms:' and entry_section[syn_list_start]!='Synonym:':
      syn_list_start +=1
   syn_list_start += 2 #Skip 'Synonyms:' and following empty line.
   syn_list_end = syn_list_start
   while entry_section[syn_list_end]: #Search next empty line
      syn_list_end +=1
   synonyms_list = entry_section[syn_list_start:syn_list_end]

   for line in synonyms_list:
      words = [word for word in line.split(',') if word] #remove empty
      for word in words:
         syns.append( word.strip() )
   
   last_syn = syns.pop(-1)
   syns.append( last_syn[:-1] ) #remove punctation at the end.

   syns = [syn_word for syn_word in syns.copy() if ' ' not in syn_word  ] #remove compound word synonyms

   return (entry,syns)



partI_range = (453,22058)
syn_pairs = dict()
nline = partI_range[0]
while nline < partI_range[1]:
    nline, entry_section = extract_next_entry_section(lines, nline, partI_range[1])
    if not entry_section:
        break
    entry, syns = get_synonyms(entry_section)
    syn_pairs[entry] = syns

with open(f'{WORDS_FOLDER}/ADJ_list.pkl','rb') as f:
    adjs = set(pickle.load(f))
with open(f'{WORDS_FOLDER}/N_list.pkl','rb') as f:
    nouns = set(pickle.load(f))
with open(f'{WORDS_FOLDER}/V_list.pkl','rb') as f:
    verbs = set(pickle.load(f))


final_syn_pairs = { 'ADJ': dict(), 'N': dict() ,'V':dict()}

for full_entry, syns in tqdm(syn_pairs.items()):
    full_entry = full_entry.split('_')
    if len(full_entry)>1:
        entry, pos = full_entry[:2]
    else:
        entry = full_entry[0]
        pos = ''

    # ADJS
    if entry in adjs and (pos=='a.' or not pos):
        select = list()
        for syn in syns:
            if syn in adjs:
                select.append(syn)
        if select:
            final_syn_pairs['ADJ'][entry] = select.copy()

    # NOUNS
    if entry in nouns and (pos=='n.' or not pos):
        select = list()
        for syn in syns:
            if syn in nouns:
                select.append(syn)
        if select:
            final_syn_pairs['N'][entry] = select.copy()

    # VERBS
    if entry in verbs and (pos=='v.' or not pos):
        select = list()
        for syn in syns:
            if syn in verbs:
                select.append(syn)
        if select:
            final_syn_pairs['V'][entry] = select.copy()

for pos in final_syn_pairs.keys():
    print(f'PoS : {pos}, nb heads : {len(final_syn_pairs[pos])}, nb pairs: {sum( [ len(syns) for syns in final_syn_pairs[pos].values()] )}' )

with open(f'{WORDS_FOLDER}/source_synonyms.pkl','wb') as f:
    pickle.dump(final_syn_pairs, f)