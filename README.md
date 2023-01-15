# synonyms-semchange
Repository for the study of Semantic Change of synonyms

## Files

Scripts and notebooks, in the order they were used:
- ` context_target_words.ipynb`: select context words and targets (following all criterions except the cooccurrence threshold).
- `get_GNgrams_matrices.py`: read the Google Books Ngrams corpus to store cooccurrence matrices for targets and contexts.
- ` targets_final_selection.ipynb`: apply the "above 1% of cooccurring contexts" filter to have the reduced target list.
- `rearranging_matrices.ipynb`: copy the *old* co-occurrences matrices and reduced them according to the final target list.
- `synonyms_pairs_extraction.ipynb`: select synonymous pairs.

Input and output files:
- `fernald.txt`: Fernald's English Synonyms and Antonyms book, in raw txt format. Used as input for *synonyms_pairs_extraction.ipynb*.
- `contexts_list.txt` (produced by *context_target_words.ipynb*): list of context words. 
- `candidates_target_list.csv` (produced by *context_target_words.ipynb*): table containing potential target words, their frequency in the CCOHA for each decade and their POS tag.

- `targets_list.csv` (produced by *targets_final_selection.ipynb*): table containing target words (after the full selection process), their POS tag, their index in the old global matrices, and the number of different contexts with which they appear.

- `nouns_list.pickle`,`adjs_list.pickle`,`verbs_list.pickle` (produced by *rearranging_matrices.ipynb*): lists of target nouns, adjectives and verbs in *pickle* file.


distance_matrix.py
nearest_neighbors.py
semchange.py
frequency.py
words_labelling.py
synpairs_analysis.py

pairs_selection.py
synpairs_measures.py
    synpairs_ldlpc_results.py
OR
    synpairs_graphs.py
    synpairs_closest.py