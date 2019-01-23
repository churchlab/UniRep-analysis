"""
Usage:

from vectorize_for_benchmarking_lib import from_sequence_csv_to_X_and_dict, get_representations
import pandas as pd

path = "./path_to_training_sequences.csv"
path2 = "./path_to_test_sequences.csv"

X, dictionary, t = from_sequence_csv_to_X_and_dict(path)
X_tfidf, dictionary, t = from_sequence_csv_to_X_and_dict(path, ngramness=3, tfidf=True)
X_1000, dictionary, t = from_sequence_csv_to_X_and_dict(path, keep_n=1000) # limit bag of words to 1000 most frequent ngrams

#BUILDING REPRESENTATION FOR ANOTHER DATASET (I.E. TEST SET) AFTER "TRAINING":

#first, call it on the first dataset
X, dictionary_X, model_tfidf_X = from_sequence_csv_to_X_and_dict(path, tfidf=True)
#then pass that dictionary when you want representations for the second dataset
array_of_strings = pd.read_csv(path2).values[:,0]
Y, d, t = get_representations(array_of_strings, tfidf=True, dictionary=dictionary_X, model_tfidf = model_tfidf_X)

"""


import pandas as pd
import numpy as np
import nltk
from gensim import corpora, similarities, models
from gensim.matutils import corpus2dense
from Bio.SeqUtils.ProtParam import ProteinAnalysis

aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def ngrams(string,n): 
    return [''.join(x) for x in nltk.ngrams(list(string),n=n)]

def get_aa_percentage_vectors(X):
    res = np.zeros((X.shape[0], 20))
    for i,seq in enumerate(X):
        analysed_seq = ProteinAnalysis(seq)
        res[i] = pd.Series(analysed_seq.get_amino_acids_percent())[
            aas # to ensure the same order every time just in case
        ].values
    return res

def get_biopython_features(X):
    res = np.zeros((X.shape[0], 6))
    for i,seq in enumerate(X):
        analysed_seq = ProteinAnalysis(seq)
        res[i] = np.array([analysed_seq.molecular_weight()]+[analysed_seq.instability_index()] + [analysed_seq.isoelectric_point()] + list(analysed_seq.secondary_structure_fraction()))
        
    return res

def get_representations(X, ngramness=3, tfidf=False, keep_n=None, dictionary=False, model_tfidf=False):
    
    # reference for different transformations: https://radimrehurek.com/gensim/tut2.html
    # optimal num_topics = 200–500
    
    n = ngramness

    texts = [ngrams(string, n) for string in X]
    
    if not dictionary:
    	dictionary = corpora.Dictionary(texts)
    	dictionary.filter_extremes(keep_n=keep_n)

    corpus = [dictionary.doc2bow(text) for text in texts]
    
    
    if not model_tfidf:
        model_tfidf = models.TfidfModel(corpus)
    
    if tfidf:
    	corpus = model_tfidf[corpus]
    else:
    	model_tfidf = None
    	
    
    X = corpus2dense(corpus, num_terms=len(dictionary.token2id)).T # to make each protein be a row
    
    """
    PARAMS: lda=False, lsa=False, num_topics=200,
    if lsa:
        if lda:
                print("For getting benchmark vectorizations, can't do both lda and lsa - pick one")
                raise
        model = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
        corpus = model[corpus]
        X = corpus2dense(corpus, num_terms=num_topics).T # to make each protein be a row
    else:
        if lda:
            if tfidf:
                print("For getting benchmark vectorizations, doing both lda and tfidf does not make sense. try lsa+tfidf combo instead")
                raise
            else:
                model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
                corpus = model[corpus]
                X = corpus2dense(corpus, num_terms=num_topics).T # to make each protein be a row
     """
    
    return X.tolist(), dictionary, model_tfidf
    
def from_sequence_csv_to_X_and_dict(path, ngramness=3, tfidf=False, keep_n=None, dictionary=False, model_tfidf=False):

    seqs = pd.read_csv(path).values[:,0]

    return get_representations(seqs, ngramness=ngramness, tfidf=tfidf, keep_n=keep_n, dictionary=dictionary, model_tfidf=model_tfidf)
    
# THESE FUNCTIONS ARE NOT GENERAL - USE WITH CARE - expect X to be a Series of string aminoacid sequences
    
def get_dumb_features(X):
    arr = np.concatenate((get_aa_percentage_vectors(X),  # frequencies 
                    get_biopython_features(X), # couple of simple biophysical features 
                    X.map(len).values.reshape(X.shape[0],1)), axis=1) # length of protein
    return arr.tolist()

def get_freq_and_len(X):
    arr = np.concatenate((get_aa_percentage_vectors(X),  # frequencies 
                    X.map(len).values.reshape(X.shape[0],1) / float(337) ), axis=1) # length of protein divided by avg length of prot in uniprot
    return arr.tolist()
    
    
    
