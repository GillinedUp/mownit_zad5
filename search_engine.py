import re
import numpy as np
import pickle
from math import floor
from os import listdir
from numpy.linalg import inv
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix, save_npz, load_npz


# custom tokenizer with stemmer
class PorterTokenizer(object):
    def __init__(self):
        self.pt = PorterStemmer()

    def __call__(self, doc):
        return [self.pt.stem(t) for t in RegexpTokenizer(r'(?u)\b\w\w+\b').tokenize(doc)]


# load saved info
def load_list(path, filename):
    l = []
    with open(path + filename, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            l.append(line)
    return l


def load_info(path, matrix_filename, file_list_filename, voc_filename, lsa_filename, matrix_t_filename):
    tfidf_matrix = load_npz(file=path + matrix_filename + '.npz')
    with open(path + matrix_t_filename, 'rb') as f:
        tfidf_matrix_t = np.load(f)
    file_list = load_list(path, file_list_filename)
    voc = load_list(path, voc_filename)
    with open('lsa.pickle', 'rb') as f:
        lsa = pickle.load(f)
    return tfidf_matrix, file_list, voc, lsa, tfidf_matrix_t


class SearchEngine:

    def __int__(self, resources_path, matrix_filename, matrix_t_filename, file_list_filename, voc_filename):
        self.tfidf_matrix = load_npz(file=resources_path + matrix_filename + '.npz')
        with open(resources_path + matrix_t_filename, 'rb') as f:
            self.tfidf_matrix_t = np.load(f)
        self.file_list = load_list(resources_path, file_list_filename)
        self.voc = load_list(resources_path, voc_filename)
        with open('lsa.pickle', 'rb') as f:
            self.lsa = pickle.load(f)





# read each file into separate string

def read_docs(art_path):
    file_list = [name for name in listdir(art_path)]
    doc_list = []
    for file in file_list:
        with open(art_path + file, 'r') as myfile:
            doc_list.append(myfile.read())
    return file_list, doc_list


# vectorize strings using tf-idf vectorizer

def tfidf_transform(doc_list):
    tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', tokenizer=PorterTokenizer(), smooth_idf=True)
    tfidf_trans_matrix = tfidf_vectorizer.fit_transform(doc_list)
    return tfidf_trans_matrix, list(tfidf_vectorizer.vocabulary_.keys())


# perform low rank approximation with given multiplier

def lra_mult(tfidf_matrix, mult):
    (_, n_org) = tfidf_matrix.shape
    n_comp = floor(n_org * mult)
    if n_comp < 2:
        n_comp = 2
    svd = TruncatedSVD(n_comp, "arpack")
    reduced_m = svd.fit_transform(tfidf_matrix)
    return reduced_m


def lra(tfidf_matrix, n_comp):
    if n_comp < 2:
        n_comp = 2
    svd = TruncatedSVD(n_comp, "arpack")
    reduced_m = svd.fit_transform(tfidf_matrix)
    return reduced_m


# save info

def save_list(path, filename, l):
    with open(path + filename, 'w+') as f:
        for item in l:
            f.write("%s\n" % item)


def save_info(path, tfidf_matrix, file_list, voc, lsa, tfidf_matrix_t):
    save_npz(file=path + 'tfidf_matrix', matrix=tfidf_matrix)
    with open(path + 'tfidf_matrix_t', 'wb') as f:
        np.save(f, tfidf_matrix_t)
    save_list(path, 'voc', voc)
    save_list(path, 'file_list', file_list)
    with open('lsa.pickle', 'wb') as f:
        pickle.dump(lsa, f, pickle.HIGHEST_PROTOCOL)





# search

def search(search_str, n, tfidf_matrix, voc, file_list):
    tfidf_search_vectorizer = TfidfVectorizer(vocabulary=voc, stop_words='english', tokenizer=PorterTokenizer(),
                                              smooth_idf=True)
    tfidf_search_matrix = tfidf_search_vectorizer.fit_transform(search_str)
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_search_matrix)
    a = [i[0] for i in sorted(enumerate(list(sim_matrix[:, 0])), key=lambda x: x[1], reverse=True)]
    res = [file_list[i] for i in a]
    return res[:n]


def lsa_learn(tfidf_matrix, n_comp):
    lsa = TruncatedSVD(n_components=n_comp)
    tfidf_matrix_t = lsa.fit_transform(tfidf_matrix)
    tfidf_matrix_t = Normalizer(copy=False).fit_transform(tfidf_matrix_t)
    return lsa, tfidf_matrix_t


def lsa_search(query_str, n, tfidf_matrix_t, voc, lsa, file_list):
    tfidf_query_vectorizer = TfidfVectorizer(vocabulary=voc, stop_words='english', tokenizer=PorterTokenizer(),
                                             smooth_idf=True)
    tfidf_query_matrix = tfidf_query_vectorizer.fit_transform(query_str)
    query = lsa.transform(tfidf_query_matrix)
    sim_matrix = cosine_similarity(tfidf_matrix_t, query)
    a = [i[0] for i in sorted(enumerate(list(sim_matrix[:, 0])), key=lambda x: x[1], reverse=True)]
    res = [file_list[i] for i in a]
    return res[:n]