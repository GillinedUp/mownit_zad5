import numpy as np
import pickle
from math import floor
from os import listdir
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from scipy.sparse import save_npz, load_npz


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


class SearchEngine:

    def __int__(self, resources_path):
        self.tfidf_matrix = load_npz(file=resources_path + 'tfidf_matrix.npz')
        with open(resources_path + 'tfidf_matrix_t', 'rb') as f:
            self.tfidf_matrix_t = np.load(f)
        self.file_list = load_list(resources_path, 'file_list')
        self.voc = load_list(resources_path, 'voc')
        with open('lsa.pickle', 'rb') as f:
            self.lsa = pickle.load(f)

    def search(self, search_str, n):
        tfidf_search_vectorizer = TfidfVectorizer(vocabulary=self.voc,
                                                  stop_words='english',
                                                  tokenizer=PorterTokenizer(),
                                                  smooth_idf=True)
        tfidf_search_matrix = tfidf_search_vectorizer.fit_transform(search_str)
        sim_matrix = cosine_similarity(self.tfidf_matrix, tfidf_search_matrix)
        a = [i[0] for i in sorted(enumerate(list(sim_matrix[:, 0])),
                                  key=lambda x: x[1],
                                  reverse=True)]
        res = [self.file_list[i] for i in a]
        return res[:n]

    def lsa_search(self, query_str, n):
        tfidf_query_vectorizer = TfidfVectorizer(vocabulary=self.voc,
                                                 stop_words='english',
                                                 tokenizer=PorterTokenizer(),
                                                 smooth_idf=True)
        tfidf_query_matrix = tfidf_query_vectorizer.fit_transform(query_str)
        query = self.lsa.transform(tfidf_query_matrix)
        sim_matrix = cosine_similarity(self.tfidf_matrix_t, query)
        a = [i[0] for i in sorted(enumerate(list(sim_matrix[:, 0])),
                                  key=lambda x: x[1],
                                  reverse=True)]
        res = [self.file_list[i] for i in a]
        return res[:n]


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


def lsa_learn(tfidf_matrix, n_comp):
    lsa = TruncatedSVD(n_components=n_comp)
    tfidf_matrix_t = lsa.fit_transform(tfidf_matrix)
    tfidf_matrix_t = Normalizer(copy=False).fit_transform(tfidf_matrix_t)
    return lsa, tfidf_matrix_t


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


class SearchIndexer:

    def __int__(self, art_path):
        self.art_path = art_path
        self.file_list, self.doc_list = read_docs(self.art_path)
        self.tfidf_matrix, self.voc = tfidf_transform(self.doc_list)
        self.lsa, self.tfidf_matrix_t = lsa_learn(self.tfidf_matrix, 100)

    def save(self, path):
        save_npz(file=path + 'tfidf_matrix', matrix=self.tfidf_matrix)
        np.save(path + 'tfidf_matrix_t', self.tfidf_matrix_t)
        save_list(path, 'voc', self.voc)
        save_list(path, 'file_list', self.file_list)
        with open('lsa.pickle', 'wb') as f:
            pickle.dump(self.lsa, f, pickle.HIGHEST_PROTOCOL)
