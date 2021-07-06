from multiprocessing import Pool
from typing import List, Iterable, Tuple, Optional

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models.keyedvectors import KeyedVectors
from pyemd import emd
from scipy.spatial.distance import cdist
import numpy as np


WMD_W2V_MODEL: Optional[KeyedVectors] = None
WMD_DICTIONARY: Optional[Dictionary] = None
WMD_TFIDF_MODEL: Optional[TfidfModel] = None


def _get_wmds_worker(args: Tuple[List[str], List[str]]) -> float:
    document1, document2 = args
    distance = WMD_W2V_MODEL.wmdistance(document1, document2)
    return distance


def get_wmds(w2v_model: KeyedVectors, tokenized_texts: Iterable[Tuple[List[str], List[str]]]) -> List[float]:
    global WMD_W2V_MODEL
    WMD_W2V_MODEL = w2v_model
    distances = []
    with Pool(None) as pool:
        for distance in pool.imap(_get_wmds_worker, tokenized_texts):
            distances.append(distance)
    WMD_W2V_MODEL = None
    return distances


def _get_wmds_tfidf_worker(args: Tuple[List[str], List[str]]) -> float:
    document1, document2 = args

    document1 = [token for token in document1 if token in WMD_W2V_MODEL]
    document2 = [token for token in document2 if token in WMD_W2V_MODEL]

    if not document1 or not document2:
        return float('inf')

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(Dictionary([document1, document2]))

    if vocab_len == 1:
        # Both documents are composed of a single unique token => zero distance.
        return 0.0

    doclist1 = list(set(document1))
    doclist2 = list(set(document2))
    v1 = np.array([WMD_W2V_MODEL.get_vector(token, norm=True) for token in doclist1])
    v2 = np.array([WMD_W2V_MODEL.get_vector(token, norm=True) for token in doclist2])
    doc1_indices = dictionary.doc2idx(doclist1)
    doc2_indices = dictionary.doc2idx(doclist2)

    # Compute distance matrix.
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    distance_matrix[np.ix_(doc1_indices, doc2_indices)] = cdist(v1, v2)

    if abs(np.sum(distance_matrix)) < 1e-8:
        # `emd` gets stuck if the distance matrix contains only zeros.
        return float('inf')

    def nbow(document):
        d = np.zeros(vocab_len, dtype=np.double)
        nbow = [
            (dictionary.token2id[WMD_DICTIONARY.id2token[term_id]], term_weight)
            for term_id, term_weight
            in WMD_TFIDF_MODEL[WMD_DICTIONARY.doc2bow(document)]
        ]
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized term weights.
        return d

    # Compute nBOW representation of documents. This is what pyemd expects on input.
    d1 = nbow(document1)
    d2 = nbow(document2)

    return emd(d1, d2, distance_matrix)


def get_wmds_tfidf(w2v_model: KeyedVectors, dictionary: Dictionary, tfidf_model: TfidfModel,
                   tokenized_texts: Iterable[Tuple[List[str], List[str]]]) -> List[float]:
    global WMD_W2V_MODEL, WMD_DICTIONARY, WMD_TFIDF_MODEL
    WMD_W2V_MODEL = w2v_model
    WMD_DICTIONARY = dictionary
    WMD_TFIDF_MODEL = tfidf_model
    distances = []
    with Pool(None) as pool:
        for distance in pool.imap(_get_wmds_tfidf_worker, tokenized_texts):
            distances.append(distance)
    WMD_W2V_MODEL = None
    WMD_DICTIONARY = None
    WMD_TFIDF_MODEL = None
    return distances
