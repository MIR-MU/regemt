from itertools import chain
from typing import List

from gensim.corpora import Dictionary
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix
from nltk.corpus import stopwords
from gensim.models.fasttext import load_facebook_vectors
import nltk

from common import Metric, Judgements


class SCM(Metric):
    label = "SCM"
    w2v_model = None
    stopwords = None
    similarity_matrix = None
    dictionary = None
    tfidf = None

    def __init__(self, tgt_lang: str, use_tfidf: bool):
        if tgt_lang == "en":
            nltk.download('stopwords')
            self.stopwords = stopwords.words('english')
        else:
            raise ValueError(tgt_lang)

        self.use_tfidf = use_tfidf
        if use_tfidf:
            self.label = self.label + "_tfidf"

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        tokenized_texts = train_judgements.get_tokenized_texts(self.stopwords)
        if test_judgements != train_judgements:
            tokenized_texts = chain(tokenized_texts, test_judgements.get_tokenized_texts(self.stopwords))
        reference_corpus, translation_corpus = map(list, zip(*tokenized_texts))
        corpus = reference_corpus + translation_corpus

        self.dictionary = Dictionary(corpus)
        noncontextual_embeddings = load_facebook_vectors('embeddings/cc.en.300.bin').wv
        word_similarity_index = WordEmbeddingSimilarityIndex(noncontextual_embeddings)

        if self.use_tfidf:
            from gensim.models import TfidfModel
            self.tfidf = TfidfModel(dictionary=self.dictionary)
            self.similarity_matrix = SparseTermSimilarityMatrix(word_similarity_index, self.dictionary, self.tfidf)
        else:
            self.similarity_matrix = SparseTermSimilarityMatrix(word_similarity_index, self.dictionary)

    def compute(self, judgements: Judgements) -> List[float]:
        # https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
        out_scores = []
        for reference_words, translation_words in judgements.get_tokenized_texts(self.stopwords, desc=self.label):
            if self.use_tfidf:
                ref_index = self.tfidf[self.dictionary.doc2bow(reference_words)]
                trans_index = self.tfidf[self.dictionary.doc2bow(translation_words)]

                # take only the top 50% most-important terms
                safe_sum = len(ref_index + trans_index) if len(ref_index + trans_index) > 0 else 1
                threshold_tfidf = sum([val for idx, val in ref_index + trans_index]) / safe_sum
                ref_index = [(idx, val) for idx, val in ref_index if val >= threshold_tfidf]
                trans_index = [(idx, val) for idx, val in trans_index if val >= threshold_tfidf]
            else:
                ref_index = self.dictionary.doc2bow(reference_words)
                trans_index = self.dictionary.doc2bow(translation_words)

            out_scores.append(self.similarity_matrix.inner_product(ref_index, trans_index, normalized=(True, True)))

        return out_scores
