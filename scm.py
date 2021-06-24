from typing import List

from gensim.corpora import Dictionary, dictionary
from gensim.utils import simple_preprocess
from gensim.similarities import WordEmbeddingSimilarityIndex, SparseTermSimilarityMatrix
from nltk.corpus import stopwords
import gensim.downloader as api
from gensim.models.fasttext import load_facebook_model
from tqdm import tqdm
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
            self.w2v_model = load_facebook_model('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz')
            self.w2v_model.init_sims(replace=True)
            nltk.download('stopwords')
            self.stopwords = stopwords.words('english')
        else:
            raise ValueError(tgt_lang)

        self.use_tfidf = use_tfidf
        if use_tfidf:
            self.label = self.label + "_tfidf"

    def fit(self, judgements: Judgements):
        self.dictionary = Dictionary([[t.lower() for t in simple_preprocess(refs[0])] for refs in judgements.references])
        similarity_index = WordEmbeddingSimilarityIndex(self.w2v_model)

        if self.use_tfidf:
            from gensim.models import TfidfModel
            self.tfidf = TfidfModel(dictionary=self.dictionary)
            self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary, self.tfidf)
        else:
            self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary)

    def compute(self, judgements: Judgements) -> List[float]:
        # https://stackoverflow.com/questions/59573454/soft-cosine-similarity-between-two-sentences
        out_scores = []
        for reference, translation in tqdm(zip(judgements.references, judgements.translations),
                                           desc="SCM", total=len(judgements)):
            reference_words = [w.lower() for w in simple_preprocess(reference[0]) if w.lower() not in self.stopwords]
            translation_words = [w.lower() for w in simple_preprocess(translation) if w.lower() not in self.stopwords]

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
