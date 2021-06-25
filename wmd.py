from typing import List

from gensim.models.fasttext import load_facebook_vectors

from common import Metric, Judgements
import nltk
from nltk.corpus import stopwords


class WMD(Metric):

    label = "WMD"
    w2v_model = None
    stopwords = None

    def __init__(self, tgt_lang: str):
        if tgt_lang == "en":
            self.w2v_model = load_facebook_vectors('embeddings/cc.en.300.bin')
            self.w2v_model.init_sims(replace=True)
            nltk.download('stopwords')
            self.stopwords = stopwords.words('english')

        else:
            raise ValueError(tgt_lang)

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        pass

    def compute(self, judgements: Judgements) -> List[float]:
        out_scores = [self.w2v_model.wmdistance(reference_words, translation_words)
                      for reference_words, translation_words
                      in judgements.get_tokenized_texts(self.stopwords, desc=self.label)]
        return out_scores
