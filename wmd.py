from typing import List

from gensim.parsing import preprocess_string
from gensim.utils import simple_preprocess
from tqdm import tqdm
from gensim.models.fasttext import load_facebook_vectors

from common import Metric, Judgements
import gensim.downloader as api
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
        out_scores = []
        for reference, translation in tqdm(zip(judgements.references, judgements.translations),
                                           desc="WMD", total=len(judgements)):

            reference_words = [w.lower() for w in simple_preprocess(reference[0]) if w.lower() not in self.stopwords]
            translation_words = [w.lower() for w in simple_preprocess(translation) if w.lower() not in self.stopwords]

            out_scores.append(self.w2v_model.wv.wmdistance(reference_words, translation_words))
        return out_scores
