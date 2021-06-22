from typing import List

from bert_score import BERTScorer

from common import Metric, Judgements


class BERTScore(Metric):

    def __init__(self, tgt_lang: str):
        self.scorer = BERTScorer(lang=tgt_lang, rescale_with_baseline=True)

    def fit(self, judgements: Judgements):
        pass

    def compute(self, judgements: Judgements) -> List[float]:
        P, R, F1 = self.scorer.score(judgements.translations, judgements.references)

        return F1
