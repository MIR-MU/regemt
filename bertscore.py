from typing import List

from bert_score import BERTScorer
from tqdm import tqdm

from common import Metric, Judgements


class BERTScore(Metric):
    label = "BERTScore"

    def __init__(self, tgt_lang: str):
        self.scorer = BERTScorer(lang=tgt_lang, rescale_with_baseline=True)

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        pass

    def compute(self, judgements: Judgements) -> List[float]:
        f_scores = [self.scorer.score([trans], [ref])[-1][0]
                    for trans, ref in tqdm(zip(judgements.translations, judgements.references),
                                           desc="BERTScore", total=len(judgements))]

        return f_scores
