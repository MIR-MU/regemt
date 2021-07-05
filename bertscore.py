from typing import List

from bert_score import BERTScorer
from tqdm.autonotebook import tqdm

from common import Metric, Judgements


class BERTScore(Metric):
    label = "BERTScore"

    def __init__(self, tgt_lang: str, batch_size: int = 32):
        self.scorer = BERTScorer(lang=tgt_lang, rescale_with_baseline=True)
        self.batch_size = batch_size

    def fit(self, train_judgements: Judgements, test_judgements: Judgements):
        pass

    def compute(self, judgements: Judgements) -> List[float]:
        f_scores = []

        batch_iter = range(0, len(judgements), self.batch_size)
        for i, j in tqdm(((batch_i, batch_i+self.batch_size) for batch_i in batch_iter),
                         desc=self.label, total=int(len(judgements) / self.batch_size)):
            b_prec, b_rec, b_f_scores = self.scorer.score(judgements.translations[i:j], judgements.references[i:j])
            f_scores.extend(b_f_scores.detach().cpu().tolist())

        return f_scores
