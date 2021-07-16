from typing import List
from tqdm.autonotebook import tqdm

import nltk

from common import Metric, Judgements
from sacrebleu import corpus_bleu
from nltk.translate.meteor_score import meteor_score


class BLEU(Metric):

    label = "BLEU"

    def compute(self, judgements: Judgements) -> List[float]:
        return [
            max([corpus_bleu(actual_item, expected_item).score for expected_item in expected_items])
            for actual_item, expected_items in tqdm(zip(judgements.translations, judgements.references),
                                                    desc=self.label, total=len(judgements))]


class METEOR(Metric):

    label = "METEOR"

    def __init__(self):
        nltk.download('wordnet', quiet=True)

    def compute(self, judgements: Judgements) -> List[float]:
        return [meteor_score(expected_items, actual_item)
                for actual_item, expected_items in tqdm(zip(judgements.translations, judgements.references),
                                                        desc=self.label, total=len(judgements))]
