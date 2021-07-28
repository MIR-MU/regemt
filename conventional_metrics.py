from typing import List, Any
from functools import lru_cache

from tqdm.autonotebook import tqdm
import nltk
from .common import Metric, Judgements
from sacrebleu import corpus_bleu
from nltk.translate.meteor_score import meteor_score


class BLEU(Metric):

    label = "BLEU"

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        return [
            max([corpus_bleu(actual_item, expected_item).score for expected_item in expected_items])
            for actual_item, expected_items in tqdm(zip(judgements.translations, judgements.references),
                                                    desc=self.label, total=len(judgements))]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BLEU):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(True)


class METEOR(Metric):

    label = "METEOR"

    def __init__(self):
        nltk.download('wordnet', quiet=True)

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        return [meteor_score(expected_items, actual_item)
                for actual_item, expected_items in tqdm(zip(judgements.translations, judgements.references),
                                                        desc=self.label, total=len(judgements))]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, METEOR):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(True)
