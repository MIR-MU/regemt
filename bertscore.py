from typing import List, Any
from functools import lru_cache

from bert_score import BERTScorer
from bert_score.utils import lang2model
from tqdm.autonotebook import tqdm

from common import ReferenceFreeMetric, Judgements


MULTILINGUAL_MODEL = lang2model[None]


class BERTScore(ReferenceFreeMetric):

    label = "BERTScore"

    def __init__(self, tgt_lang: str, batch_size: int = 32, reference_free: bool = False):
        if reference_free:
            # force to use multilingual model, presume that both source and target langs are supported
            self.scorer = BERTScorer(model_type=MULTILINGUAL_MODEL)
            self.tgt_lang = 'multi'
        else:
            # infer used model from target language -> language of both reference and translation
            self.scorer = BERTScorer(lang=tgt_lang)
            self.tgt_lang = tgt_lang if lang2model[tgt_lang] != MULTILINGUAL_MODEL else 'multi'

        self.reference_free = reference_free
        self.batch_size = batch_size

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        f_scores = []

        batch_iter = range(0, len(judgements), self.batch_size)
        for i, j in tqdm(((batch_i, batch_i+self.batch_size) for batch_i in batch_iter),
                         desc=self.label, total=len(batch_iter)):
            sources = judgements.src_texts[i:j] if self.reference_free else judgements.references[i:j]
            b_prec, b_rec, b_f_scores = self.scorer.score(judgements.translations[i:j], sources)
            f_scores.extend(b_f_scores.detach().cpu().tolist())

        return f_scores

    @staticmethod
    def supports(src_lang: str, tgt_lang: str, reference_free: bool) -> bool:
        return reference_free or tgt_lang in ('cs', 'de', 'en', 'es', 'et', 'fi', 'fr', 'it', 'lv', 'pt', 'zh')

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BERTScore):
            return NotImplemented
        return all([
            self.reference_free == other.reference_free,
            self.tgt_lang == other.tgt_lang,
            self.batch_size == other.batch_size,
        ])

    def __hash__(self) -> int:
        return hash((self.reference_free, self.tgt_lang, self.batch_size))
