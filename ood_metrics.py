from typing import Dict, Tuple, List, Callable, Iterable, Any, Optional
from functools import lru_cache
from contextlib import redirect_stdout, redirect_stderr
import os

import numpy as np
import spacy
from tqdm.autonotebook import tqdm

from common import Judgements, ReferenceFreeMetric

taggers: Dict[str, Callable[[str], Iterable[Tuple[str, str]]]] = {}


class TransitionModel:

    def __init__(self, dataset: List[str], lang: str):
        if lang not in taggers:
            tagger = self._init_tagger(lang)
            taggers[lang] = tagger
        else:
            tagger = taggers[lang]

        self.corpus_words_tagged = [list(tagger(dataset[i])) for i in range(len(dataset))]
        corpus_tags = [[tag for token, tag in tagged_seq] for tagged_seq in self.corpus_words_tagged]
        self.all_tags, self.transition_probs = self._transition_graph_from_tags(corpus_tags)

    def distance(self, other: "TransitionModel") -> float:
        matching_indices = [i for i, tag in enumerate(self.all_tags) if tag in other.all_tags]
        self_transitions_subset = self.transition_probs[matching_indices, :][:, matching_indices]

        other_matching_indices = [i for i, tag in enumerate(other.all_tags) if tag in self.all_tags]
        other_transitions = other.transition_probs
        other_transitions_subset = other_transitions[other_matching_indices, :][:, other_matching_indices]

        return np.linalg.norm(self_transitions_subset - other_transitions_subset)

    @staticmethod
    def _transition_graph_from_tags(tag_sequences: List[List[str]]) -> Tuple[List[str], np.ndarray]:
        # construct 2-grams from sequences of tags and count an occurrence of each 2-gram for the transition graph
        counts: Dict[Tuple[str, str], int] = {}
        for sequence in tag_sequences:
            for i in range(2, len(sequence)):
                tags_from_to = tuple(sequence[i-2:i])
                try:
                    counts[tags_from_to] += 1
                except KeyError:
                    counts[tags_from_to] = 1
        all_tags = sorted(set([k[0] for k in counts.keys()] + [k[0] for k in counts.keys()]))
        transition_matrix = [[counts.get((tag_x, tag_y), 0) for tag_x in all_tags] for tag_y in all_tags]
        if not transition_matrix:
            # text is a single-word tag - can happen in initial training phases
            # we need to keep the dimensionality
            transition_matrix = [[]]

        transition_matrix_np = np.array(transition_matrix)
        return all_tags, transition_matrix_np / max(transition_matrix_np.sum(), 1)

    @staticmethod
    def supports(lang: str) -> bool:
        return lang in ("no", "en", "de", "zh")

    @staticmethod
    def _init_tagger(lang: str) -> Callable[[str], Iterable[Tuple[str, str]]]:
        if lang == "no":
            model_id = "nb_core_news_md"
        elif lang == "en":
            model_id = "en_core_web_md"
        elif lang == "de":
            model_id = "de_core_news_md"
        elif lang == "zh":
            model_id = "zh_core_web_md"
        else:
            raise ValueError("Language '%s' has no defined tagger" % lang)

        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            try:
                spacy_tagger = spacy.load(model_id)
            except OSError:
                # tagger not-yet downloaded
                # spacy.cli.download(model_id, False, "-q")
                spacy.cli.download(model_id)
                spacy_tagger = spacy.load(model_id)

        def _spacy_pos_tagger_wrapper(text: str) -> Iterable[Tuple[str, str]]:
            tokens_tagged = spacy_tagger(text)
            for token in tokens_tagged:
                yield token.text, token.pos_

        return _spacy_pos_tagger_wrapper


class SyntacticCompositionality(ReferenceFreeMetric):

    pos_tagger: Callable[[str], Tuple[str, str]]
    src_lang: Optional[str] = None
    label = "Compositionality"

    def __init__(self, tgt_lang: str, src_lang: Optional[str] = None, reference_free: bool = False):
        """
        Compares syntactic compositionality's perplexity on train distribution and outer distribution.
        Syntactic compositionality is a transition matrix of PoS tags
        """
        self.tgt_lang = tgt_lang
        self.reference_free = reference_free

        if reference_free:
            self.src_lang = src_lang

    @staticmethod
    def supports(src_lang: str, tgt_lang: str, reference_free: bool) -> bool:
        return TransitionModel.supports(tgt_lang) and (not reference_free or TransitionModel.supports(src_lang))

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        if self.reference_free:
            base_transitions = [TransitionModel([src_text], self.src_lang)
                                for src_text in tqdm(judgements.src_texts, desc=self.label)]
        else:
            base_transitions = [TransitionModel([ref_texts[0]], self.tgt_lang)
                                for ref_texts in tqdm(judgements.references, desc=self.label)]

        translated_model = [TransitionModel([t_text], self.tgt_lang)
                            for t_text in tqdm(judgements.translations, desc=self.label)]

        distances = [base_t.distance(translated_t) for base_t, translated_t in zip(base_transitions, translated_model)]

        return distances

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SyntacticCompositionality):
            return NotImplemented
        return all([
            self.reference_free == other.reference_free,
            self.src_lang == other.src_lang,
            self.tgt_lang == other.tgt_lang,
        ])

    def __hash__(self) -> int:
        return hash((self.reference_free, self.src_lang, self.tgt_lang))
