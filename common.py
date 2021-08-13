import abc
import os
from typing import List, Tuple, Iterable, Dict, Optional, Any, Union
from statistics import mean
from itertools import repeat, chain
import logging

import pandas as pd
from gensim.utils import simple_preprocess
from tqdm.autonotebook import tqdm
import sklearn.utils

LOGGER = logging.getLogger(__name__)

TRAIN_DATASET_FILE_TEMPLATE = "DAseg-wmt-newstest2015/DAseg.newstest2015.%s.%s"
TEST_DATASET_FILE_TEMPLATE = "DAseg-wmt-newstest2016/DAseg.newstest2016.%s.%s"

Report = Dict[str, List[float]]


class Judgements:

    def __init__(self, src_texts: List[str], references: Optional[List[List[str]]],
                 translations: List[str], scores: List[float], shuffle: bool = True,
                 shuffle_random_state: int = 42, make_unique: bool = True):
        assert references is None or len(references) == len(src_texts)
        assert len(translations) == len(src_texts)
        assert len(scores) == len(src_texts)

        if make_unique:
            new_src_texts, new_references, new_translations, new_scores = [], [] if references else None, [], dict()
            for row in zip(src_texts, map(tuple, references) if references else repeat(None), translations, scores):
                src_text, reference, translation, score = row
                if (src_text, translation) not in new_scores:
                    new_scores[(src_text, translation)] = []
                    new_src_texts.append(src_text)
                    if new_references is not None:
                        new_references.append(reference)
                    new_translations.append(translation)
                new_scores[(src_text, translation)].append(score)
            if len(new_src_texts) < len(src_texts):
                num_non_uniques = len(src_texts) - len(new_src_texts)
                msg = f'Averaged {num_non_uniques} non-unique judgements: {len(src_texts)} -> {len(new_src_texts)}'
                LOGGER.warning(msg)
            src_texts, references, translations = new_src_texts, new_references, new_translations
            scores = [mean(new_scores[(src_text, translation)])
                      for src_text, translation in zip(src_texts, translations)]

        if shuffle:
            src_texts, translations, scores = sklearn.utils.shuffle(
                src_texts, translations, scores, random_state=shuffle_random_state)
            if references is not None:
                references = sklearn.utils.shuffle(references, random_state=shuffle_random_state)

        self.src_texts = tuple(src_texts)
        self.references = tuple(map(tuple, references)) if references is not None else None
        self.translations = tuple(translations)
        self.scores = tuple(scores)

    def get_tokenized_texts(self, desc: Optional[str] = None) -> Iterable[Tuple[List[str], List[str]]]:
        sources = [t[0] for t in self.references] if self.references is not None else self.src_texts
        corpus = zip(sources, self.translations)
        if desc:
            corpus = tqdm(corpus, desc=desc, total=len(self))
        for source, translation in corpus:
            source_words = list(map(str.lower, simple_preprocess(source)))
            translation_words = list(map(str.lower, simple_preprocess(translation)))
            yield source_words, translation_words

    def __getitem__(self, indexes: slice) -> 'Judgements':
        src_texts = list(self.src_texts[indexes])
        references = list(self.references[indexes]) if self.references is not None else None
        translations = list(self.translations[indexes])
        scores = list(self.scores[indexes])
        return Judgements(src_texts, references, translations, scores, shuffle=False, make_unique=False)

    def split(self, *other_lists: List, split_ratio: float = 0.8) -> Tuple[Tuple['Judgements', List[List]],
                                                                           Tuple['Judgements', List[List]]]:
        for other_list in other_lists:
            assert len(other_list) == len(self)

        unique_src_texts = sorted(set(self.src_texts))
        pivot = int(round(len(unique_src_texts) * split_ratio))
        train_unique_src_texts = set(unique_src_texts[:pivot])
        test_unique_src_texts = set(unique_src_texts[pivot:])

        train_src_texts, train_references, train_translations, train_scores, train_other_lists = \
            [], [] if self.references else None, [], [], [[] for other_list in other_lists]
        test_src_texts, test_references, test_translations, test_scores, test_other_lists = \
            [], [] if self.references else None, [], [], [[] for other_list in other_lists]
        for row in zip(self.src_texts, self.references or repeat(None), self.translations, self.scores, *other_lists):
            src_text, reference, translation, score, *other_elements = row
            assert src_text in train_unique_src_texts | test_unique_src_texts
            if src_text in train_unique_src_texts:
                train_src_texts.append(src_text)
                if train_references is not None:
                    train_references.append(list(reference))
                train_translations.append(translation)
                train_scores.append(score)
                for train_other_list, other_element in zip(train_other_lists, other_elements):
                    train_other_list.append(other_element)
            else:
                test_src_texts.append(src_text)
                if test_references is not None:
                    test_references.append(list(reference))
                test_translations.append(translation)
                test_scores.append(score)
                for test_other_list, other_element in zip(test_other_lists, other_elements):
                    test_other_list.append(other_element)

        train_judgements = Judgements(train_src_texts, train_references, train_translations, train_scores,
                                      shuffle=False, make_unique=False)
        test_judgements = Judgements(test_src_texts, test_references, test_translations, test_scores,
                                     shuffle=False, make_unique=False)
        assert len(train_judgements) + len(test_judgements) == len(self)
        assert not train_judgements.overlaps(test_judgements)

        return (train_judgements, train_other_lists), (test_judgements, test_other_lists)

    def overlaps(self, other: 'Judgements') -> bool:
        if self == other:
            return True
        self_corpus = set(self.src_texts)
        other_corpus = set(other.src_texts)
        return len(self_corpus & other_corpus) > 0

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Judgements):
            return NotImplemented
        return all([
            self.src_texts == other.src_texts,
            self.references == other.references,
            self.translations == other.translations,
            self.scores == other.scores,
        ])

    def __hash__(self) -> int:
        return hash((self.src_texts, self.references, self.translations, self.scores))

    def __len__(self):
        return len(self.src_texts)


class Metric(abc.ABC):
    label: str = 'None'

    @staticmethod
    def supports(src_lang: str, tgt_lang: str, reference_free: bool) -> bool:
        return True

    def fit(self, train_judgements: Judgements) -> None:
        pass

    @abc.abstractmethod
    def compute(self, test_judgements: Judgements) -> List[float]:
        pass

    def __repr__(self) -> str:
        return self.label


class ReferenceFreeMetric(Metric):

    def compute_ref_free(self, test_judgements: Judgements) -> List[float]:
        return self.compute(test_judgements)


class AugmentedCorpus:
    def __init__(self, prefix: str, corpus: Iterable[List[str]]):
        if ' ' in prefix:
            raise ValueError(f'Prefix {prefix} contains spaces')
        self.prefix = prefix
        self.corpus: List[List[str]] = self._augment_corpus(corpus)

    def get_matching_tokens(self, augmented_tokens: Iterable[str],
                            searched_token: str) -> Iterable[str]:
        for augmented_token in augmented_tokens:
            if self.unaugment_token(augmented_token) == searched_token:
                yield augmented_token

    def unaugment_token(self, augmented_token: str) -> str:
        prefix, text_index, token_index, token = augmented_token.split(' ', maxsplit=3)
        return token

    def _augment_corpus(self, corpus: Iterable[List[str]]) -> List[List[str]]:
        augmented_corpus = [
            [
                f'{self.prefix} {text_index} {token_index} {token}'
                for token_index, token in enumerate(text)
            ]
            for text_index, text in enumerate(corpus)
        ]
        return augmented_corpus


class Evaluator:

    def __init__(self, data_dir: str,
                 test_lang_pair: str,
                 metrics: List[Union[Metric, ReferenceFreeMetric]],
                 judgements_type: str,
                 firstn: Optional[int] = 100,
                 reference_free: bool = False,
                 test_lang_selection: str = "leave-one-out"):
        self.data_dir = data_dir
        self.metrics = metrics
        self.judgements_type = judgements_type
        self.firstn = firstn
        self.reference_free = reference_free

        all_langs = self.langs_for_judgements(judgements_type)

        if test_lang_selection == "leave-one-out" and test_lang_pair in all_langs:
            train_langs = [lang for lang in all_langs if lang != test_lang_pair]
            self.train_judgements = self.load_judgements(train_langs, "train")
            self.test_judgements = self.load_judgements([test_lang_pair], "test")
        else:
            raise ValueError("%s: %s: %s" % (judgements_type, test_lang_selection, test_lang_pair))

        for metric in self.metrics:
            metric.fit(self.train_judgements)

    @staticmethod
    def langs_for_judgements(judgements_type: str):
        if judgements_type == "DA":
            return ["cs-en", "de-en", "fi-en", "ru-en"]
        elif judgements_type == "PSQM" or judgements_type == "MQM":
            return ["zh-en", "en-de"]
        elif judgements_type == "catastrophic":
            return ["en-cs", "en-de", "en-ja", "en-zh"]
        else:
            raise ValueError(judgements_type)

    def load_judgements(self, lang_pairs: List[str], split: str = "train",
                        error_type: Optional[str] = None, first_reference_only: bool = True) -> Judgements:

        judgements_all = []

        for lang_pair in lang_pairs:
            if self.judgements_type == "DA":
                split_file_template = os.path.join(self.data_dir, TEST_DATASET_FILE_TEMPLATE)
                src_texts = self._load_file(split_file_template % ("source", lang_pair))
                references = [[ref] for ref in self._load_file(split_file_template % ("reference", lang_pair))]
                translations = self._load_file(split_file_template % ("mt-system", lang_pair))
                scores = [float(s) for s in self._load_file(split_file_template % ("human", lang_pair))]

            elif self.judgements_type == "PSQM":
                split_file_template = os.path.join(self.data_dir, "psqm_newstest2020_zhen.tsv")
                all_df = pd.read_csv(split_file_template, sep="\t")
                all_df = all_df.set_index(["doc_id", "seg_id"])
                all_df = all_df.sort_index()

                src_texts = []
                references = []
                translations = []
                scores = []

                for i in tqdm(all_df.index, total=len(all_df)):
                    segment_df = all_df.loc[i]

                    ref_judgements = segment_df[segment_df.system.apply(lambda label: "Human" in label)]
                    if len(ref_judgements):
                        max_scored_ref = ref_judgements[ref_judgements.score == ref_judgements.score.max()].iloc[0]
                        max_scored_rater = max_scored_ref.rater
                        same_rater_judgements = segment_df[segment_df.rater == max_scored_rater]
                        for idx, judgement in same_rater_judgements.iterrows():
                            src_texts.append(judgement.source)
                            references.append([max_scored_ref.target])
                            translations.append(judgement.target)
                            scores.append(judgement.score)
                            break  # we want variable translation pairs, but if more is needed, we can remove this
                    else:
                        print("No reference judgements: %s" % i)

            elif self.judgements_type == "MQM":
                df = pd.read_csv(os.path.join(self.data_dir, "mqm_newstest2020_%s.tsv" % lang_pair.replace("-", "")),
                                 sep="\t")
                df = df.set_index(["system", "seg_id"])

                judgements_df = pd.read_csv(
                    os.path.join(self.data_dir, "mqm_newstest2020_%s.avg_seg_scores.tsv" % lang_pair.replace("-", "")),
                    sep=" ")
                judgements_df = judgements_df.set_index(["system", "seg_id"])

                df = df.join(judgements_df)
                df = df.reset_index().set_index(["seg_id", "doc_id", "rater"])
                system_df = df[~df["system"].isin(["Human-A.0", "Human-B.0"])]

                human_df = df[df["system"].isin(["Human-A.0", "Human-B.0"])]
                human_df_ok = human_df[human_df.category == "no-error"]
                human_df_translated = human_df_ok.join(system_df, lsuffix="_human", rsuffix="_system")

                all_references = human_df_translated["target_human"].groupby(level=[0, 1, 2]).unique()
                all_references.name = 'all_references'
                ref_translation_df = human_df_translated.join(all_references)

                if first_reference_only:
                    ref_translation_df['all_references'] = ref_translation_df['all_references'].apply(lambda x: [x[0]])

                translated_clean_df = ref_translation_df[~pd.isna(ref_translation_df).any(axis=1)]

                if error_type is None:
                    selected_df = translated_clean_df
                else:
                    selected_df = translated_clean_df[translated_clean_df["category_system"] == error_type]

                src_texts = selected_df["source_system"].tolist()
                references = selected_df["all_references"].tolist()
                translations = selected_df["target_system"].tolist()
                scores = selected_df["mqm_avg_score_system"].tolist()

            elif self.judgements_type == "catastrophic":
                df = pd.read_csv("data_dir/%s_majority_dev.tsv" % lang_pair.replace("-", ""),
                                 sep="\t", names=["source", "translation", "judgements", "is_critical"])
                df.judgements = df.judgements.apply(lambda j:
                                                    sum(map(int, j.replace("[", "").replace("]", "").split(", "))))
                src_texts = df["source"].tolist()
                references = None
                translations = df["translation"].tolist()
                scores = df["judgements"].tolist()

            else:
                raise ValueError(self.judgements_type)

            judgements = Judgements(src_texts, references, translations, scores)

            if split == "train":
                (judgements, []), _ = judgements.split()
            elif split == "test":
                _, (judgements, []) = judgements.split()
            else:
                raise ValueError(split)
            judgements_all.append(judgements)

        # unify judgements for given split over all languages
        judgements = Judgements(*(list(chain(*[j.src_texts for j in judgements_all])),
                                  list(chain(*[j.references for j in judgements_all]))
                                  if not self.reference_free else None,
                                  list(chain(*[j.translations for j in judgements_all])),
                                  list(chain(*[j.scores for j in judgements_all]))))

        if self.firstn is not None:
            if self.firstn > len(judgements):
                message = 'Requested firstn={} judgements, but only {} exist in {}-{}'
                message = message.format(self.firstn, len(judgements), self.judgements_type, split)
                LOGGER.warning(message)
            else:
                judgements = judgements[:self.firstn]

        return judgements

    @staticmethod
    def _load_file(fpath: str) -> List[str]:
        with open(fpath) as f:
            return [line.strip() for line in f.readlines()]

    def evaluate(self) -> Report:
        report = {}
        report["human"] = list(self.test_judgements.scores)
        if not self.reference_free:
            for metric in self.metrics:
                report[metric.label] = [float(val) for val in metric.compute(self.test_judgements)]
        else:
            for metric in [m for m in self.metrics if isinstance(m, ReferenceFreeMetric)]:
                report[metric.label] = [float(val) for val in metric.compute_ref_free(self.test_judgements)]

        return report
