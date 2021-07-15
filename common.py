import abc
import os
from typing import List, Tuple, Iterable, Dict, Optional, Set, Any, Union
import pandas as pd
from gensim.utils import simple_preprocess
from tqdm.autonotebook import tqdm
import sklearn.utils

TRAIN_DATASET_FILE_TEMPLATE = "DAseg-wmt-newstest2015/DAseg.newstest2015.%s.%s"
TEST_DATASET_FILE_TEMPLATE = "DAseg-wmt-newstest2016/DAseg.newstest2016.%s.%s"


class Judgements:

    def __init__(self, src_texts: List[str], references: Optional[List[List[str]]],
                 translations: List[str], scores: List[float], shuffle: bool = True,
                 shuffle_random_state: int = 42):
        assert references is None or len(references) == len(src_texts)
        assert len(translations) == len(src_texts)
        assert len(scores) == len(src_texts)

        if shuffle:
            src_texts, translations, scores = sklearn.utils.shuffle(
                src_texts, translations, scores, random_state=shuffle_random_state)
            if references is not None:
                references = sklearn.utils.shuffle(references, random_state=shuffle_random_state)

        self.src_texts = src_texts
        self.references = references
        self.translations = translations
        self.scores = scores

    def get_tokenized_texts(self, stopwords: Optional[Set] = None,
                            desc: Optional[str] = None) -> Iterable[Tuple[List[str], List[str]]]:
        if not stopwords:
            stopwords = set()
        corpus = zip(self.references, self.translations)
        if desc:
            corpus = tqdm(corpus, desc=desc, total=len(self))
        for reference, translation in corpus:
            reference_words = [w.lower() for w in simple_preprocess(reference[0])
                               if w.lower() not in stopwords]
            translation_words = [w.lower() for w in simple_preprocess(translation)
                                 if w.lower() not in stopwords]
            yield reference_words, translation_words

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Judgements):
            return NotImplemented
        return all([
            self.src_texts == other.src_texts,
            self.references == other.references,
            self.translations == other.translations,
            self.scores == other.scores,
        ])

    def __getitem__(self, indexes: slice) -> 'Judgements':
        src_texts = self.src_texts[indexes]
        references = self.references[indexes] if self.references is not None else None
        translations = self.translations[indexes]
        scores = self.scores[indexes]
        return Judgements(src_texts, references, translations, scores, shuffle=False)

    def __len__(self):
        return len(self.src_texts)


class Metric(abc.ABC):
    label: str = 'None'

    def fit(self, train_judgements: Judgements):
        pass

    @abc.abstractmethod
    def compute(self, test_judgements: Judgements) -> List[float]:
        pass


class ReferenceFreeMetric(Metric):

    @abc.abstractmethod
    def compute_ref_free(self, test_judgements: Judgements) -> List[float]:
        pass


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

    def __init__(self, data_dir: str, lang_pair: str, metrics: List[Union[Metric, ReferenceFreeMetric]],
                 judgements_type: str, firstn: Optional[int] = 100, reference_free: bool = False):
        self.lang_pair = lang_pair
        self.data_dir = data_dir
        self.metrics = metrics
        self.judgements_type = judgements_type
        self.firstn = firstn
        self.reference_free = reference_free

        train_judgements = self.load_judgements("train")
        for metric in self.metrics:
            metric.fit(train_judgements)

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

    def load_judgements(self, split: str = "train", error_type: str = None, first_reference_only: bool = True,
                        split_ratio: float = 0.8) -> Judgements:
        if self.judgements_type == "DA":
            split_file_template = os.path.join(self.data_dir, TEST_DATASET_FILE_TEMPLATE)
            src_texts = self._load_file(split_file_template % ("source", self.lang_pair))
            references = [[ref] for ref in self._load_file(split_file_template % ("reference", self.lang_pair))]
            translations = self._load_file(split_file_template % ("mt-system", self.lang_pair))
            scores = [float(s) for s in self._load_file(split_file_template % ("human", self.lang_pair))]

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
                    max_scored_reference = ref_judgements[ref_judgements.score == ref_judgements.score.max()].iloc[0]
                    max_scored_rater = max_scored_reference.rater
                    same_rater_judgements = segment_df[segment_df.rater == max_scored_rater]
                    for idx, judgement in same_rater_judgements.iterrows():
                        src_texts.append(judgement.source)
                        references.append([max_scored_reference.target])
                        translations.append(judgement.target)
                        scores.append(judgement.score)
                        break  # we want variable translation pairs, but if more is needed, we can remove this
                else:
                    print("No reference judgements: %s" % i)

        elif self.judgements_type == "MQM":
            df = pd.read_csv(os.path.join(self.data_dir, "mqm_newstest2020_%s.tsv" % self.lang_pair.replace("-", "")),
                             sep="\t")
            df = df.set_index(["system", "seg_id"])

            judgements_df = pd.read_csv(
                os.path.join(self.data_dir, "mqm_newstest2020_%s.avg_seg_scores.tsv" % self.lang_pair.replace("-", "")),
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
            df = pd.read_csv("data_dir/%s_majority_dev.tsv" % self.lang_pair.replace("-", ""),
                             sep="\t", names=["source", "translation", "judgements", "is_critical"])
            df.judgements = df.judgements.apply(lambda j:
                                                sum(map(int, j.replace("[", "").replace("]", "").split(", "))))
            src_texts = df["source"].tolist()
            references = None
            translations = df["translation"].tolist()
            scores = df["judgements"].tolist()

        else:
            raise ValueError(self.judgements_type)

        if self.reference_free:
            references = None

        judgements = Judgements(src_texts, references, translations, scores)

        pivot = int(round(len(judgements) * split_ratio))
        if split == "train":
            judgements = judgements[:pivot]
        elif split == "test":
            judgements = judgements[pivot:]
        else:
            raise ValueError(split)

        if self.firstn is not None:
            judgements = judgements[:self.firstn]

        return judgements

    @staticmethod
    def _load_file(fpath: str) -> List[str]:
        with open(fpath) as f:
            return [line.strip() for line in f.readlines()]

    def evaluate(self) -> Dict[str, List[float]]:
        report = {}
        test_judgements = self.load_judgements("test")
        report["human"] = test_judgements.scores
        if not self.reference_free:
            for metric in self.metrics:
                report[metric.label] = [float(val) for val in metric.compute(test_judgements)]
        else:
            for metric in [m for m in self.metrics if issubclass(type(m), ReferenceFreeMetric)]:
                report[metric.label] = [float(val) for val in metric.compute_ref_free(test_judgements)]

        return report
