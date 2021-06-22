import abc
import os
from typing import List, Tuple, Iterable, Dict
import pandas as pd


TRAIN_DATASET_FILE_TEMPLATE = "DAseg-wmt-newstest2015/DAseg.newstest2015.%s.%s"
TEST_DATASET_FILE_TEMPLATE = "DAseg-wmt-newstest2016/DAseg.newstest2016.%s.%s"

LANGS = ["cs-en", "de-en", "fi-en", "ru-en"]


class Judgements:

    def __init__(self, src_texts: List[str], references: List[List[str]], translations: List[str], scores: List[float]):
        self.src_texts = src_texts
        self.references = references
        self.translations = translations
        self.scores = scores

    def __len__(self):
        return len(self.src_texts)


class Metric(abc.ABC):
    label: str = None

    @abc.abstractmethod
    def fit(self, judgements: Judgements):
        pass

    @abc.abstractmethod
    def compute(self, judgements: Judgements) -> List[float]:
        pass


class Evaluator:

    def __init__(self, data_dir: str, lang_pair: Tuple[str, str], metrics: List[Metric]):
        self.lang_pair = lang_pair
        self.data_dir = data_dir
        self.metrics = metrics

        train_judgements = self.load_judgements("train")
        for metric in self.metrics:
            metric.fit(train_judgements)

    def load_judgements(self, split: str = "train") -> Judgements:
        split_file_template = os.path.join(self.data_dir, TRAIN_DATASET_FILE_TEMPLATE if split == "train"
                                                          else TEST_DATASET_FILE_TEMPLATE)
        src_texts = self._load_file(split_file_template % ("source", self.lang_pair))
        references = [[ref] for ref in self._load_file(split_file_template % ("reference", self.lang_pair))]
        translations = self._load_file(split_file_template % ("mt-system", self.lang_pair))
        scores = self._load_file(split_file_template % ("human", self.lang_pair))

        return Judgements(src_texts, references, translations, scores)

    def _load_file(self, fpath: str) -> List[str]:
        with open(fpath) as f:
            return [l.strip() for l in f.readlines()]

    def evaluate(self) -> Dict[str, List[float]]:
        report = {}
        test_judgements = self.load_judgements("test")
        report["human"] = test_judgements.scores

        for metric in self.metrics:
            report[metric.label] = metric.compute(test_judgements)

        return report

    def evaluate_no_references(self):
        # TODO
        pass
