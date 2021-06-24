import abc
import os
from typing import List, Tuple, Iterable, Dict
import pandas as pd
from tqdm import tqdm

TRAIN_DATASET_FILE_TEMPLATE = "DAseg-wmt-newstest2015/DAseg.newstest2015.%s.%s"
TEST_DATASET_FILE_TEMPLATE = "DAseg-wmt-newstest2016/DAseg.newstest2016.%s.%s"


class Judgements:

    def __init__(self, src_texts: List[str], references: List[List[str]], translations: List[str], scores: List[float]):
        self.src_texts = src_texts
        self.references = references
        self.translations = translations
        self.scores = scores

    def __eq__(self, other: Judgements) -> bool:
        if not isinstance(other, Judgements):
            raise NotImplemented
        return all(
            self.src_texts == other.src_texts,
            self.references == other.references,
            self.translations == other.translations,
            self.scores == other.scores,
        )

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


MQM_RATING_MAPPING = {}


class Evaluator:

    langs = ["cs-en", "de-en", "fi-en", "ru-en"]
    langs_psqm = ["zh-en"]

    def __init__(self, data_dir: str, lang_pair: Tuple[str, str], metrics: List[Metric], psqm: bool = False):
        self.lang_pair = lang_pair
        self.data_dir = data_dir
        self.metrics = metrics
        self.psqm = psqm

        train_judgements = self.load_judgements("train")
        test_judgements = self.load_judgements("test")
        for metric in self.metrics:
            metric.fit(train_judgements, test_judgements)

    def load_judgements(self, split: str = "train", firstn: int = 10000) -> Judgements:
        if self.psqm:
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
                if len(src_texts) >= firstn:
                    break
        else:
            split_file_template = os.path.join(self.data_dir, TRAIN_DATASET_FILE_TEMPLATE if split == "train"
                                                              else TEST_DATASET_FILE_TEMPLATE)
            src_texts = self._load_file(split_file_template % ("source", self.lang_pair))
            references = [[ref] for ref in self._load_file(split_file_template % ("reference", self.lang_pair))]
            translations = self._load_file(split_file_template % ("mt-system", self.lang_pair))
            scores = [float(s) for s in self._load_file(split_file_template % ("human", self.lang_pair))]

        return Judgements(src_texts, references, translations, scores)

    def _load_file(self, fpath: str) -> List[str]:
        with open(fpath) as f:
            return [l.strip() for l in f.readlines()]

    def evaluate(self) -> Dict[str, List[float]]:
        report = {}
        test_judgements = self.load_judgements("test")
        report["human"] = test_judgements.scores

        for metric in self.metrics:
            report[metric.label] = [float(val) for val in metric.compute(test_judgements)]

        return report

    def evaluate_no_references(self):
        # TODO
        pass
