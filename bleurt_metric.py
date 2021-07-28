from typing import List

from common import Judgements, Metric


class BLEUrt(Metric):
    # https://unbabel.github.io/COMET/html/running.html
    # https://aclanthology.org/2020.emnlp-main.213.pdf#page=5

    label = "BLEUrt"

    def __init__(self, model_dir="bleurt/model_dir/bleurt-base-128"):
        try:
            from bleurt import score
            self.scorer = score.BleurtScorer(model_dir)
        except OSError:
            print("Bleurt missing model: run:\n"
                  "mkdir -p bleurt/model_dir\n"
                  "cd bleurt/model_dir\n"
                  "wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip .\n"
                  "unzip bleurt-base-128.zip\n"
                  "and pass model_dir=%s" % model_dir)
            raise

    def compute(self, judgements: Judgements) -> List[float]:
        scores = self.scorer.score(references=[rs[0] for rs in judgements.references],
                                   candidates=judgements.translations)
        return scores
