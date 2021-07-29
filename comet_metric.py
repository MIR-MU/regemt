from typing import List, Any
from functools import lru_cache
from contextlib import redirect_stdout, redirect_stderr
import os

from comet.models import download_model

from common import Judgements, Metric


class Comet(Metric):
    # https://unbabel.github.io/COMET/html/running.html
    # https://aclanthology.org/2020.emnlp-main.213.pdf#page=5

    label = "Comet"

    def __init__(self, model_name: str = 'wmt-large-da-estimator-1719'):
        self.model_name = model_name
        print(f'{self}: Initializing {model_name}')
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            self.model = download_model(model_name, "comet_model/")

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        data = [
            {
                "src": src_text,
                "mt": translation,
                "ref": rs[0]
            }
            for src_text, translation, rs
            in zip(judgements.src_texts, judgements.translations, judgements.references)
        ]
        return self.model.predict(data, cuda=True, show_progress=True)[1]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Comet):
            return NotImplemented
        return all([
            self.model_name == other.model_name,
        ])

    def __hash__(self) -> int:
        return hash(self.model_name)
