from typing import List, Any
from pathlib import Path
import zipfile
from urllib.request import urlretrieve
from functools import lru_cache

from .common import Judgements, Metric
from bleurt import score


class BLEUrt(Metric):
    # https://unbabel.github.io/COMET/html/running.html
    # https://aclanthology.org/2020.emnlp-main.213.pdf#page=5

    label = "BLEUrt"

    def __init__(self, model_dir="bleurt/model_dir"):
        model_path = Path(model_dir)

        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)
            zipfile_path = model_path / 'bleurt-base-128.zip'
            print(f'Downloading BLEUrt model to {zipfile_path}')
            url = 'https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip'
            urlretrieve(url, zipfile_path)
            print(f'Extracting BLEUrt model to {model_path}')
            with zipfile.ZipFile(zipfile_path) as zf:
                zf.extractall(path=model_path)
            zipfile_path.unlink()

        self.scorer = score.BleurtScorer(f'{model_dir}/bleurt-base-128')
        self.model_dir = model_dir

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        scores = self.scorer.score(references=[rs[0] for rs in judgements.references],
                                   candidates=judgements.translations)
        return scores

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BLEUrt):
            return NotImplemented
        return all([
            self.model_dir == other.model_dir,
        ])

    def __hash__(self) -> int:
        return hash(self.model_dir)
