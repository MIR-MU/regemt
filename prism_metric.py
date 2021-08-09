from typing import List, Any
from pathlib import Path
import tarfile
from urllib.request import urlretrieve
from functools import lru_cache

from prism.prism import Prism
from common import ReferenceFreeMetric, Judgements


class PrismMetric(ReferenceFreeMetric):
    # https://github.com/thompsonb/prism
    # https://aclanthology.org/2020.emnlp-main.8.pdf

    label = "Prism"

    def __init__(self, tgt_lang: str, src_lang: str, reference_free=False,
                 model_dir="prism/model_dir", device="cuda:1"):
        assert self.__class__.supports(tgt_lang)
        if reference_free:
            assert self.__class__.supports(src_lang)

        model_path = Path(model_dir)

        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)
            tarfile_path = model_path / 'm39v1.tar'
            print(f'Downloading Prism model to {tarfile_path}')
            url = 'http://data.statmt.org/prism/m39v1.tar'
            urlretrieve(url, tarfile_path)
            print(f'Extracting Prism model to {model_path}')
            with tarfile.open(tarfile_path, 'r') as tar:
                tar.extractall(path=model_path)
            tarfile_path.unlink()

        print(f'{self}: Initializing {model_dir}/m39v1')
        self.model = Prism(f'{model_dir}/m39v1', lang=tgt_lang, device=device)
        self.model_dir = model_dir
        self.reference_free = reference_free

    @lru_cache(maxsize=None)
    def compute(self, judgements: Judgements) -> List[float]:
        if self.reference_free:
            return self.model.score(cand=judgements.translations, src=judgements.src_texts, segment_scores=True)
        else:
            return self.model.score(cand=judgements.translations, ref=[rs[0] for rs in judgements.references],
                                    segment_scores=True)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PrismMetric):
            return NotImplemented
        return all([
            self.reference_free == other.reference_free,
            self.model_dir == other.model_dir,
        ])

    def __hash__(self) -> int:
        return hash((self.reference_free, self.model_dir))
