from typing import List

from prism.prism import Prism
from common import ReferenceFreeMetric, Judgements


class PrismMetric(ReferenceFreeMetric):
    # https://github.com/thompsonb/prism
    # https://aclanthology.org/2020.emnlp-main.8.pdf

    label = "Prism"

    def __init__(self, tgt_lang: str, reference_free=False, model_dir="prism/model_dir/m39v1"):
        try:
            self.model = Prism(model_dir, lang=tgt_lang)
        except OSError:
            from io import BytesIO
            from zipfile import ZipFile
            from urllib.request import urlopen
            # or: requests.get(url).content

            resp = urlopen("http://www.test.com/file.zip")
            zipfile = ZipFile(BytesIO(resp.read()))
            for line in zipfile.open(file).readlines():
                print(line.decode('utf-8'))

            print("Prism missing model: run:\n"
                  "mkdir -p prism/model_dir\n"
                  "cd prism/model_dir\n"
                  "wget http://data.statmt.org/prism/m39v1.tar\n"
                  "tar xf m39v1.tar\n"
                  "and pass model_dir=%s" % model_dir)
            raise

        self.reference_free = reference_free

    def compute(self, judgements: Judgements) -> List[float]:
        if self.reference_free:
            return self.model.score(cand=judgements.translations, src=judgements.src_texts, segment_scores=True)
        else:
            return self.model.score(cand=judgements.translations, ref=[rs[0] for rs in judgements.references],
                                    segment_scores=True)
