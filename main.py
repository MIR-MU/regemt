import os
from itertools import product
import logging
from typing import Tuple, Set, Optional, List
import sys

import pandas as pd
import seaborn as sns
import transformers
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from bertscore import BERTScore
from common import Evaluator, Report
from conventional_metrics import BLEU, METEOR
from ood_metrics import SyntacticCompositionality
from scm import SCM, ContextualSCM, DecontextualizedSCM
from wmd import WMD, ContextualWMD, DecontextualizedWMD
from ensemble import Regression
import warnings

LOGGER = logging.getLogger(__name__)

evaluator = None


def main(firstn: Optional[float] = None,
         reference_frees: Tuple[bool, ...] = (True, False),
         judgements_types: Tuple[str, ...] = ('challengeset', 'florestest2021', 'newstest2021', 'tedtalks'),
         src_langs: Optional[Set[str]] = None,
         tgt_langs: Optional[Set[str]] = None,
         enable_compositionality: bool = True,
         enable_sota_metrics: bool = True,
         enable_fasttext_metrics: bool = True,
         enable_contextual_scm: bool = False,
         human: bool = False):
    global evaluator
    for reference_free in reference_frees:
        print("Evaluating %sreference-free metrics" % ('' if reference_free else 'non-'))
        for judgements_type in judgements_types:
            if judgements_type == 'catastrophic' and not reference_free:
                continue

            if judgements_type in Evaluator.submission_judgement_types:
                print("Generating WMT21 submission")

            langs = Evaluator.langs_for_judgements(judgements_type)

            for lang_pair in langs:
                src_lang, tgt_lang = lang_pair.split("-")
                if src_langs is not None and src_lang not in src_langs:
                    continue
                if tgt_langs is not None and tgt_lang not in tgt_langs:
                    continue

                print("Evaluating lang pair %s" % lang_pair)

                metrics = []

                def make_metric(cls, *args, **kwargs):
                    if not cls.supports(tgt_lang):
                        LOGGER.warning(f'{cls} does not support tgt_lang={tgt_lang}')
                        return None
                    if reference_free and not cls.supports(src_lang):
                        LOGGER.warning(f'{cls} does not support src_lang={src_lang}')
                        return None
                    metric = cls(*args, **kwargs)
                    return metric

                if enable_sota_metrics:
                    from prism_metric import PrismMetric
                    from comet_metric import Comet
                    metrics += [
                        make_metric(Comet),
                        make_metric(PrismMetric, tgt_lang=tgt_lang, reference_free=reference_free),
                    ]
                    pass

                metrics += [
                    make_metric(BERTScore, tgt_lang=tgt_lang, reference_free=reference_free),
                    make_metric(ContextualWMD, tgt_lang=tgt_lang, reference_free=reference_free),
                ]

                if enable_contextual_scm:
                    metrics += [make_metric(ContextualSCM, tgt_lang=tgt_lang, reference_free=reference_free)]

                metrics += [
                    make_metric(DecontextualizedWMD, tgt_lang=tgt_lang, use_tfidf=False, reference_free=reference_free),
                    make_metric(DecontextualizedWMD, tgt_lang=tgt_lang, use_tfidf=True, reference_free=reference_free),
                    make_metric(DecontextualizedSCM, tgt_lang=tgt_lang, use_tfidf=False, reference_free=reference_free),
                    make_metric(DecontextualizedSCM, tgt_lang=tgt_lang, use_tfidf=True, reference_free=reference_free),
                ]

                if enable_fasttext_metrics:
                    metrics += [
                        make_metric(SCM, tgt_lang=tgt_lang, use_tfidf=False),
                        make_metric(SCM, tgt_lang=tgt_lang, use_tfidf=True),
                        make_metric(WMD, tgt_lang=tgt_lang, use_tfidf=False),
                        make_metric(WMD, tgt_lang=tgt_lang, use_tfidf=True),
                    ]

                if enable_sota_metrics:
                    from bleurt_metric import BLEUrt
                    metrics += [
                        make_metric(BLEUrt)
                    ]

                metrics += [
                    make_metric(BLEU),
                    make_metric(METEOR),
                ]

                if enable_compositionality:
                    metrics += [
                        make_metric(SyntacticCompositionality, src_lang=src_lang, tgt_lang=tgt_lang,
                                    reference_free=reference_free)
                    ]

                regression = make_metric(Regression, metrics, reference_free=reference_free)
                regression_baseline = make_metric(Regression, None, reference_free=reference_free)
                metrics = [regression] + metrics + [regression_baseline]

                metrics = list(filter(lambda metric: metric is not None, metrics))

                evaluator = Evaluator("data_dir", lang_pair, metrics,
                                      judgements_type=judgements_type,
                                      human=human,
                                      reference_free=reference_free, firstn=firstn)
                if judgements_type in evaluator.submission_judgement_types:
                    print("submit_and_report")
                    evaluator.submit_and_report(submitted_metrics_labels=["Regression", "Regression_baseline", "WMD"])
                else:
                    evaluator.evaluate()


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    transformers.logging.set_verbosity_error()
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        pass
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    parameters = dict()
    for arg in sys.argv[1:]:
        if arg == '--fast':
            parameters = {
                **parameters,
                **{
                    'firstn': 100,
                    'judgements_types': ('MQM', ),
                    'src_langs': {'en'},
                    'enable_compositionality': False,
                    'enable_sota_metrics': False,
                    'enable_fasttext_metrics': False,
                }
            }
        else:
            raise ValueError(f'Unrecognized command-line argument: {arg}')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
    main(**parameters)
