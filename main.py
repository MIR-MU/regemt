import os
import logging
from typing import Tuple, Set, Optional
import sys
import warnings

import transformers
from bertscore import BERTScore
from common import Evaluator
from conventional_metrics import BLEU, METEOR
from ood_metrics import SyntacticCompositionality
from scm import SCM, ContextualSCM, DecontextualizedSCM
from wmd import WMD, ContextualWMD, DecontextualizedWMD
from ensemble import Regression

LOGGER = logging.getLogger(__name__)


# Notes:
#  - Running with firstn=100 is a good idea even when producing a submission.
#    It will allow us to catch any issues in a fraction of the time to do the full
#    run. This is good, because some issues may render the entire submission useless.
def main(firstn: Optional[float] = None,
         reference_frees: Tuple[bool, ...] = (True, False),
         judgements_types: Tuple[str, ...] = ('challengeset', 'florestest2021', 'tedtalks', 'newstest2021'),
         human: bool = False,
         src_langs: Optional[Set[str]] = None,
         tgt_langs: Optional[Set[str]] = None,
         enable_compositionality: bool = True,
         enable_sota_metrics: bool = True,
         enable_fasttext_metrics: bool = True,
         enable_contextual_scm: bool = False):
    for reference_free in reference_frees:

        if human and reference_free:
            continue

        print("Evaluating %sreference-free metrics" % ('' if reference_free else 'non-'))
        for judgements_type in judgements_types:
            if judgements_type == 'catastrophic' and not reference_free:
                continue

            print("Evaluating %s judgements" % judgements_type)
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
                    if not cls.supports(src_lang, tgt_lang, reference_free):
                        message = '%s does not support src_lang=%s, tgt_lang=%s, reference_free=%s'
                        message = message % (cls, src_lang, tgt_lang, reference_free)
                        LOGGER.warning(message)
                        return None
                    metric = cls(*args, **kwargs)
                    return metric

                if enable_sota_metrics:
                    from prism_metric import PrismMetric
                    from comet_metric import Comet
                    metrics += [
                        make_metric(Comet),
                        make_metric(PrismMetric, src_lang=src_lang, tgt_lang=tgt_lang,
                                    reference_free=reference_free),
                    ]
                    pass

                metrics += [
                    make_metric(BERTScore, tgt_lang=tgt_lang, reference_free=reference_free),
                    make_metric(ContextualWMD, tgt_lang=tgt_lang, reference_free=reference_free),
                ]

                if enable_contextual_scm:
                    metrics += [make_metric(ContextualSCM, tgt_lang=tgt_lang, reference_free=reference_free)]

                metrics += [
                    make_metric(DecontextualizedWMD, tgt_lang=tgt_lang, use_tfidf=False,
                                reference_free=reference_free),
                    make_metric(DecontextualizedWMD, tgt_lang=tgt_lang, use_tfidf=True,
                                reference_free=reference_free),
                    make_metric(DecontextualizedSCM, tgt_lang=tgt_lang, use_tfidf=False,
                                reference_free=reference_free),
                    make_metric(DecontextualizedSCM, tgt_lang=tgt_lang, use_tfidf=True,
                                reference_free=reference_free),
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

                metrics = list(filter(lambda metric: metric is not None, metrics))

                regression = make_metric(Regression, metrics, reference_free=reference_free)
                regression_baseline = make_metric(Regression, None, reference_free=reference_free)
                assert regression is not None and regression_baseline is not None
                metrics = [regression] + metrics + [regression_baseline]

                evaluator = Evaluator("data_dir", lang_pair, metrics, ["Regression", "Regression_baseline"],
                                      judgements_type=judgements_type, human=human,
                                      reference_free=reference_free, firstn=firstn)
                if judgements_type in evaluator.submission_judgement_types:
                    evaluator.submit_and_report()
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
