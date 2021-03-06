import os
from itertools import product
import logging
from typing import Tuple, Set, Optional, List
import sys
import warnings

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

LOGGER = logging.getLogger(__name__)


def main(firstn: Optional[int] = None,
         reference_frees: Tuple[bool, ...] = (True, False),
         judgements_types: Tuple[str, ...] = ('MQM', ),
         src_langs: Optional[Set[str]] = None,
         tgt_langs: Optional[Set[str]] = None,
         figsize: Tuple[int, int] = (10, 10),
         enable_compositionality: bool = True,
         enable_sota_metrics: bool = True,
         enable_fasttext_metrics: bool = True,
         enable_contextual_scm: bool = False,
         submit_dir: str = '.'):
    for reference_free in reference_frees:
        print("Evaluating %sreference-free metrics" % ('' if reference_free else 'non-'))
        for judgements_type in judgements_types:
            if judgements_type == 'catastrophic' and not reference_free:
                continue

            print("Evaluating %s judgements" % judgements_type)

            reports: List[Report] = []
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

                metrics = list(filter(lambda metric: metric is not None, metrics))

                regression = make_metric(Regression, metrics, reference_free=reference_free)
                regression_baseline = make_metric(Regression, None, reference_free=reference_free)
                assert regression is not None and regression_baseline is not None
                metrics = [regression] + metrics + [regression_baseline]

                evaluator = Evaluator("data_dir", lang_pair, metrics,
                                      judgements_type=judgements_type,
                                      reference_free=reference_free, firstn=firstn)
                report = evaluator.evaluate()
                reports.append(report)

                def plot_correlations(report: Report, method: str, dpi: int = 300) -> None:
                    method_names = {
                        'pearson': "Pearson's $r$",
                        'spearman': r"Spearman's $\rho$",
                        'kendall': r"Kendall's $\tau$",
                    }
                    title = r"%s, %s%s%s, %s $\rightarrow$ %s" % \
                        (method_names[method], judgements_type, ' (reference-free)' if reference_free else '',
                         f', first {firstn}' if firstn is not None else '', src_lang, tgt_lang)
                    basename = "heatmap-%s-%s-firstn=%s-reference_free=%s-%s_%s" % \
                        (method, judgements_type, firstn, reference_free, src_lang, tgt_lang)

                    correlations = pd.DataFrame(report).applymap(float).corr(method=method).applymap(abs)

                    fig, ax = plt.subplots(figsize=figsize)
                    sns.heatmap(correlations, annot=True, ax=ax)
                    ax.set_title(title)
                    plt.show()
                    plt.tight_layout()
                    plt.savefig(os.path.join(submit_dir, f'{basename}.png'), dpi=dpi)
                    plt.savefig(os.path.join(submit_dir, f'{basename}.pdf'), dpi=dpi)
                    plt.close()

                plot_correlations(report, 'pearson')
                plot_correlations(report, 'spearman')
                plot_correlations(report, 'kendall')

                def plot_metric(metric: Regression, max_len: int = 1000, dpi: int = 300) -> None:
                    if metric.model is None:
                        raise ValueError('Using plot_metric() before fit()')

                    title = r"%s%s%s, %s $\rightarrow$ %s" % \
                        (judgements_type, ' (reference-free)' if reference_free else '',
                         f', first {firstn}' if firstn is not None else '', src_lang, tgt_lang)
                    basename = "metric-%s-%s-firstn=%s-reference_free=%s-%s_%s" % \
                        (str(metric).lower(), judgements_type, firstn, reference_free, src_lang, tgt_lang)

                    fig, ax = plt.subplots(figsize=figsize)
                    X = [[i, j] for i, j in product(range(max_len + 1), range(max_len + 1))]
                    Y = metric.model.predict(X).reshape((max_len + 1, max_len + 1))
                    ai = ax.imshow(Y, interpolation='none', origin='lower')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(ai, cax=cax)
                    ax.set_title(title)
                    ax.set_xlabel('source length' if reference_free else 'reference length')
                    ax.set_ylabel('translation length')
                    plt.show()
                    plt.tight_layout()
                    plt.savefig(os.path.join(submit_dir, f'{basename}.png'), dpi=dpi)
                    plt.savefig(os.path.join(submit_dir, f'{basename}.pdf'), dpi=dpi)
                    plt.close()

                plot_metric(regression_baseline)

    print("Done")


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
