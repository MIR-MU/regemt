import os
from itertools import count
import logging
from typing import Tuple, Set, Optional, List
import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import transformers
from matplotlib import pyplot as plt
from bertscore import BERTScore
from common import Evaluator, Metric
from conventional_metrics import BLEU, METEOR
from ood_metrics import SyntacticCompositionality
from scm import SCM, ContextualSCM, DecontextualizedSCM
from wmd import WMD, ContextualWMD, DecontextualizedWMD
from ensemble import Regression


METHOD_NAMES = {
    'pearson': "Pearson's $r$",
    'spearman': r"Spearman's $\rho$",
}
LOGGER = logging.getLogger(__name__)


def main(firstn: Optional[int] = None,
         reference_frees: Tuple[bool, ...] = (True, False),
         judgements_types: Tuple[str, ...] = ('MQM', ),
         src_langs: Optional[Set[str]] = None,
         tgt_langs: Optional[Set[str]] = None,
         figsize: Tuple[int, int] = (10, 10),
         dpi: int = 300,
         enable_compositionality: bool = True,
         enable_sota_metrics: bool = True,
         enable_fasttext_metrics: bool = True,
         enable_contextual_scm: bool = False,
         ablation_study: bool = True,
         submit_dir: str = '.'):
    for reference_free in reference_frees:
        print("Evaluating %sreference-free metrics" % ('' if reference_free else 'non-'))
        for judgements_type in judgements_types:
            if judgements_type == 'catastrophic' and not reference_free:
                continue

            print("Evaluating %s judgements" % judgements_type)

            langs = Evaluator.langs_for_judgements(judgements_type)

            for lang_pair in langs:
                src_lang, tgt_lang = lang_pair.split("-")
                if src_langs is not None and src_lang not in src_langs:
                    continue
                if tgt_langs is not None and tgt_lang not in tgt_langs:
                    continue

                print("Evaluating lang pair %s" % lang_pair)

                base_metrics: List[Optional[Metric]] = []

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
                    base_metrics += [
                        make_metric(Comet),
                        make_metric(PrismMetric, src_lang=src_lang, tgt_lang=tgt_lang,
                                    reference_free=reference_free),
                    ]

                base_metrics += [
                    make_metric(BERTScore, tgt_lang=tgt_lang, reference_free=reference_free),
                    make_metric(ContextualWMD, tgt_lang=tgt_lang, reference_free=reference_free),
                ]

                if enable_contextual_scm:
                    base_metrics += [make_metric(ContextualSCM, tgt_lang=tgt_lang, reference_free=reference_free)]

                base_metrics += [
                    make_metric(DecontextualizedWMD, tgt_lang=tgt_lang, use_tfidf=False, reference_free=reference_free),
                    make_metric(DecontextualizedWMD, tgt_lang=tgt_lang, use_tfidf=True, reference_free=reference_free),
                    make_metric(DecontextualizedSCM, tgt_lang=tgt_lang, use_tfidf=False, reference_free=reference_free),
                    make_metric(DecontextualizedSCM, tgt_lang=tgt_lang, use_tfidf=True, reference_free=reference_free),
                ]

                if enable_fasttext_metrics:
                    base_metrics += [
                        make_metric(SCM, tgt_lang=tgt_lang, use_tfidf=False),
                        make_metric(SCM, tgt_lang=tgt_lang, use_tfidf=True),
                        make_metric(WMD, tgt_lang=tgt_lang, use_tfidf=False),
                        make_metric(WMD, tgt_lang=tgt_lang, use_tfidf=True),
                    ]

                if enable_sota_metrics:
                    from bleurt_metric import BLEUrt
                    base_metrics += [
                        make_metric(BLEUrt)
                    ]

                base_metrics += [
                    make_metric(BLEU),
                    make_metric(METEOR),
                ]

                if enable_compositionality:
                    base_metrics += [
                        make_metric(SyntacticCompositionality, src_lang=src_lang, tgt_lang=tgt_lang,
                                    reference_free=reference_free)
                    ]

                base_metrics: List[Metric] = list(filter(lambda metric: metric is not None, base_metrics))
                ablation_study_results: List[float] = []

                for ablation_study_step in count() if ablation_study else range(1):

                    regression = make_metric(Regression, base_metrics, reference_free=reference_free)
                    assert regression is not None
                    metrics = [regression] + base_metrics

                    evaluator = Evaluator("data_dir", lang_pair, metrics,
                                          judgements_type=judgements_type,
                                          reference_free=reference_free, firstn=firstn)
                    report = evaluator.evaluate()

                    def plot_correlations(method: str) -> pd.DataFrame:
                        title = r"%s, %s%s%s, %s $\rightarrow$ %s" % \
                            (METHOD_NAMES[method], judgements_type, ' (reference-free)' if reference_free else '',
                             f', first {firstn}' if firstn is not None else '', src_lang, tgt_lang)
                        basename = "heatmap-%s-%s-firstn=%s-reference_free=%s-ablation_study=%s-%s_%s" % \
                            (method, judgements_type, firstn, reference_free,
                             ablation_study_step if ablation_study else False, src_lang, tgt_lang)

                        correlations = pd.DataFrame(report).applymap(float).corr(method=method).applymap(abs)

                        fig, ax = plt.subplots(figsize=figsize)
                        sns.heatmap(correlations, annot=True, ax=ax)
                        ax.set_title(title)
                        plt.show()
                        plt.tight_layout()
                        plt.savefig(os.path.join(submit_dir, f'{basename}.png'), dpi=dpi)
                        plt.savefig(os.path.join(submit_dir, f'{basename}.pdf'), dpi=dpi)
                        plt.close()

                        return correlations

                    def plot_ablation_study(method: str, text_offset: Tuple[int, int] = (0, 10)) -> None:
                        title = r"%s%s%s, %s $\rightarrow$ %s" % \
                            (judgements_type, ' (reference-free)' if reference_free else '',
                             f', first {firstn}' if firstn is not None else '', src_lang, tgt_lang)
                        basename = "ablation-%s-%s-firstn=%s-reference_free=%s-%s_%s" % \
                            (method, judgements_type, firstn, reference_free, src_lang, tgt_lang)

                        fig, ax = plt.subplots(figsize=figsize)
                        X, Y = zip(*enumerate(ablation_study_results))
                        ax.plot(X, Y)
                        for x, y in zip(X, Y):
                            ax.annotate(f'{y:.4f}', xy=(x, y), xytext=text_offset, textcoords='offset points')
                        ax.set_title(title)
                        ax.set_xlabel('Number of eliminated metrics')
                        ax.set_ylabel(f'{METHOD_NAMES[method]} of {regression.label}')
                        plt.show()
                        plt.tight_layout()
                        plt.savefig(os.path.join(submit_dir, f'{basename}.png'), dpi=dpi)
                        plt.savefig(os.path.join(submit_dir, f'{basename}.pdf'), dpi=dpi)
                        plt.close()

                    plot_correlations('pearson')
                    correlations = plot_correlations('spearman')
                    np.fill_diagonal(correlations.values, -np.inf)

                    if ablation_study:
                        ablation_study_result = float(correlations['human'][regression.label])
                        ablation_study_results.append(ablation_study_result)

                        if len(report) <= 3:
                            print(f'Finished the ablation study after {ablation_study_step + 1} steps')
                            plot_ablation_study('spearman')
                            break

                        worst_label, highest_correlation = None, 0
                        for label, correlation in correlations.max(axis=0).iteritems():
                            if label not in ('human', regression.label) and correlation > highest_correlation:
                                worst_label, highest_correlation = label, correlation
                        assert worst_label is not None

                        len_before = len(base_metrics)
                        base_metrics = list(filter(lambda metric: metric.label != worst_label, base_metrics))
                        assert len(base_metrics) + 1 == len_before
                        print(f'Eliminated {worst_label} in step {ablation_study_step + 1} of an ablation study')

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
