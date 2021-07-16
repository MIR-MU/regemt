import os
import logging
from typing import Tuple, Set, Optional, List
import sys

import pandas as pd
import seaborn as sns
import transformers
from matplotlib import pyplot as plt
from bertscore import BERTScore
from common import Evaluator, Report
from conventional_metrics import BLEU, METEOR
from ood_metrics import SyntacticCompositionality
from scm import SCM, ContextualSCM, DecontextualizedSCM
from wmd import WMD, ContextualWMD, DecontextualizedWMD
from ensemble import Regression


def main(firstn: Optional[float] = None,
         reference_frees: Tuple[bool, ...] = (False, True),
         judgements_types: Tuple[str, ...] = ('DA', 'MQM', 'catastrophic'),
         src_langs: Optional[Set[str]] = None,
         tgt_langs: Optional[Set[str]] = None,
         figsize: Tuple[int, int] = (10, 10),
         enable_compositionality: bool = False,
         enable_fasttext_metrics: bool = True,
         enable_contextual_scm: bool = False):
    for reference_free in reference_frees:
        print("Evaluating %sreference-free metrics" % ('' if reference_free else 'non-'))
        for judgements_type in judgements_types:
            print("Evaluating %s judgements" % judgements_type)

            reports: List[Report] = []
            langs = Evaluator.langs_for_judgements(judgements_type)

            for lang_pair in langs:
                src_lang, tgt_lang = lang_pair.split("-")
                if src_langs is not None and src_lang not in src_langs:
                    continue
                if tgt_langs is not None and tgt_lang not in tgt_langs:
                    continue

                metrics = [
                    BERTScore(tgt_lang=tgt_lang, reference_free=reference_free),
                    ContextualWMD(tgt_lang=tgt_lang, reference_free=reference_free),
                ]

                if enable_contextual_scm:
                    metrics += [ContextualSCM(tgt_lang=tgt_lang, reference_free=reference_free)]

                metrics += [
                    DecontextualizedWMD(tgt_lang=tgt_lang, use_tfidf=False, reference_free=reference_free),
                    DecontextualizedWMD(tgt_lang=tgt_lang, use_tfidf=True, reference_free=reference_free),
                    DecontextualizedSCM(tgt_lang=tgt_lang, use_tfidf=False, reference_free=reference_free),
                    DecontextualizedSCM(tgt_lang=tgt_lang, use_tfidf=True, reference_free=reference_free),
                ]

                if enable_fasttext_metrics:
                    metrics += [
                        SCM(tgt_lang=tgt_lang, use_tfidf=False),
                        SCM(tgt_lang=tgt_lang, use_tfidf=True),
                        WMD(tgt_lang=tgt_lang, use_tfidf=False),
                        WMD(tgt_lang=tgt_lang, use_tfidf=True),
                    ]

                metrics += [
                    BLEU(),
                    METEOR(),
                ]

                if enable_compositionality:
                    metrics += [SyntacticCompositionality(src_lang=src_lang, tgt_lang=tgt_lang,
                                                          reference_free=reference_free)]

                metrics = [Regression(metrics, reference_free=reference_free)] + metrics

                print("Evaluating lang pair %s" % lang_pair)
                evaluator = Evaluator("data_dir", lang_pair, metrics,
                                      judgements_type=judgements_type,
                                      reference_free=reference_free, firstn=firstn)
                report = evaluator.evaluate()
                reports.append(report)

                def plot_correlations(report: Report, method: str, dpi: int = 300) -> None:
                    method_names = {
                        'pearson': "Pearson's $r$",
                        'spearman': r"Spearman's $\rho$",
                    }
                    title = r"%s, %s%s%s, %s $\rightarrow$ %s" % \
                        (method_names[method], judgements_type, ' (reference-free)' if reference_free else '',
                         f', first {firstn}' if firstn is not None else '', src_lang, tgt_lang)
                    basename = "heatmap-%s-%s-firstn=%s-reference_free=%s-%s_%s" % \
                        (method, judgements_type, firstn, reference_free, src_lang, tgt_lang)

                    correlations = pd.DataFrame(report).applymap(float).corr(method=method).applymap(abs)

                    plt.clf()
                    fig, ax = plt.subplots(figsize=figsize)
                    sns.heatmap(correlations, annot=True, ax=ax)
                    ax.set_title(title)
                    plt.show()
                    plt.tight_layout()
                    plt.savefig(f'{basename}.png', dpi=dpi)
                    plt.savefig(f'{basename}.pdf', dpi=dpi)

                plot_correlations(report, 'pearson')
                plot_correlations(report, 'spearman')

    print("Done")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    parameters = dict()
    for arg in sys.argv[1:]:
        if arg == '--fast':
            parameters = {
                **parameters,
                **{
                    'firstn': 100,
                    'judgements_types': ('MQM',),
                    'src_langs': {'en'},
                    'enable_fasttext_metrics': False,
                }
            }
        else:
            raise ValueError(f'Unrecognized command-line argument: {arg}')

    main(**parameters)
