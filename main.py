import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from bertscore import BERTScore  # noqa: F401
from common import Evaluator
from conventional_metrics import BLEU, METEOR  # noqa: F401
from ood_metrics import SyntacticCompositionality  # noqa: F401
from scm import SCM, ContextualSCM, DecontextualizedSCM  # noqa: F401
from wmd import WMD, ContextualWMD, DecontextualizedWMD  # noqa: F401

if __name__ == '__main__':
    NO_REFERENCE = False
    FIRSTN = 100

    for judgements_type in ['DA', 'MQM']:
        reports = []
        langs = Evaluator.langs_for_judgements(judgements_type)

        for lang_pair in langs:
            src_lang, tgt_lang = lang_pair.split("-")
            if tgt_lang != 'en':
                continue

            metrics = [
                BLEU(),
                METEOR(),
                BERTScore(tgt_lang="en"),
                # ContextualSCM(tgt_lang="en", reference_free=NO_REFERENCE),
                ContextualWMD(tgt_lang="en", reference_free=NO_REFERENCE),
                DecontextualizedSCM(tgt_lang="en", use_tfidf=False, reference_free=NO_REFERENCE),
                DecontextualizedSCM(tgt_lang="en", use_tfidf=True, reference_free=NO_REFERENCE),
                DecontextualizedWMD(tgt_lang="en", use_tfidf=False, reference_free=NO_REFERENCE),
                DecontextualizedWMD(tgt_lang="en", use_tfidf=True, reference_free=NO_REFERENCE),
                # SCM(tgt_lang="en", use_tfidf=False),
                # SCM(tgt_lang="en", use_tfidf=True),
                # WMD(tgt_lang="en", use_tfidf=False),
                # WMD(tgt_lang="en", use_tfidf=True),
            ]

            print("Evaluating lang pair %s" % lang_pair)
            evaluator = Evaluator("data_dir", lang_pair, metrics,
                                  judgements_type=judgements_type,
                                  reference_free=NO_REFERENCE, firstn=FIRSTN)
            report = evaluator.evaluate()
            reports.append(report)

            pearson = pd.DataFrame(report).applymap(float).corr(method="pearson").applymap(abs)
            assert pearson.gt(0.2).all(axis=None), pearson
            sns.heatmap(pearson, annot=True)
            plt.tight_layout()
            plt.show()
            plt.savefig('heatmap-pearson-%s-%s.png' % (judgements_type, lang_pair))
            plt.clf()

            spearman = pd.DataFrame(report).applymap(float).corr(method="spearman").applymap(abs)
            assert spearman.gt(0.2).all(axis=None), spearman
            sns.heatmap(spearman, annot=True)
            plt.show()
            plt.tight_layout()
            plt.savefig('heatmap-spearman-%s-%s.png' % (judgements_type, lang_pair))
            plt.clf()

    print("Done")
