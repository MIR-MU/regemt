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
    JUDGEMENTS_TYPE = "MQM"
    NO_REFERENCE = True

    reports = []

    langs = Evaluator.langs_for_judgements(JUDGEMENTS_TYPE)

    for lang_pair in langs:
        src_lang, tgt_lang = lang_pair.split("-")
        metrics = [
            ContextualWMD(tgt_lang=tgt_lang, reference_free=NO_REFERENCE),
            BERTScore(tgt_lang=tgt_lang, reference_free=NO_REFERENCE),
            # ContextualSCM(tgt_lang=tgt_lang, reference_free=NO_REFERENCE),
            DecontextualizedSCM(tgt_lang=tgt_lang, use_tfidf=False, reference_free=NO_REFERENCE),
            DecontextualizedSCM(tgt_lang=tgt_lang, use_tfidf=True, reference_free=NO_REFERENCE),
            DecontextualizedWMD(tgt_lang=tgt_lang, use_tfidf=False, reference_free=NO_REFERENCE),
            DecontextualizedWMD(tgt_lang=tgt_lang, use_tfidf=True, reference_free=NO_REFERENCE),
            # # SyntacticCompositionality needs PoS tagger, that can be automatically resolved only for de + zh
            SyntacticCompositionality(tgt_lang=tgt_lang, src_lang=src_lang, reference_free=NO_REFERENCE)
        ]

        print("Evaluating lang pair %s" % lang_pair)
        evaluator = Evaluator("data_dir", lang_pair, metrics,
                              judgements_type=JUDGEMENTS_TYPE, reference_free=NO_REFERENCE, firstn=100000)
        report = evaluator.evaluate()
        reports.append(report)

        pearson = pd.DataFrame(report).applymap(float).corr(method="pearson").applymap(abs)
        assert pearson.gt(0.001).all(axis=None), pearson
        sns.heatmap(pearson, annot=True)
        plt.tight_layout()
        plt.show()
        plt.savefig('heatmap_no_reference-pearson-%s.png' % lang_pair)
        plt.clf()

        spearman = pd.DataFrame(report).applymap(float).corr(method="spearman").applymap(abs)
        assert spearman.gt(0.001).all(axis=None), spearman
        sns.heatmap(spearman, annot=True)
        plt.show()
        plt.tight_layout()
        plt.savefig('heatmap_no_reference-spearman-%s.png' % lang_pair)
        plt.clf()

    print("Done")
