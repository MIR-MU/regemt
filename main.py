import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from bertscore import BERTScore  # noqa: F401
from common import Evaluator
from conventional_metrics import BLEU, METEOR  # noqa: F401
from ood_metrics import SyntacticCompositionality
from scm import SCM, ContextualSCM  # noqa: F401
from wmd import WMD, ContextualWMD  # noqa: F401

if __name__ == '__main__':
    JUDGEMENTS_TYPE = "catastrophic"
    NO_REFERENCE = True

    metrics = [
        BLEU(),
        METEOR(),
        BERTScore(tgt_lang="de"),
        # ContextualSCM(tgt_lang="cs"),
        ContextualWMD(tgt_lang="de", reference_free=NO_REFERENCE),
        SyntacticCompositionality(tgt_lang="de", src_lang="en", reference_free=NO_REFERENCE)
        # SyntacticCompositionality(tgt_lang="cs", src_lang="en", reference_free=REFERENCE_FREE)
        # SCM(tgt_lang="en", use_tfidf=False),
        # SCM(tgt_lang="en", use_tfidf=True),
        # SCM(tgt_lang="en", use_tfidf=False),
        # WMD(tgt_lang="en"),
    ]
    correlations = {m.label: {} for m in metrics}
    correlations["human"] = {}

    reports = []

    langs = Evaluator.langs_for_judgements(JUDGEMENTS_TYPE)

    for lang_pair in langs:
        print("Evaluating lang pair %s" % lang_pair)
        evaluator = Evaluator("data_dir", lang_pair, metrics, judgements_type=JUDGEMENTS_TYPE, firstn=100)
        report = evaluator.evaluate(reference_free=NO_REFERENCE)
        reports.append(report)

        pearson = pd.DataFrame(report).applymap(float).corr(method="pearson").applymap(abs)
        assert pearson.gt(0.4).all(axis=None), pearson
        sns.heatmap(pearson, annot=True)
        plt.tight_layout()
        plt.show()
        plt.savefig('heatmap-pearson-%s.png' % lang_pair)
        plt.clf()

        spearman = pd.DataFrame(report).applymap(float).corr(method="spearman").applymap(abs)
        assert spearman.gt(0.25).all(axis=None), spearman
        sns.heatmap(spearman, annot=True)
        plt.show()
        plt.tight_layout()
        plt.savefig('heatmap-spearman-%s.png' % lang_pair)
        plt.clf()

    print("Done")
