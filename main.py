from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import pandas as pd

from bertscore import BERTScore  # noqa: F401
from conventional_metrics import BLEU, METEOR  # noqa: F401
from scm import SCM, ContextualSCM, DecontextualizedSCM  # noqa: F401
from wmd import WMD, ContextualWMD, DecontextualizedWMD  # noqa: F401
from common import Evaluator

if __name__ == '__main__':
    JUDGEMENTS_TYPE = "MQM"

    metrics = [
        BLEU(),
        METEOR(),
        BERTScore(tgt_lang="en"),
        ContextualSCM(tgt_lang="en"),
        ContextualWMD(tgt_lang="en"),
        DecontextualizedSCM(tgt_lang="en", use_tfidf=False),
        DecontextualizedSCM(tgt_lang="en", use_tfidf=True),
        DecontextualizedWMD(tgt_lang="en", use_tfidf=False),
        DecontextualizedWMD(tgt_lang="en", use_tfidf=True),
        # SCM(tgt_lang="en", use_tfidf=False),
        # SCM(tgt_lang="en", use_tfidf=True),
        # WMD(tgt_lang="en", use_tfidf=False),
        # WMD(tgt_lang="en", use_tfidf=True),
    ]
    correlations = {m.label: {} for m in metrics}
    correlations["human"] = {}

    reports = []

    langs = Evaluator.langs_qm if "QM" in JUDGEMENTS_TYPE is not None else Evaluator.langs

    for lang_pair in [pair for pair in langs if pair.split("-")[-1] == "en"]:
        print("Evaluating lang pair %s" % lang_pair)
        evaluator = Evaluator("data_dir", lang_pair, metrics, judgements_type=JUDGEMENTS_TYPE, firstn=100)
        report = evaluator.evaluate()
        reports.append(report)

        human_judgements = report["human"]
        for metric_label, vals in report.items():
            correlations[metric_label][lang_pair] = spearmanr(vals, human_judgements).correlation

        pearson = pd.DataFrame(report).applymap(float).corr(method="pearson").applymap(abs)
        assert pearson.gt(0.4).all(axis=None), pearson
        sns.heatmap(pearson, annot=True)
        plt.tight_layout()
        plt.show()
        plt.savefig('heatmap-pearson-%s.png' % lang_pair)
        plt.clf()

        spearman = pd.DataFrame(report).applymap(float).corr(method="spearman").applymap(abs)
        assert spearman.gt(0.4).all(axis=None), spearman
        sns.heatmap(spearman, annot=True)
        plt.show()
        plt.tight_layout()
        plt.savefig('heatmap-spearman-%s.png' % lang_pair)
        plt.clf()

    print("Done")
