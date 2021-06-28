from matplotlib import pyplot as plt
from scipy.stats import spearmanr

from bertscore import BERTScore  # noqa: F401
from conventional_metrics import BLEU, METEOR  # noqa: F401
from scm import SCM, ContextualSCM  # noqa: F401
from wmd import WMD  # noqa: F401
from common import Evaluator
import pandas as pd

if __name__ == '__main__':
    JUDGEMENTS_TYPE = "MQM"

    metrics = [
        BLEU(),
        METEOR(),
        BERTScore(tgt_lang="en"),
        # ContextualSCM(tgt_lang="en"),
        # SCM(tgt_lang="en", use_tfidf=False),
        # SCM(tgt_lang="en", use_tfidf=True),
        # SCM(tgt_lang="en", use_tfidf=False),
        # WMD(tgt_lang="en"),
    ]
    correlations = {m.label: {} for m in metrics}
    correlations["human"] = {}

    reports = []

    langs = Evaluator.langs_qm if "QM" in JUDGEMENTS_TYPE is not None else Evaluator.langs

    for lang_pair in [pair for pair in langs if pair.split("-")[-1] == "en"]:
        print("Evaluating lang pair %s" % lang_pair)
        evaluator = Evaluator("data_dir", lang_pair, metrics, judgements_type=JUDGEMENTS_TYPE)
        report = evaluator.evaluate()
        reports.append(report)

        human_judgements = report["human"]
        for metric_label, vals in report.items():
            correlations[metric_label][lang_pair] = spearmanr(vals, human_judgements).correlation

    sns.heatmap(pd.DataFrame(report).applymap(float).corr(method="pearson").applymap(abs), annot=True)
    plt.show()
    sns.heatmap(pd.DataFrame(report).applymap(float).corr(method="spearman").applymap(abs), annot=True)
    plt.show()

    print("Done")
