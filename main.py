from scipy.stats import spearmanr

from bertscore import BERTScore  # noqa: F401
from conventional_metrics import BLEU, METEOR
from scm import SCM, ContextualSCM  # noqa: F401
from wmd import WMD  # noqa: F401
from common import Evaluator
import pandas as pd

if __name__ == '__main__':
    USE_PSQM_JUDGEMENTS = False

    metrics = [
        BLEU(),
        METEOR(),
        # BERTScore(tgt_lang="en"),
        # ContextualSCM(tgt_lang="en"),
        # SCM(tgt_lang="en", use_tfidf=False),
        SCM(tgt_lang="en", use_tfidf=True),
        # SCM(tgt_lang="en", use_tfidf=False),
        # WMD(tgt_lang="en"),
    ]
    correlations = {m.label: {} for m in metrics}
    correlations["human"] = {}

    langs = Evaluator.langs_psqm if USE_PSQM_JUDGEMENTS else Evaluator.langs

    for lang_pair in [pair for pair in langs if pair.split("-")[-1] == "en"]:
        print("Evaluating lang pair %s" % lang_pair)
        source_language, target_language = lang_pair.split("-")
        evaluator = Evaluator("data_dir", (source_language, target_language), metrics, psqm=USE_PSQM_JUDGEMENTS)
        report = evaluator.evaluate()
        print(report)
        human_judgements = report["human"]
        for metric_label, vals in report.items():
            correlations[metric_label][lang_pair] = spearmanr(vals, human_judgements).correlation
    print(pd.DataFrame(correlations))

    print("Done")
