from scipy.stats import pearsonr
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--metricname")
parser.add_argument("-s", "--src",  help="include this if your metric is reference-free", default=False,
                    action='store_true')

args = parser.parse_args()

print(args)
metricname = args.metricname
if args.src:
    metrictype = 'src'
else:
    metrictype = 'ref'


cols = {
    'sys': ['metric', 'lp', 'testset', 'refset', 'sysid', 'score'],
    'seg': ['metric', 'lp', 'testset', 'refset', 'sysid', 'segid',  'score'],
}
colformat = {
    'sys': '<METRIC NAME>   <LANG-PAIR>   <TEST SET>    <REF SET>   <SYSTEM>   <SYSTEM LEVEL SCORE>',
    'seg': '<METRIC NAME>   <LANG-PAIR>   <TESTSET>    <REF SET>   <SYSTEM-ID>   <SEGMENT-ID>   SEGMENT SCORE>',
}

for level in ['sys', 'seg']:
    print('checking', level, "level:")

    try:
        mymetric = pd.read_csv(f'./{metricname}.{level}.score', sep='\t', header=None)
#         mymetric = pd.read_csv(f'./{metricname}.{level}.score.gz', sep='\t', header=None)

    except FileNotFoundError:
        print(f"File not found: './{metricname}.{level}.score'")
        continue

    try:
        demo = pd.read_csv(f'./{metrictype}-metric.{level}.score', sep='\t', header=None)
    except FileNotFoundError:
        print(f"Please download the f'./{metrictype}-metric.{level}.score' from the Google Drive Folder")
        continue

    if len(mymetric.columns) != len(cols[level]):
        print(f"Columns of './{metricname}.{level}.score' should be in format {colformat[level]}  ")
        continue

    demo.columns = cols[level]
    mymetric.columns = cols[level]
    if any(mymetric.score.isna()):
        print('Please check scores of segments for rows:')
        metric_na = mymetric[mymetric['score'].isna()]
        for row in metric_na.iterrows():
            print(row)

    for refset, rdf in demo.groupby('refset'):
        for testset, tdf in rdf.groupby('testset'):
            if testset not in mymetric.testset.unique():
                print(f'{testset}: Metric scores missing for the testset. ')
                continue

            for lp, ldf in tdf.groupby('lp'):
                metriclp = mymetric[(mymetric.testset == testset) & (mymetric.refset == refset) & (mymetric.lp == lp)]

                if len(metriclp) == 0:
                    print(f'{refset}, {lp}: Metric scores missing for the language pair and refset.')
                    continue

                if set(ldf.sysid.unique()) - set(metriclp.sysid.unique()) - set([refset]):
                    print(f'{refset}, {lp}: Metric scores missing for systems:',
                          set(ldf.sysid.unique()) - set(metriclp.sysid.unique()) - set([refset]))
                elif level == 'sys':
                    merged_scores = pd.merge(ldf, metriclp, how='left', on=['lp', 'testset', 'refset', 'sysid'])
                    merged_scores = merged_scores[merged_scores.refset != merged_scores.sysid]
                    mergedna = merged_scores[merged_scores['score_y'].isna()]
                    try:
                        subm_scores = [float(sc) for sc in merged_scores.score_y]
                    except Exception as e:
                        print(f'{refset}, {lp}: error', e)
                        print()
                        continue
                    try:
                        (pearsonr(merged_scores.score_x, subm_scores))
                    except Exception as e:
                        print(f'{refset}, {lp}: error somewhere when attemptng to compute pearson correlation between '
                              'the scores of the demo metric and  your metric.')
                        print('error: ', e)
                        print()
                elif level == 'seg':
                    merged_scores = pd.merge(ldf, metriclp,
                                             how='left', on=['lp', 'testset', 'refset', 'sysid', 'segid'])
                    merged_scores = merged_scores[merged_scores.refset != merged_scores.sysid]
                    mergedna = merged_scores[merged_scores['score_y'].isna()]
                    if len(mergedna):

                        print(f'{refset}, {lp}: Metric scores missing for segments:')
                        print('\t', 'lp', 'refset', 'sysid', 'segid')

                        for _, row in mergedna.iterrows():
                            # print(row)
                            print('\t', row['lp'], row['refset'], row['sysid'], row['segid'])
                    else:
                        try:
                            subm_scores = [float(sc) for sc in merged_scores.score_y]
                        except Exception as e:
                            print(f'{refset}, {lp}: error:', e)
                            print()
                            continue
                        try:
                            (pearsonr(merged_scores.score_x, subm_scores))
                        except Exception as e:
                            print(f'{refset}, {lp}: error somewhere when attemptng to compute pearson correlation '
                                  'between the scores of the demo metric and  your metric.')
                            print('error: ', e)
                            print()

    print('Done!')
    print('If you cant find the source of any errors that this script flags, please submit your metrics ASAP with '
          'subject "WMT Metrics submission (error)" so we can work with you to solve this')
