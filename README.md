# RegEMT: Regressive ensemble for machine translation evaluation

This branch contains sources for reproducing the results reported in WMT21 Metrics workshop.

See `ablation-study` for evaluating an impact of each of the ensembled metrics to the result, `xling` for zero-shot cross-lingual metric evaluation, `multiling` for evaluation of the fit on multiple languages, or `wmt21_metrics` for re-generating the submission.


### Results: steps to reproduce:

```sh
git clone https://github.com/MIR-MU/regemt.git

cd regemt

conda create --name wmt_eval python=3.7.11

pip install -r requirements.txt

# test the installation on a data subsample before running the full evaluation process:
python -m main --fast

# simply run the evaluation on the full data sets: 
# this takes ~4hrs on Tesla T4, might take longer on CPU
python -m main
```

The evaluation process will generate the correlation reports in `.png` and `.pdf` format for each of the evaluated configurations into the current directory.
