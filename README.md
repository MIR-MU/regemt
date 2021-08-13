# RegEMT: Regressive ensemble for machine translation evaluation

This branch contains sources for reproducing the results reported in WMT21
Metrics workshop.

See `ablation-study` for evaluating an impact of each of the ensembled metrics
to the result, `xling` for zero-shot cross-lingual metric evaluation,
`multiling` for evaluation of the fit on multiple languages, or
`test_judgements` for re-generating the submission.

### How to reproduce our results

To reproduce our results, you can use [our `miratmu/regemt` Docker
image][docker] using the [NVIDIA Container Toolkit][nvidia-docker]:

```sh
mkdir submit_dir

# test the installation on a data subsample before running the full evaluation process:
docker run --rm --gpus all -u $(id -u):$(id -g) -v "$PWD"/submit_dir:/app/mt-eval/submit_dir miratmu/regemt --fast

# simply run the evaluation on the full data sets:
# this takes ~4hrs on Tesla T4, might take longer on CPU
docker run --rm --gpus all -u $(id -u):$(id -g) -v "$PWD"/submit_dir:/submit_dir miratmu/regemt
```

Alternatively, you can install our package using Python:

```sh
git clone https://github.com/MIR-MU/regemt.git
cd regemt

# install the dependencies
conda create --name wmt_eval python=3.8
conda activate wmt_eval
pip install -r requirements.txt

# test the installation on a data subsample before running the full evaluation process:
python -m main --fast

# simply run the evaluation on the full data sets:
# this takes ~4hrs on Tesla T4, might take longer on CPU
python -m main
```

The evaluation process will generate the correlation reports in `.png` and
`.pdf` format for each of the evaluated configurations into the `submit_dir/`
directory.

We're trying to keep it simple, but if you get into any trouble, or have a
question, don't hesitate to [create an issue][issues] and we'll take a look!

 [docker]: https://hub.docker.com/r/miratmu/regemt
 [nvidia-docker]: https://github.com/NVIDIA/nvidia-docker
 [issues]: https://github.com/MIR-MU/regemt/issues
