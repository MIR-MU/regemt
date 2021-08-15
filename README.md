# RegEMT: Regressive ensemble for machine translation evaluation

[![Test and publish](https://github.com/MIR-MU/regemt/workflows/Test%20and%20publish/badge.svg)](https://github.com/MIR-MU/regemt/actions?query=workflow%3ATest%20and%20publish)

The `master` branch contains sources for reproducing our results reported in
the WMT21 Metrics workshop.

See `ablation-study` for evaluating an impact of each of the ensembled metrics
to the result, `xling` for zero-shot cross-lingual metric evaluation,
`multiling` for evaluation of the fit on multiple languages, `test_judgements`
for re-generating the submission, and `docker-build` for building a Docker image.

### How to reproduce our results

#### Docker

To reproduce our results, you can use [our `miratmu/regemt` Docker
image][docker] using the [NVIDIA Container Toolkit][nvidia-docker]:

```sh
mkdir submit_dir
chmod 777 submit_dir

# test the installation on a data subsample before running the full evaluation process:
docker run --rm --gpus all -v "$PWD"/submit_dir:/submit_dir miratmu/regemt --fast

# simply run the evaluation on the full data sets:
# this takes ~4hrs on Tesla T4, might take longer on CPU
docker run --rm --gpus all -v "$PWD"/submit_dir:/submit_dir miratmu/regemt
```

The evaluation process will generate the correlation reports in `.png` and
`.pdf` format for each of the evaluated configurations into the `submit_dir/`
directory.

#### Python

Alternatively, you can install our package using Python:

```sh
git clone https://github.com/MIR-MU/regemt.git
cd regemt
chmod 777 submit_dir

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
`.pdf` format for each of the evaluated configurations into the `regemt/`
directory.

***

We're trying to keep it simple, but if you get into any trouble, or have a
question, don't hesitate to [create an issue][issues] and we'll take a look!

 [docker]: https://hub.docker.com/r/miratmu/regemt
 [nvidia-docker]: https://github.com/NVIDIA/nvidia-docker
 [issues]: https://github.com/MIR-MU/regemt/issues
