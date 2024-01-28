# RL based text classification

by Dmitry Beresnev
IU / <pro100pro10010@gmail.com>

## Introduction

TBW

## Requirements

Code was tested on Windows 11, Python 3.11 and CUDA 11.8.

<!-- All the requirement packages are listed in the file `requirements.txt`. -->

## Before start

<!-- Install all the packages from _requirements.txt_ using `pip install -r requirements.txt` or using **pipenv** `pipenv install`. -->

Optionally, you can run `bash setup_precommit.sh` to setup pre-commit hook for GitHub for code formatting using [ruff](https://docs.astral.sh/ruff/).

<!-- I also highly recommend to read reports in corresponding `reports` folder to fully understand context and purpose of some files and folders. -->

## Repository structure

```text
├── README.md       # The top-level README
│
├── notebooks       #  Jupyter notebooks
|
├── references      # Data dictionaries, manuals, and all other explanatory materials
│
├── reports         # Generated analysis as HTML, PDF, LaTeX, etc.
│
├── requirements.txt  # The requirements file for reproducing the analysis environment
│                      generated with `pip freeze › requirements. txt`
|
├── pyproject.toml            # Formatter and linter settings
└── setup_precommit.sh        # Script for creating pre-commit GitHub hook
```

## Basic usage

<!-- This section briefly describes how to use scripts from `benchmark/` folder.

For all scripts help messages are available with `-h` flag. For example, `python ./benchmark/evaluate.py -h` explains all the available flags and their purpose.
Generally, for all scripts two modes are available: verbose and non-verbose.
By default verbose mode is active, and to run the script in silent mode you need the `--no-verbose` flag. -->

Generally, you can run each script with no flags at all. However, I **highly recommend** to always read the help messages before using scripts.

<!-- `./benchmark/evaluate.py` script is used for model performance evaluation.
You can specify model (by path) and data (also by path) for evaluation. Note, that by default script interprets data path
as path to the folder with several .csv files. If you want to pass single file, enable file mode by `-f` flag.
Be default, resulting data is saved to `./benchmark/data/generated/`.

`./benchmark/interactive.py` script is used for real-time interaction with model.
You can specify user parameters, such as age (i.e. `-a 21`),
occupation (i.e. `-o 19` for student),
gender (i.e. `-g 1` for male)
and favorite movies (i.e. `-f 1 56` for "Toy Story" and "Pulp Fiction") to get new movies recommendation. -->

## References

TBW

<!-- ### Metrics

- [Retrieval precision on K](https://pytorch.org/torcheval/main/generated/torcheval.metrics.functional.retrieval_precision.html)
- [MAP@K](https://machinelearninginterview.com/topics/machine-learning/mapatk_evaluation_metric_for_ranking/)

### Datasets

- [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) -->

## Contacts

In case of any questions you can contact me via email <pro100pro10010@gmail.com> or Telegram **@flip_floppa**
