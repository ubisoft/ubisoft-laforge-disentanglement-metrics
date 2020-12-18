# Supervised Disentanglement Metrics

This repository contains code for the paper:

J. Zaidi, J. Boilard, G. Gagnon and M.A. Carbonneau, "*Measuring Disentanglement: A Review of Metrics*", arXiv:2012.09276, 2020.

Please cite this paper if you use the code in this repository as part of a published research project.

## Setup

The code was run using python 3.8:

1. create a python [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
2. clone this repo: ```git clone https://github.com/Ubisoft-LaForge/ubisoft-laforge-DisentanglementMetrics.git```
3. navigate to the repository: ```cd disentanglement_metrics```
4. install python requirements: ```pip install -r requirements.txt```


## Disentanglement Metrics

All 12 disentanglement metrics studied in the paper are implemented in ```src/metrics/``` folder.

Each metric implementation takes as input:
1. a set of factors of shape ```(nb_examples, nb_factors)```
2. a set of codes of shape ```(nb_examples, nb_codes)```
3. additional hyper-parameters specific to the metric

The default hyper-parameters used for each metric in the paper are in script  ```src/experiments/config.py```

## Reproducing The Results

We provide the code that was used to compute results from experiments of Section 5.2 to 5.6.
All the scripts are in ```src/experiments/``` folder. Each script can be run in 2 modes:
* ```run```: get scores for each metric and save them
* ```plot```: plot scores for each metric family and save plots

Metric scores and plots will be automatically saved into ```results/``` folder, at the root of the repository.

For example, to reproduce noise experiment of section 5.2:
1. navigate to experiments folder: ```cd src/experiments```
2. compute metric scores: ```python section5.2_noise.py run```
3. plot scores: ```python section5.2_noise.py plot```

**NOTES**:
* By default, all metrics scores are computed. It is possible to target specific metrics as follows: ```python section5.2_noise.py run --metrics "metric_1" ... "metric_N"```
* For example, to only compute scores for *predictor-based* metrics: ```python section5.2_noise.py run --metrics "Explicitness" "SAP" "DCI Lasso" "DCI RF"```

## Feedback

Please send any feedback to marc-andre.carbonneau2@ubisoft.com and julian.zaidi@ubisoft.com.
