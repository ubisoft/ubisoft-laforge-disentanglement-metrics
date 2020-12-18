# coding=utf-8
# Copyright 2018 Ubisoft La Forge Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import pickle
import time

import numpy as np

from functools import partial

from config import METRICS, PLOTS
from utils import estimate_required_time, get_artificial_factors_dataset, \
    get_experiment_seeds, get_nb_jobs, get_score, launch_multi_process, plot_curves


'''
    Third Experiment of section 5.4: "Modular but not Compact"
'''

# set variables for the experiment
NB_JOBS = get_nb_jobs('max')
NB_RANDOM_REPRESENTATIONS = 100
NB_RUNS = 1
NB_EXAMPLES = 20000
NB_FACTORS = 2
DISTRIBUTION = [np.random.uniform, {'low': 0., 'high': 1.}]


def get_factors_codes_dataset():
    ''' Create factors-codes dataset
    '''
    # create factors dataset
    dist, dist_kwargs = DISTRIBUTION
    factors = get_artificial_factors_dataset(nb_examples=NB_EXAMPLES, nb_factors=NB_FACTORS,
                                             distribution=dist, dist_kwargs=dist_kwargs)
    
    # compute codes from continuous factors
    codes = np.concatenate((factors, factors, factors, factors), axis=1)
    
    return factors, codes


def run_compactness_experiment(sub_parser_args):
    ''' Run compactness experiment using several metrics and save score results
    
    :param sub_parser_args:     arguments of "run" sub-parser command
                                metrics (list):         metrics to use in the experiment
                                output_dir (string):    directory to save metric scores
    '''
    # extract sub-parser arguments
    metrics = sub_parser_args.metrics
    output_dir = sub_parser_args.output_dir
    
    # seeds to use for the experiment
    seeds = get_experiment_seeds(nb_representations=NB_RANDOM_REPRESENTATIONS, nb_runs=NB_RUNS)
    
    # iterate over metrics
    for metric in metrics:
        # track time
        begin = time.time()
        print(f'Running {metric} metric')
        
        # initialize scores array
        scores_array = np.zeros((NB_RANDOM_REPRESENTATIONS, NB_RUNS)).squeeze()
        
        # depending on the metric, we can have several scores per representation
        if 'DCI' in metric and 'MIG' not in metric: 
            # DCI metric returns Modularity, Compactness and Explicitness scores
            metric_scores = {f'{metric} Mod': scores_array.copy(),
                             f'{metric} Comp': scores_array.copy(),
                             f'{metric} Expl': scores_array.copy()}
        else:
            # only one score is returned
            metric_scores = {f'{metric}': scores_array}
        
        # set metric function and its hyper-params
        metric_func = METRICS[metric]['function']
        metric_kwargs = METRICS[metric]['kwargs']
        metric_func = partial(metric_func, **metric_kwargs)
        
        # run metric
        for _ in range(1):
            # get scores using multi-processing
            scores = launch_multi_process(iterable=seeds, func=get_score, n_jobs=NB_JOBS, timer_verbose=False,
                                          metric=metric_func, factors_codes_dataset=get_factors_codes_dataset)
            
            # fill arrays
            for idx, key in enumerate(metric_scores):
                if len(metric_scores) == 1:
                    metric_scores[key][:] = [score for score in scores]
                else:
                    metric_scores[key][:] = [score[idx] for score in scores]
            
            # display remaining time
            estimate_required_time(nb_items_in_list=len(seeds),
                                   current_index=len(seeds) - 1,
                                   time_elapsed=time.time() - begin)
        
        # save dictionaries
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)  # create output directory
            for key in metric_scores:
                with open(os.path.join(output_dir, key), 'wb') as output:
                    pickle.dump({f'{key}': metric_scores[key]}, output)
        
        # display time
        duration = (time.time() - begin) / 60
        print(f'\nTotal time to run experiment on {metric} metric -- {duration:.2f} min')


def scores_table(sub_parser_args):
    ''' Get scores table
    
    :param sub_parser_args:     arguments of "plot" sub-parser command
                                output_dir (string):    directory to save table plots
    '''
    # extract sub-parser arguments
    output_dir = sub_parser_args.output_dir
    
    # extract metric scores
    scores = {}
    metrics = [os.path.join(output_dir, x) for x in os.listdir(output_dir) if not x.endswith('.json')] 
    for metric in metrics:
        with open(metric, 'rb') as input:
            metric_scores = pickle.load(input)
        scores.update(metric_scores)
    
    # compute means
    for metric, values in scores.items():
        scores_array = values
        scores[metric] = np.mean(scores_array).item()
    
    # print to console results for easier overleaf update
    overleaf_mean = ''
    for _, metrics in PLOTS['FAMILIES'].items():
        for metric in metrics:
            if metric in scores:
                mean = scores[metric]
                overleaf_mean = f'{overleaf_mean} & {mean:.1f}'
            else:
                overleaf_mean = f'{overleaf_mean} & -'
    print(overleaf_mean.strip())
    
    # save dict to .json
    output_file = os.path.join(output_dir, 'scores.json')
    for metric in scores:
        mean = scores[metric]
        scores[metric] = f'{mean:.1f}'
    with open(output_file, 'w') as fp:
        json.dump(scores, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    # project ROOT
    FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.realpath(os.path.dirname(os.path.dirname(FILE_ROOT)))
    
    # default metrics and default output directory
    metrics = [metric for metric in METRICS]
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'section5.4_compactness3')
    
    # create parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    # parser for the "run" command -- run compactness experiment
    parser_run = subparsers.add_parser('run', help='compute scores')
    parser_run.set_defaults(func=run_compactness_experiment)
    parser_run.add_argument('--metrics', nargs='+', default=metrics, required=False,
                            help='metrics to use to compute scores: "metric_1" ... "metric_N"')
    parser_run.add_argument('--output_dir', type=str, default=output_dir, required=False,
                            help='output directory to store scores results')
    
    # parser fot the "plot" command -- plot table of metric scores
    parser_table = subparsers.add_parser('plot', help='scores table')
    parser_table.set_defaults(func=scores_table)
    parser_table.add_argument('--output_dir', type=str, default=output_dir, required=False,
                              help='output directory to store scores table')
    
    args = parser.parse_args()
    args.func(args)
