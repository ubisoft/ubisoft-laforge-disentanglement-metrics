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
import multiprocessing as mp
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from dateutil.relativedelta import relativedelta
from functools import partial
from itertools import cycle
from multiprocessing import Pool

from pyitlib import discrete_random_variable as drv


def get_artificial_factors_dataset(nb_examples, nb_factors, distribution, dist_kwargs):
    ''' Create artificial factors dataset using a specified distribution
        Each column is a factor and each line is a data point
    
    :param nb_examples:         number of examples in the dataset
    :param nb_factors:          number of generative factors
    :param distribution:        distribution of the artificial factors
    :param dist_kwargs:         additional keyword arguments for the distribution
    '''
    # initialize factors dataset
    factors = np.zeros((nb_examples, nb_factors))
    
    # fill array with random continuous factors values from the distribution
    for line_idx in range(0, nb_examples):
        for column_idx in range(0, nb_factors):
            factor_value = distribution(**dist_kwargs)
            factors[line_idx, column_idx] = factor_value
    
    return factors


def get_score(items, metric, factors_codes_dataset):
    ''' Compute metric score on a specific factors-codes representation
    
    :param items:                      representation seed and run seed
    :param metric:                     metric to use to compute score
    :param factors_codes_dataset:      function to create factors-codes dataset
    '''
    # extract seeds
    representation_seed, run_seed = items
    
    # create factors-codes dataset
    np.random.seed(representation_seed)
    factors, codes = factors_codes_dataset()
    
    # compute score
    np.random.seed(run_seed)
    score = metric(factors=factors, codes=codes)
    
    return score


def get_experiment_seeds(nb_representations, nb_runs):
    ''' Extract all seeds to use in the experiment
    
    :param nb_representations:      number of random representations to generate
    :param nb_runs:                 number of times we run the metric on the same random representation
    '''
    # seeds corresponding to different random representations
    repr_seeds = [repr_seed for repr_seed in range(nb_representations)]
    
    # seeds corresponding to the experiment runs
    # each pair of representation/run has a unique seed
    # it allows to take into account the stochasticity of the metrics
    run_seeds = [run_seed for run_seed in range(nb_representations * nb_runs)]
    
    # combine representation seeds with their corresponding run seeds
    seeds = [(repr_seed, run_seed)
             for repr_idx, repr_seed in enumerate(repr_seeds)
             for run_idx, run_seed in enumerate(run_seeds)
             if repr_idx * nb_runs <= run_idx < (repr_idx + 1) * nb_runs]
    
    return seeds


def get_bin_index(x, nb_bins):
    ''' Discretize input variable
    
    :param x:           input variable
    :param nb_bins:     number of bins to use for discretization
    '''
    # get bins limits
    bins = np.linspace(0, 1, nb_bins + 1)

    # discretize input variable
    return np.digitize(x, bins[:-1], right=False).astype(int)


def get_mutual_information(x, y, normalize=True):
    ''' Compute mutual information between two random variables
    
    :param x:      random variable
    :param y:      random variable
    '''
    if normalize:
        return drv.information_mutual_normalised(x, y, norm_factor='Y', cartesian_product=True)
    else:
        return drv.information_mutual(x, y, cartesian_product=True)


def plot_curves(plots, output_file, format='png', figsize=(25,6), x_scale='linear',
                y_scale='linear', x_lims=None, y_lims=(-0.05, 1.05), x_label="", y_label="",
                title_font_size=22, axis_font_size=16, legend_font_size=13, colors=None,
                line_styles=None, legend_positions=None, legend_columns=None):
    ''' Plot curves from a dictionary into the same figure
    
    :param plots:                   {title: {curves_to_plot}} dictionary
                                    the following format is assumed:
                                    {
                                        title_plot_1: 
                                        {
                                            <legend_curve_1>: ([x0, x1, ..., xN], [y0, y1, ..., yN]),
                                            ...
                                        }
                                        ...
                                    }    
    :param output_file:             where and under which name the plot is saved
    :param format:                  format to save the plot
    :param figsize:                 size of the figure
    :param x_scale:                 scale on x axis
    :param y_scale:                 scale on y axis
    :param x_lims:                  limit values on x axis
    :param y_lims:                  limit values on y axis
    :param x_label:                 label for the x axis
    :param y_label:                 label for the y axis
    :param title_font_size:         font size for title of each plot
    :param axis_font_size:          font size of axis labels                            
    :param legend_font_size:        font size of legend labels
    :param colors:                  colors to use for the curves in the plots
    :param line_styles:             line styles to use for the curves in the plot
    :param legend_positions:        positioning of the legend for each plot
    :param legend_columns:          number of columns in the legend for each plot
    '''
    _, axes = plt.subplots(nrows=1, ncols=len(plots), figsize=figsize)
    
    for idx, title in enumerate(plots):
        # extract legend labels
        curves = plots[title]
        legends = [legend for legend in curves]
        
        # cycle through colors
        colors_cycle = cycle(colors) if colors is not None else cycle(['blue', 'green', 'red'])
        colors_cycle = [next(colors_cycle) for _ in range(len(curves))]
        
        # cycle through line styles
        lines_cycle = cycle(line_styles) if line_styles is not None else cycle(['-'])
        lines_cycle = [next(lines_cycle) for _ in range(len(curves))]
        
        # plot curves
        ax = axes.ravel()[idx]
        for datapoints, legend, color, line_style in zip(curves.values(), legends, colors_cycle, lines_cycle):
            x, y = datapoints
            ax.plot(x, y, label=legend, color=color, linestyle=line_style)
        
        # set scales on each axis
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        
        # set limits on x and y axis
        if x_lims is not None:
            ax.set_xlim(x_lims[0], x_lims[1])
        if y_lims is not None:
            ax.set_ylim(y_lims[0], y_lims[1])
        
        # set title and labels on axis
        ax.set_title(title, fontsize=title_font_size, pad=20)
        ax.set_xlabel(x_label, fontsize=axis_font_size)
        ax.set_ylabel(y_label, fontsize=axis_font_size, labelpad=10)
        
        # set x-ticks and y-ticks font size
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(axis_font_size)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(axis_font_size) 
        
        # set legends
        if legend_positions is not None and legend_columns is not None:
            ax.legend(handlelength=2, loc='lower center', bbox_to_anchor=legend_positions[idx],
                      fontsize=legend_font_size, ncol=legend_columns[idx])
        elif legend_positions is not None:
            ax.legend(handlelength=2, loc='lower center', bbox_to_anchor=legend_positions[idx], fontsize=legend_font_size)
        elif legend_columns is not None:
            ax.legend(handlelength=2, loc='lower center', fontsize=legend_font_size, ncol=legend_columns[idx])
        else:
            ax.legend(handlelength=2, loc='lower center', fontsize=legend_font_size)
    
    # save the plot
    save_dir = os.path.dirname(output_file)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{output_file}.{format}', bbox_inches='tight', format=format)
    plt.close()


def get_nb_jobs(n_jobs):
    """ Return the number of parallel jobs specified by n_jobs

    :param n_jobs:      the number of jobs the user want to use in parallel

    :return: the number of parallel jobs
    """
    # set nb_jobs to max by default
    nb_jobs = mp.cpu_count()

    if n_jobs != 'max':
        if int(n_jobs) > mp.cpu_count():
            print(f'Max number of parallel jobs is "{mp.cpu_count()}" but received "{int(n_jobs)}" -- '
                  f'setting nb of parallel jobs to {nb_jobs}')
        else:
            nb_jobs = int(n_jobs)

    return nb_jobs


def launch_multi_process(iterable, func, n_jobs, chunksize=1, ordered=True, timer_verbose=True, interval=100, **kwargs):
    """ Calls function using multi-processing pipes
        https://guangyuwu.wordpress.com/2018/01/12/python-differences-between-imap-imap_unordered-and-map-map_async/

    :param iterable:        items to process with function func
    :param func:            function to multi-process
    :param n_jobs:          number of parallel jobs to use
    :param chunksize:       size of chunks given to each worker
    :param ordered:         True: iterable is returned while still preserving the ordering of the input iterable
                            False: iterable is returned regardless of the order of the input iterable -- better perf
    :param timer_verbose:   display time estimation when set to True
    :param interval:        estimate remaining time when (current_index % interval) == 0
    :param kwargs:          additional keyword arguments taken by function func

    :return: function outputs
    """
    # define pool of workers
    pool = Pool(processes=n_jobs)

    # define partial function and pool function
    func = partial(func, **kwargs)
    pool_func = pool.imap if ordered else pool.imap_unordered

    # initialize variables
    func_returns = []
    nb_items_in_list = len(iterable) if timer_verbose else None
    start = time.time() if timer_verbose else None

    # iterate over iterable
    for i, func_return in enumerate(pool_func(func, iterable, chunksize=chunksize)):
        # store function output
        func_returns.append(func_return)

        # compute remaining time
        if timer_verbose:
            estimate_required_time(nb_items_in_list=nb_items_in_list, current_index=i,
                                   time_elapsed=time.time() - start, interval=interval)
    if timer_verbose:
        sys.stdout.write('\n')

    # wait for all worker to finish and close the pool
    pool.close()
    pool.join()

    return func_returns


def prog_bar(i, n, bar_size=16):
    """ Create a progress bar to estimate remaining time

    :param i:           current iteration
    :param n:           total number of iterations
    :param bar_size:    size of the bar

    :return: a visualisation of the progress bar
    """
    bar = ''
    done = (i * bar_size) // n

    for j in range(bar_size):
        bar += '█' if j <= done else '░'

    message = f'{bar} {i}/{n}'
    return message


def estimate_required_time(nb_items_in_list, current_index, time_elapsed, interval=100):
    """ Compute a remaining time estimation to process all items contained in a list

    :param nb_items_in_list:        all list items that have to be processed
    :param current_index:           current list index, contained in [0, nb_items_in_list - 1]
    :param time_elapsed:            time elapsed to process current_index items in the list
    :param interval:                estimate remaining time when (current_index % interval) == 0

    :return: time elapsed since the last time estimation
    """
    current_index += 1  # increment current_idx by 1
    if current_index % interval == 0 or current_index == nb_items_in_list:
        # make time estimation and put to string format
        seconds = (nb_items_in_list - current_index) * (time_elapsed / current_index)
        time_estimation = relativedelta(seconds=int(seconds))
        time_estimation_string = f'{time_estimation.hours:02}:{time_estimation.minutes:02}:{time_estimation.seconds:02}'

        # extract progress bar
        progress_bar = prog_bar(i=current_index, n=nb_items_in_list)

        # display info
        if current_index == nb_items_in_list:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string} -- Finished!')
        else:
            sys.stdout.write(f'\r{progress_bar} -- estimated required time = {time_estimation_string}')
