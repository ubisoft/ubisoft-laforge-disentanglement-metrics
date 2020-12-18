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
import numpy as np

from numpy.core.numeric import NaN
from sklearn.preprocessing import minmax_scale

from utils import get_bin_index

    
def z_min_var(factors, codes, continuous_factors=True, nb_bins=10, batch_size=200, nb_training=800,
              nb_eval=800, nb_variance_estimate=10000, std_threshold=0.05, scale=True, verbose=False):
    ''' Z-min Variance metric from H. Kim and A. Mnih,
        “Disentangling by factorising,”
        in ICML, 2018.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    :param batch_size:                      size of batch
    :param nb_training:                     number of training points
    :param nb_eval:                         number of evaluation points
    :param nb_variance_estimate:            number of points to use for global variance estimation
    :param std_threshold:                   minimum accepted standard deviation
    :param scale:                           if True, the output will be scaled from 0 to 1
    :param verbose:                         if True, print warnings
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
    
    # compute an estimation of empirical variance for each latent dimension
    lines_idx = np.arange(codes.shape[0])
    np.random.shuffle(lines_idx)
    var_codes = codes[lines_idx][:nb_variance_estimate]
    emp_variances = _compute_variances(var_codes, axis=0)
    
    # identify active latent dimensions
    active_dims = _prune_dims(emp_variances, threshold=std_threshold, verbose=verbose)
    
    # prepare Z-min-var datasets for training and evaluation
    train_set, eval_set = _prepare_datasets(factors=factors, codes=codes, batch_size=batch_size,
                                            nb_training=nb_training, nb_eval=nb_eval, verbose=verbose,
                                            variances=emp_variances, active_dims=active_dims)
    
    # discretization is too fine grained -- score cannot be computed correctly
    if train_set is NaN and eval_set is NaN:
        return NaN
    
    # compute training accuracy
    training_votes = _compute_votes(inputs=train_set[:, 0], targets=train_set[:, 1],
                                    nb_codes=nb_codes, nb_factors=nb_factors)
    
    latent_idx = np.arange(nb_codes)  # (nb_codes, )
    classifier = np.argmax(training_votes, axis=1)  # (nb_codes, )
    train_accuracy = np.sum(training_votes[latent_idx, classifier]) * 1. / np.sum(training_votes)
    
    # compute evaluation accuracy
    eval_votes = _compute_votes(inputs=eval_set[:, 0], targets=eval_set[:, 1],
                                nb_codes=nb_codes, nb_factors=nb_factors)
    
    eval_accuracy = np.sum(eval_votes[latent_idx, classifier]) * 1. / np.sum(eval_votes)
    
    # scale scores in [0, 1]
    if scale:
        # min value corresponds to a classifier that chooses at random
        min_val, max_val = 1. / nb_factors, 1.
        train_accuracy = (train_accuracy - min_val) / (max_val - min_val)
        eval_accuracy = (eval_accuracy - min_val) / (max_val - min_val)
    
    return eval_accuracy


def _compute_votes(inputs, targets, nb_codes, nb_factors):
    ''' Compute votes for Z-min-var metric
    
    :param inputs:          inputs of Z-min-var dataset
    :param targets:         targets of Z-min-var dataset
    :param nb_codes:        number of dimensions in latent space
    :param nb_factors:      number of generative factors
    '''
    # initialize votes array
    votes = np.zeros((nb_codes, nb_factors), dtype='int64')
    
    # fill array
    for line_idx in range(len(inputs)):
        argmin = inputs[line_idx]
        factor_id = targets[line_idx]
        votes[argmin, factor_id] += 1
    
    return votes


def _prune_dims(variances, threshold, verbose):
    ''' Mask for collapsed dimensions in the representation
    
    :param variances:      variance of each dimension in the representation
    :param threshold:      minimum accepted standard deviation
    :param verbose:        if True, print warnings
    '''
    stds = np.sqrt(variances)
    mask = stds >= threshold
    if not np.all(mask) and verbose:
        print(f'WARNING -- Collapsed latent dimensions detected -- mask = {mask}')
    return mask


def _compute_variances(representation, axis):
    ''' Compute variance for each dimension of the representation
    
    :param representation:      representation to compute variance from
    :param axis:                axis of variance
    '''
    # compute variances on the considered axis
    # ddof=1 for an unbiased estimator of the variance of a hypothetical infinite population
    variances = np.var(representation, axis=axis, ddof=1)
    return variances


def _generate_sample(representation, axis, variances, active_dims):
    ''' Create a single Z-min-var example based on a mini-batch of latent representations
    
    :param representation:          representation to compute variance from
    :param axis:                    axis of variance
    :param variances:               empirical variance of the latent codes
    :param active_dims:             active latent codes dimensions
    '''
    # compute local variances over the batch and find argmin
    local_variances = _compute_variances(representation, axis=axis)
    argmin = np.argmin(local_variances[active_dims] / variances[active_dims])
    return argmin


def _prepare_datasets(factors, codes, batch_size, nb_training, nb_eval, variances, active_dims, verbose):
    ''' prepare Z-min-var datasets from a factors-codes dataset
    
    :param factors:             dataset of factors in their discrete format
                                each column is a factor and each line is a data point
    :param codes:               latent codes associated to the dataset of factors
                                each column is a latent code and each line is a data point
    :param batch_size:          size of batch
    :param nb_training:         number of training points
    :param nb_eval:             number of evaluation points
    :param variances:           empirical variance of the latent codes
    :param active_dims:         active latent codes dimensions
    :param verbose:             if True, print warnings
    '''
    # initialize Z-min-var datasets for training and evaluation
    train_set, line_idx_train = np.zeros((nb_training, 2), dtype='int64'), 0
    eval_set, line_idx_eval = np.zeros((nb_eval, 2), dtype='int64'), 0
    
    # count the number of factors
    nb_factors = factors.shape[1]
    
    # Z-min-var metric is based on the fixing of a factor chosen randomly
    # for each data point, chose randomly a factor
    training_factors = np.random.randint(low=0, high=nb_factors, size=nb_training)
    unique, counts = np.unique(training_factors, return_counts=True)
    training_factors = dict(zip(unique, counts))
    
    eval_factors = np.random.randint(low=0, high=nb_factors, size=nb_eval)
    unique, counts = np.unique(eval_factors, return_counts=True)
    eval_factors = dict(zip(unique, counts))
    
    # iterate over factor IDs
    for factor_id in range(nb_factors):
        # make sure factor ID is in both dictionaries
        if not factor_id in training_factors:
            training_factors[factor_id] = 0
        if not factor_id in eval_factors:
            eval_factors[factor_id] = 0
        
        # total number of times the factor ID occurs in the Z-min-var dataset
        factor_id_count = training_factors[factor_id] + eval_factors[factor_id]
        
        # check which factor values are available in the dataset
        # keep only factor values that have at least 2 examples
        unique, counts = np.unique(factors[:, factor_id], return_counts=True)
        available_factor_values = dict(zip(unique, counts))
        available_factor_values = [value for value in available_factor_values
                                    if available_factor_values[value] > 1]
        
        if len(available_factor_values) == 0:
            print(f'Error -- Factor ID: {factor_id} -- Cannot find factor values with more than 1 example -- '
                    f'Discretization is too fine grained -- Decrease nb_bins -- Score is set to NaN')
            return NaN, NaN
        
        # for each batch of examples, Z-min-var fix the factor ID to a random discretized value
        # choose random values to fix for each batch of examples
        fixed_factor_values = np.random.choice(available_factor_values, size=factor_id_count)
        
        # keep track of factor values for each set
        train_factor_values = fixed_factor_values[:training_factors[factor_id]]
        eval_factor_values = fixed_factor_values[training_factors[factor_id]:]
        
        # transform variables to dictionaries
        unique, counts = np.unique(fixed_factor_values, return_counts=True)
        fixed_factor_values = dict(zip(unique, counts))
        unique, counts = np.unique(train_factor_values, return_counts=True)
        train_factor_values = dict(zip(unique, counts))
        unique, counts = np.unique(eval_factor_values, return_counts=True)
        eval_factor_values = dict(zip(unique, counts))
        
        # iterate over the fixed factor values
        for factor_value, count in fixed_factor_values.items(): 
            # make sure factor value is in dictionaries
            if not factor_value in train_factor_values:
                train_factor_values[factor_value] = 0
            if not factor_value in eval_factor_values:
                eval_factor_values[factor_value] = 0
            
            # infer the number of examples needed from the factors dataset
            nb_examples_needed = batch_size * count
            batch = np.zeros(nb_examples_needed, dtype='int64')
            
            # get lines idx in factors dataset, corresponding to factor ID at the specific factor value
            factor_value_lines_idx = np.where(factors[:, factor_id] == factor_value)[0]
            
            # check if we have enough examples in the factors dataset
            # number of times we need to pass over all the factor value examples
            nb_factor_value_examples = len(factor_value_lines_idx)
            nb_loops = int(np.ceil(nb_examples_needed / nb_factor_value_examples))
            
            # iterate over all factor value examples
            line_idx = 0
            current_count = 0
            for loop_id in range(nb_loops):
                # shuffle factor value lines idx
                np.random.shuffle(factor_value_lines_idx)
                
                # compute the number of examples to extract
                if loop_id + 1 == nb_loops:
                    nb_examples = nb_examples_needed - current_count
                else:
                    nb_examples = nb_factor_value_examples
                
                # fill the batch with examples
                assert(len(factor_value_lines_idx[:nb_examples]) == nb_examples)
                batch[line_idx: line_idx + nb_examples] = factor_value_lines_idx[:nb_examples]
                
                # increment variables
                current_count += nb_examples
                line_idx += nb_examples
            
            # check values are correct
            assert(current_count == line_idx == nb_examples_needed)
            
            # check everything is correct
            factors_batch = factors[batch]
            for id in range(nb_factors):
                if id == factor_id:
                    assert(np.min(factors_batch[:, id]) == np.max(factors_batch[:, id]) == factor_value)
                else:
                    if np.min(factors_batch[:, id]) == np.max(factors_batch[:, id]) and verbose:
                        print(f'Warning -- Factor ID: {id} -- factor values are equal '
                                f'whereas they should be different -- Try to decrease nb_bins')
            
            nb_factor_value_train = batch_size * train_factor_values[factor_value]
            nb_factor_value_eval = batch_size * eval_factor_values[factor_value]
            assert(nb_factor_value_train + nb_factor_value_eval == nb_examples_needed)
            
            # fill train set
            codes_train = codes[batch][:nb_factor_value_train]
            assert(codes_train.shape[0] % batch_size == 0)
            for idx in range(0, codes_train.shape[0], batch_size):
                    argmin = _generate_sample(codes_train[idx: idx + batch_size], axis=0,
                                              variances=variances, active_dims=active_dims)
                    train_set[line_idx_train, 0] = argmin
                    train_set[line_idx_train, 1] = factor_id
                    line_idx_train += 1
            
            # fill eval set
            codes_eval = codes[batch][nb_factor_value_train:]
            assert(codes_eval.shape[0] % batch_size == 0)
            for idx in range(0, codes_eval.shape[0], batch_size):
                    argmin = _generate_sample(codes_eval[idx: idx + batch_size], axis=0,
                                              variances=variances, active_dims=active_dims)
                    eval_set[line_idx_eval, 0] = argmin
                    eval_set[line_idx_eval, 1] = factor_id
                    line_idx_eval += 1
    
    # check values are corect
    assert(line_idx_train == nb_training)
    assert(line_idx_eval == nb_eval)
    
    # shuffle randomly datasets
    lines_idx = np.arange(train_set.shape[0])
    np.random.shuffle(lines_idx)
    train_set = train_set[lines_idx]
    
    lines_idx = np.arange(eval_set.shape[0])
    np.random.shuffle(lines_idx)
    eval_set = eval_set[lines_idx]

    return train_set, eval_set
