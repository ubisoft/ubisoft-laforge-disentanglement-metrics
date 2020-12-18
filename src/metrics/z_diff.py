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
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale

from utils import get_bin_index

    
def z_diff(factors, codes, continuous_factors=True, nb_bins=10, batch_size=200,
           nb_training=10000, nb_eval=5000, nb_max_iterations=10000, scale=True):
    ''' Z-diff metric from I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner,
        “β-VAE:Learning basic visual concepts with a constrained variational framework,”
        in ICLR, 2017.
    
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
    :param nb_max_iterations:               number of training iterations for the linear model
    :param scale:                           if True, the output will be scaled from 0 to 1
    '''
    # count the number of factors
    nb_factors = factors.shape[1]
    
    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
    
    # prepare Z-diff datasets for training and evaluation
    train_set, eval_set = _prepare_datasets(factors=factors, codes=codes, batch_size=batch_size,
                                            nb_training=nb_training, nb_eval=nb_eval)
    
    # discretization is too fine grained -- score cannot be computed correctly
    if train_set is NaN and eval_set is NaN:
        return NaN
    
    # train model
    inputs, targets = train_set
    model = linear_model.LogisticRegression(max_iter=nb_max_iterations)
    model.fit(inputs, targets)
    
    # compute training accuracy
    train_accuracy = model.score(inputs, targets)
    
    # compute evaluation accuracy
    inputs, targets = eval_set
    eval_accuracy = model.score(inputs, targets)
    
    # scale scores in [0, 1]
    if scale:
        # min value corresponds to a classifier that chooses at random
        min_val, max_val = 1. / nb_factors, 1.
        train_accuracy = (train_accuracy - min_val) / (max_val - min_val)
        eval_accuracy = (eval_accuracy - min_val) / (max_val - min_val)
    
    return eval_accuracy


def _prepare_datasets(factors, codes, batch_size, nb_training, nb_eval):
    ''' prepare Z-diff datasets from a factors-codes dataset
    
    :param factors:             dataset of factors in their discrete format
                                each column is a factor and each line is a data point
    :param codes:               latent codes associated to the dataset of factors
                                each column is a latent code and each line is a data point
    :param batch_size:          size of batch
    :param nb_training:         number of training points
    :param nb_eval:             number of evaluation points
    '''
    # count the number of factors and codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # initialize Z-diff datasets for training and evaluation
    train_inputs, train_targets, line_idx_train = \
        np.zeros((nb_training, nb_codes)), np.zeros((nb_training, ), dtype='int64'), 0
    eval_inputs, eval_targets, line_idx_eval = \
        np.zeros((nb_eval, nb_codes)), np.zeros((nb_eval, ), dtype='int64'), 0
    
    # Z-diff metric is based on the fixing of a factor chosen randomly
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
        
        # total number of times the factor ID occurs in the Z-diff dataset
        factor_id_count = training_factors[factor_id] + eval_factors[factor_id]
        
        # infer the number of examples needed from the factors dataset
        nb_factor_id_examples = batch_size * factor_id_count
        batch_1 = np.zeros(nb_factor_id_examples, dtype='int64')
        batch_2 = np.zeros(nb_factor_id_examples, dtype='int64')
        
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
        
        # for each pair of example in the batch, beta-VAE fix the factor ID to a random discretized value
        # choose random values to fix for each example
        fixed_factor_values = np.random.choice(available_factor_values, size=nb_factor_id_examples)

        # transform variables to dictionaries
        unique, counts = np.unique(fixed_factor_values, return_counts=True)
        fixed_factor_values = dict(zip(unique, counts))
        
        # iterate over the fixed factor values
        line_idx = 0
        for factor_value, count in fixed_factor_values.items(): 
            # get lines idx in factors dataset, corresponding to factor ID at the specific factor value
            factor_value_lines_idx = np.where(factors[:, factor_id] == factor_value)[0]
            
            # check if we have enough examples in the factors dataset
            # number of times we need to pass over all the factor value examples
            nb_factor_value_examples = len(factor_value_lines_idx)
            nb_loops = int(np.ceil(count / (nb_factor_value_examples // 2)))
            
            # iterate over all factor value examples
            current_count = 0
            for loop_id in range(nb_loops):
                # shuffle factor value lines idx
                np.random.shuffle(factor_value_lines_idx)
                
                # compute the number of examples to extract
                if loop_id + 1 == nb_loops:
                    nb_examples = count - current_count
                else:
                    nb_examples = nb_factor_value_examples // 2
                
                # fill the batch pair with examples
                assert(len(factor_value_lines_idx[:nb_examples]) == len(factor_value_lines_idx[nb_examples: 2 * nb_examples]))
                batch_1[line_idx: line_idx + nb_examples] = factor_value_lines_idx[:nb_examples]
                batch_2[line_idx: line_idx + nb_examples] = factor_value_lines_idx[nb_examples: 2 * nb_examples]
                
                # increment variables
                current_count += nb_examples
                line_idx += nb_examples
            
            # check value is correct
            assert(current_count == count)
        
        # check value is correct
        assert(line_idx == nb_factor_id_examples)
        
        # shuffle batch pair
        batch_lines_idx = np.arange(nb_factor_id_examples)
        np.random.shuffle(batch_lines_idx)
        batch_1 = batch_1[batch_lines_idx]
        batch_2 = batch_2[batch_lines_idx]
        
        # check we don't use the same line index in a pair
        assert(np.all(batch_1 - batch_2))
        
        # check everything is correct
        factors_batch_1 = factors[batch_1]
        factors_batch_2 = factors[batch_2]
        for id in range(nb_factors):
            if id == factor_id:
                assert(np.array_equal(factors_batch_1[:, id], factors_batch_2[:, id]))
            else:
                if np.array_equal(factors_batch_1[:, id], factors_batch_2[:, id]):
                    print(f'Warning -- Factor ID: {id} -- factor values are equal '
                            f'whereas they should be different -- Try to decrease nb_bins')
        
        nb_factor_id_train = batch_size * training_factors[factor_id]
        nb_factor_id_eval = batch_size * eval_factors[factor_id]
        assert(nb_factor_id_train + nb_factor_id_eval == nb_factor_id_examples)
        
        # fill train set
        codes_train_1 = codes[batch_1][:nb_factor_id_train]
        codes_train_2 = codes[batch_2][:nb_factor_id_train]
        assert(codes_train_1.shape[0] % batch_size == 0)
        assert(codes_train_2.shape[0] % batch_size == 0)
        for idx in range(0, codes_train_1.shape[0], batch_size):
            diff = codes_train_1[idx: idx + batch_size] - codes_train_2[idx: idx + batch_size]
            input = np.mean(np.abs(diff), axis=0)
            train_inputs[line_idx_train] = input
            train_targets[line_idx_train] = factor_id
            line_idx_train += 1
        
        # fill eval set
        codes_eval_1 = codes[batch_1][nb_factor_id_train:]
        codes_eval_2 = codes[batch_2][nb_factor_id_train:]
        assert(codes_eval_1.shape[0] % batch_size == 0)
        assert(codes_eval_2.shape[0] % batch_size == 0)
        for idx in range(0, codes_eval_1.shape[0], batch_size):
            diff = codes_eval_1[idx: idx + batch_size] - codes_eval_2[idx: idx + batch_size]
            input = np.mean(np.abs(diff), axis=0)
            eval_inputs[line_idx_eval] = input
            eval_targets[line_idx_eval] = factor_id
            line_idx_eval += 1
    
    # check values are corect
    assert(line_idx_train == nb_training)
    assert(line_idx_eval == nb_eval)
    
    # shuffle randomly datasets
    lines_idx = np.arange(train_inputs.shape[0])
    np.random.shuffle(lines_idx)
    train_inputs = train_inputs[lines_idx]
    train_targets = train_targets[lines_idx]
    
    lines_idx = np.arange(eval_inputs.shape[0])
    np.random.shuffle(lines_idx)
    eval_inputs = eval_inputs[lines_idx]
    eval_targets = eval_targets[lines_idx]
    
    # gather inputs and targets
    train_set = (train_inputs, train_targets)
    eval_set = (eval_inputs, eval_targets)

    return train_set, eval_set
