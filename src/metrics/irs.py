# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

from sklearn.preprocessing import minmax_scale

from utils import get_bin_index


def irs(factors, codes, continuous_factors=True, nb_bins=10, diff_quantile=1.):
    ''' IRS metric from R. Suter, D. Miladinovic, B. Schölkopf, and S. Bauer,
        “Robustly disentangled causal mechanisms: Validatingdeep representations for interventional robustness,”
        in ICML, 2019.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    :param diff_quantile:                   float value between 0 and 1 to decide what quantile of diffs to select
                                            use 1.0 for the version in the paper
    '''
    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
    
    # remove constant dimensions
    codes = _drop_constant_dims(codes)
    
    if not codes.any():
        irs_score = 0.0
    else:
        # count the number of factors and latent codes
        nb_factors = factors.shape[1]
        nb_codes = codes.shape[1]
        
        # compute normalizer
        max_deviations = np.max(np.abs(codes - codes.mean(axis=0)), axis=0)
        cum_deviations = np.zeros([nb_codes, nb_factors])
        for i in range(nb_factors):
            unique_factors = np.unique(factors[:, i], axis=0)
            assert(unique_factors.ndim == 1)
            nb_distinct_factors = unique_factors.shape[0]
            
            for k in range(nb_distinct_factors):
                # compute E[Z | g_i]
                match = factors[:, i] == unique_factors[k]
                e_loc = np.mean(codes[match, :], axis=0)

                # difference of each value within that group of constant g_i to its mean
                diffs = np.abs(codes[match, :] - e_loc)
                max_diffs = np.percentile(diffs, q=diff_quantile*100, axis=0)
                cum_deviations[:, i] += max_diffs
            
            cum_deviations[:, i] /= nb_distinct_factors
        
        # normalize value of each latent dimension with its maximal deviation
        normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
        irs_matrix = 1.0 - normalized_deviations
        disentanglement_scores = irs_matrix.max(axis=1)
        
        if np.sum(max_deviations) > 0.0:
            irs_score = np.average(disentanglement_scores, weights=max_deviations)
        else:
            irs_score = np.mean(disentanglement_scores)
    
    return irs_score


def _drop_constant_dims(codes):
    ''' Drop constant dimensions of latent codes
    
    :param codes:       latent codes associated to the dataset of factors
                        each column is a latent code and each line is a data point
    '''
    # check we have a matrix
    if codes.ndim != 2:
        raise ValueError("Expecting a matrix.")

    # compute variances and create mask
    variances = codes.var(axis=0)
    mask = variances > 0.
    if not np.all(mask):
        print(f'WARNING -- Collapsed latent dimensions detected -- mask = {mask}')
    
    return codes[:, mask]
