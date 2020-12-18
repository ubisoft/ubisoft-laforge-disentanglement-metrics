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

from sklearn.preprocessing import minmax_scale

from utils import get_bin_index, get_mutual_information

    
def modularity(factors, codes, continuous_factors=True, nb_bins=10):
    ''' Modularity metric from K. Ridgeway and M. C. Mozer,
        “Learning deep disentangled embeddings with the f-statistic loss,”
        in NeurIPS, 2018.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    ''' 
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
    
    # quantize latent codes
    codes = minmax_scale(codes)  # normalize in [0, 1] all columns
    codes = get_bin_index(codes, nb_bins)  # quantize values and get indexes
    
    # compute mutual information matrix
    mi_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            mi_matrix[f, c] = get_mutual_information(factors[:, f], codes[:, c], normalize=False)

    # compute the score for all codes
    sum_score = 0
    for c in range(nb_codes):
        # find the index of the factor with the maximum MI
        max_mi_idx = np.argmax(mi_matrix[:, c])

        # compute numerator
        numerator = 0
        for f, mi_f in enumerate(mi_matrix[:, c]):
            if f != max_mi_idx:
                numerator += mi_f ** 2
        
        # get the score for this code
        s = 1 - numerator / (mi_matrix[max_mi_idx, c] ** 2 * (nb_factors - 1))
        sum_score += s
    
    # compute the mean gap
    modularity_score = sum_score / nb_codes
    
    return modularity_score
