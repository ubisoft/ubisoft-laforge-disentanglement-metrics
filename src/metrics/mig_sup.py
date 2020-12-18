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

    
def mig_sup(factors, codes, continuous_factors=True, nb_bins=10):
    ''' MIG-SUP metric from Z. Li, J. V. Murkute, P. K. Gyawali, and L. Wang,
        “Progressive learning and disentanglement of hierarchicalrepresentations,”
        in ICLR, 2020.
    
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
            mi_matrix[f, c] = get_mutual_information(factors[:, f], codes[:, c])

    # compute the mean gap for all codes
    sum_gap = 0
    for c in range(nb_codes):
        mi_c = np.sort(mi_matrix[:, c])
        # get diff between highest and second highest term and add it to total gap
        sum_gap += mi_c[-1] - mi_c[-2]
    
    # compute the mean gap
    mig_sup_score = sum_gap / nb_codes
    
    return mig_sup_score
