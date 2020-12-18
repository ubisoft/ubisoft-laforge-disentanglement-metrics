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

from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import minmax_scale

from utils import get_bin_index, get_mutual_information


def dcimig(factors, codes, continuous_factors=True, nb_bins=10):
    ''' DCIMIG metric from A. Sepliarskaia, J. Kiseleva, and M. de Rijke,
        “Evaluating disentangled representations,”
        arXiv:1910.05587, 2020.
    
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

    # compute the gap for all codes
    for c in range(nb_codes):
        mi_c = np.sort(mi_matrix[:, c])
        max_idx = np.argmax(mi_matrix[:, c])

        # get diff between highest and second highest term gap
        gap = mi_c[-1] - mi_c[-2]

        # replace the best by the gap and the rest by 0
        mi_matrix[:, c] = mi_matrix[:, c] * 0
        mi_matrix[max_idx, c] = gap

    # find the best gap for each factor
    gap_sum = 0
    for f in range(nb_factors):
        gap_sum += np.max(mi_matrix[f, :])

    # sum the entropy for each factors
    factor_entropy = 0
    for f in range(nb_factors):
        factor_entropy += drv.entropy(factors[:, f])

    # compute the mean gap
    dcimig_score = gap_sum / factor_entropy
    
    return dcimig_score
