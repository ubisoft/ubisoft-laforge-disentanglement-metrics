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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale, MultiLabelBinarizer

from utils import get_bin_index


def explicitness(factors, codes, continuous_factors=True, nb_bins=10, scale=True, impl=1):
    ''' Explicitness metrics from K. Ridgeway and M. C. Mozer,
        “Learning deep disentangled embeddings with the f-statistic loss,”
        in NeurIPS, 2018.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    :param scale:                           if True, the output will be scaled from 0 to 1 instead of 0.5 to 1
    :param impl:                            implementation to use for explicitness score computation
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # quantize factors if they are continuous
    if continuous_factors:
        factors = minmax_scale(factors)  # normalize in [0, 1] all columns
        factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
    
    # normalize in [0, 1] all columns
    codes = minmax_scale(codes)
    
    # compute score using one of the 2 implementations
    if impl == 1:
        return _implementation_1(factors, codes, nb_factors, scale)
    elif impl == 2:
        return _implementation_2(factors, codes, nb_factors, scale)
    else:
        raise ValueError(f'ERROR -- argument "impl" is {impl} but must be either 1 or 2')


def _implementation_1(factors, codes, nb_factors, scale):
    ''' First implementation of explicitness score
    
    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param scale:           if True, the output will be scaled from 0 to 1 instead of 0.5 to 1
    '''
    # get AUC-ROC for each class for each factor
    cum_auc = 0
    for f in range(nb_factors):
        # get binary vector for all classes
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(np.expand_dims(factors[:, f], 1))
        
        # train a classifier on class c for factor f
        clf = LogisticRegression(class_weight='balanced', multi_class='ovr')
        clf.fit(codes, factors[:, f])

        # obtain predictions from the codes
        y_pred_test_p = clf.predict_proba(codes)

        # compute AUC        
        cum_auc += roc_auc_score(labels, y_pred_test_p)

    # compute the score
    explicitness_score = cum_auc / nb_factors
    if scale:
        explicitness_score = (explicitness_score - 0.5) * 2
    
    return explicitness_score


def _implementation_2(factors, codes, nb_factors, scale):
    ''' Second implementation of explicitness score
    
    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param scale:           if True, the output will be scaled from 0 to 1 instead of 0.5 to 1
    '''
    # get AUC-ROC for each class for each factor
    cum_auc = 0
    for f in range(nb_factors):
        cum_auc_fact = 0
        
        # get binary vector for all classes
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(np.expand_dims(factors[:, f], 1))
        
        for c, class_name in enumerate(mlb.classes_):
            # train a classifier on class c for factor f
            clf = LogisticRegression(class_weight='balanced')
            clf.fit(codes, labels[:, c])

            # obtain predictions from the codes
            y_pred_test_p = clf.predict_proba(codes)

            # compute the AUC ROC 
            cum_auc_fact += roc_auc_score(y_true=labels[:, c], y_score=y_pred_test_p[:, 1])
            
        # get mean AUC for this factor
        cum_auc += cum_auc_fact / len(mlb.classes_)

    explicitness_score = cum_auc / nb_factors
    if scale:
        explicitness_score = (explicitness_score - 0.5) * 2
    
    return explicitness_score
