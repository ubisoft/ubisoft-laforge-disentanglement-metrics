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

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale

from utils import get_bin_index

    
def sap(factors, codes, continuous_factors=True, nb_bins=10, regression=True):
    ''' SAP metric from A. Kumar, P. Sattigeri, and A. Balakrishnan,
        “Variational inference of disentangled latent concepts from unlabeledobservations,”
        in ICLR, 2018.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param nb_bins:                         number of bins to use for discretization
    :param regression:                      True:   compute score using regression algorithms
                                            False:  compute score using classification algorithms
    '''
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # perform regression
    if regression:
        assert(continuous_factors), f'Cannot perform SAP regression with discrete factors.'
        return _sap_regression(factors, codes, nb_factors, nb_codes)  
    
    # perform classification
    else:
        # quantize factors if they are continuous
        if continuous_factors:
            factors = minmax_scale(factors)  # normalize in [0, 1] all columns
            factors = get_bin_index(factors, nb_bins)  # quantize values and get indexes
        
        # normalize in [0, 1] all columns
        codes = minmax_scale(codes)
        
        # compute score using classification algorithms
        return _sap_classification(factors, codes, nb_factors, nb_codes)


def _sap_regression(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using regression algorithms
    
    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    # compute R2 score matrix
    s_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            # train a linear regressor
            regr = LinearRegression()

            # train the model using the training sets
            regr.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))

            # make predictions using the testing set
            y_pred = regr.predict(codes[:, c].reshape(-1, 1))

            # compute R2 score
            r2 = r2_score(factors[:, f], y_pred)
            s_matrix[f, c] = max(0, r2) 

    # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        # get diff between highest and second highest term and add it to total gap
        s_f = np.sort(s_matrix[f, :])
        sum_gap += s_f[-1] - s_f[-2]
    
    # compute the mean gap
    sap_score = sum_gap / nb_factors
    
    return sap_score


def _sap_classification(factors, codes, nb_factors, nb_codes):
    ''' Compute SAP score using classification algorithms
    
    :param factors:         factors dataset
    :param codes:           latent codes dataset
    :param nb_factors:      number of factors in the factors dataset
    :param nb_codes:        number of codes in the latent codes dataset
    '''
    # compute accuracy matrix
    s_matrix = np.zeros((nb_factors, nb_codes))
    for f in range(nb_factors):
        for c in range(nb_codes):
            # find the optimal number of splits
            best_score, best_sp = 0, 0
            for sp in range(1, 10):
                # perform cross validation on the tree classifiers
                clf = tree.DecisionTreeClassifier(max_depth=sp)
                scores = cross_val_score(clf, codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1), cv=10)
                scores = scores.mean()
                
                if scores > best_score:
                    best_score = scores
                    best_sp = sp
            
            # train the model using the best performing parameter
            clf = tree.DecisionTreeClassifier(max_depth=best_sp)
            clf.fit(codes[:, c].reshape(-1, 1), factors[:, f].reshape(-1, 1))

            # make predictions using the testing set
            y_pred = clf.predict(codes[:, c].reshape(-1, 1))

            # compute accuracy
            s_matrix[f, c] = accuracy_score(y_pred, factors[:, f])

    # compute the mean gap for all factors
    sum_gap = 0
    for f in range(nb_factors):
        # get diff between highest and second highest term and add it to total gap
        s_f = np.sort(s_matrix[f, :])
        sum_gap += s_f[-1] - s_f[-2]
    
    # compute the mean gap
    sap_score = sum_gap / nb_factors
    
    return sap_score
