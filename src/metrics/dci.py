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
import math

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import minmax_scale


def dci(factors, codes, continuous_factors=True, model='lasso'):
    ''' DCI metrics from C. Eastwood and C. K. I. Williams,
        “A framework for the quantitative evaluation of disentangled representations,”
        in ICLR, 2018.
    
    :param factors:                         dataset of factors
                                            each column is a factor and each line is a data point
    :param codes:                           latent codes associated to the dataset of factors
                                            each column is a latent code and each line is a data point
    :param continuous_factors:              True:   factors are described as continuous variables
                                            False:  factors are described as discrete variables
    :param model:                           model to use for score computation
                                            either lasso or random_forest
    '''
    # TODO: Support for discrete data
    assert(continuous_factors), f'Only continuous factors are supported'
        
    # count the number of factors and latent codes
    nb_factors = factors.shape[1]
    nb_codes = codes.shape[1]
    
    # normalize in [0, 1] all columns
    factors = minmax_scale(factors)
    codes = minmax_scale(codes)
    
    # compute entropy matrix and informativeness per factor
    e_matrix = np.zeros((nb_factors, nb_codes))
    informativeness = np.zeros((nb_factors, ))
    for f in range(nb_factors):
        if model == 'lasso':
            informativeness[f], weights = _fit_lasso(factors[:, f].reshape(-1, 1), codes)
            e_matrix[f, :] = weights
        elif model == 'random_forest':
            informativeness[f], weights = _fit_random_forest(factors[:, f].reshape(-1, 1), codes)
            e_matrix[f, :] = weights
        else:
            raise ValueError("Regressor must be lasso or random_forest")
    
    # compute disentanglement per code
    rho = np.zeros((nb_codes, ))
    disentanglement = np.zeros((nb_codes, ))
    for c in range(nb_codes):
        # get importance weight for code c
        rho[c] = np.sum(e_matrix[:, c])
        if rho[c] == 0:
            disentanglement[c] = 0
            break
        
        # transform weights in probabilities
        prob = e_matrix[:, c] / rho[c]
        
        # compute entropy for code c
        H = 0
        for p in prob:
            if p:
                H -= p * math.log(p, len(prob))
        
        # get disentanglement score
        disentanglement[c] = 1 - H
    
    # compute final disentanglement 
    if np.sum(rho):
        rho = rho / np.sum(rho)
    else:
        rho = rho * 0
    
    # compute completeness
    completeness = np.zeros((nb_factors, ))
    for f in range(nb_factors):
        if np.sum(e_matrix[f, :]) != 0:
            prob = e_matrix[f, :] / np.sum(e_matrix[f, :])
        else:
            prob = np.ones((len(e_matrix[f, :]), 1)) / len(e_matrix[f, :]) 
        
        # compute entropy for code c
        H = 0
        for p in prob:
            if p:
                H -= p * math.log(p, len(prob))
        
        # get disentanglement score
        completeness[f] = 1 - H
    
    # average all results
    disentanglement = np.dot(disentanglement, rho)
    completeness = np.mean(completeness)
    informativeness = np.mean(informativeness)
    
    return disentanglement, completeness, informativeness


def _fit_lasso(factors, codes):
    ''' Fit a Lasso regressor on the data
    
    :param factors:         factors dataset
    :param codes:           latent codes dataset
    '''
    # alpha values to try
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1]

    # make sure factors are N by 1
    factors.reshape(-1, 1)
    
    # find the optimal alpha regularization parameter
    best_a = 0
    best_mse = 10e10
    for a in alphas:
        # perform cross validation on the tree classifiers
        clf = Lasso(alpha=a, max_iter=5000)
        mse = cross_val_score(clf, codes, factors, cv=10, scoring='neg_mean_squared_error')
        mse = -mse.mean()
        
        if mse < best_mse:
            best_mse = mse
            best_a = a
            
    # train the model using the best performing parameter
    clf = Lasso(alpha=best_a)
    clf.fit(codes, factors)

    # make predictions using the testing set
    y_pred = clf.predict(codes)

    # compute informativeness from prediction error (max value for mse/2 = 1/12) 
    mse = mean_squared_error(y_pred, factors)    
    informativeness = max(1 - 12 * mse, 0)
    
    # get the weight from the regressor
    predictor_weights = np.ravel(np.abs(clf.coef_))

    return informativeness, predictor_weights


def _fit_random_forest(factors, codes):
    ''' Fit a Random Forest regressor on the data
    
    :param factors:         factors dataset
    :param codes:           latent codes dataset
    '''
    # alpha values to try
    max_depth = [8, 16, 32, 64, 128]
    max_features = [0.2, 0.4, 0.8, 1.0]
    
    # make sure factors are N by 0
    factors = np.ravel(factors)
    
    # find the optimal alpha regularization parameter
    best_mse = 10e10
    best_mf = 0
    best_md = 0
    for md in max_depth:
        for mf in max_features:
            # perform cross validation on the tree classifiers
            clf = RandomForestRegressor(n_estimators=10, max_depth=md, max_features=mf)
            mse = cross_val_score(clf, codes, factors, cv=10, scoring='neg_mean_squared_error')
            mse = -mse.mean()
            
            if mse < best_mse:
                best_mse = mse
                best_mf = mf
                best_md = md
            
    # train the model using the best performing parameter
    clf = RandomForestRegressor(n_estimators=10, max_depth=best_md, max_features=best_mf)
    clf.fit(codes, factors)

    # make predictions using the testing set
    y_pred = clf.predict(codes)

    # compute informativeness from prediction error (max value for mse/2 = 1/12) 
    mse = mean_squared_error(y_pred, factors)    
    informativeness = max(1 - 12 * mse, 0)
    
    # get the weight from the regressor
    predictor_weights = clf.feature_importances_

    return informativeness, predictor_weights
