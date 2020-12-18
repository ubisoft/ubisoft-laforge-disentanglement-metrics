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
import os
import sys

# set PYTHONPATH
FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
SRC_ROOT = os.path.realpath(os.path.dirname(FILE_ROOT))
sys.path.append(SRC_ROOT)

from metrics.dci import dci
from metrics.dcimig import dcimig
from metrics.explicitness import explicitness
from metrics.irs import irs
from metrics.jemmig import jemmig
from metrics.mig import mig
from metrics.mig_sup import mig_sup
from metrics.modularity import modularity
from metrics.sap import sap
from metrics.z_diff import z_diff
from metrics.z_max_var import z_max_var
from metrics.z_min_var import z_min_var


'''
    Default values to use for all experiments
'''


# metrics to use and their associated default hyper-parameters
METRICS = {
    'MIG-RMIG': {
        'function': mig,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10
            }
        },
    'DCIMIG': {
        'function': dcimig,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10
            }
        },
    'JEMMIG': {
        'function': jemmig,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10
            }
        },
    'MIG-sup': {
        'function': mig_sup,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10
            }
        },
    'Modularity Score': {
        'function': modularity,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10
            }
        },
    'Explicitness Score': {
        'function': explicitness,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10,
            'scale': True,
            'impl': 1
            }
        },
    'SAP': {
        'function': sap,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10,
            'regression': True
            }
        },
    'IRS': {
        'function': irs,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10,
            'diff_quantile': 1.0
            }
        },
    'Z-min Variance': {
        'function': z_min_var,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10,
            'batch_size': 200,
            'nb_training': 800,
            'nb_eval': 800,
            'nb_variance_estimate': 10000,
            'std_threshold': 0.02,
            'scale': True
            }
        },
    'Z-max Variance': {
        'function': z_max_var,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10,
            'batch_size': 200,
            'nb_training': 800,
            'nb_eval': 800,
            'nb_variance_estimate': 10000,
            'std_threshold': 0.02,
            'scale': True
            }
        },
    'Z-diff': {
        'function': z_diff,
        'kwargs': {
            'continuous_factors': True,
            'nb_bins': 10,
            'batch_size': 200,
            'nb_training': 10000,
            'nb_eval': 5000,
            'nb_max_iterations': 10000,
            'scale': True
            }
        },
    'DCI Lasso': {
        'function': dci,
        'kwargs': {
            'continuous_factors': True,
            'model': 'lasso',
            }
        },
    'DCI RF': {
        'function': dci,
        'kwargs': {
            'continuous_factors': True,
            'model': 'random_forest',
            }
        }
    }


# config parameters for plots
loosely_dashed = (0, (5, 10))
densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))
densely_dotted = (0, (1, 1))
PLOTS = {
    'FAMILIES': {
        'Intervention-based': ['Z-diff', 'Z-min Variance', 'Z-max Variance', 'IRS'],
        'Predictor-based': ['DCI Lasso Mod', 'DCI Lasso Comp', 'DCI Lasso Expl',
                            'DCI RF Mod', 'DCI RF Comp', 'DCI RF Expl',
                            'Explicitness Score', 'SAP'],
        'Information-based': ['MIG-RMIG', 'MIG-sup', 'JEMMIG', 'Modularity Score', 'DCIMIG']
    },
    'COLORS': ['blue', 'green', 'red', 'darkturquoise', 'magenta', 'orange', 'black'],
    'LINE STYLES': ['--', '-.', loosely_dashed, densely_dashdotdotted, ':', densely_dotted],
    'LEGEND POSITIONS': [(0.5, -0.295), (0.5, -0.35), (0.5, -0.295)],
    'NB LEGEND COLUMNS': [2, 3, 3]
}
