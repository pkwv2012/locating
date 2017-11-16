# coding: utf-8

import os

from logger import LOGGER

class Config(object):
    # data_dir = '/media/burk/000F0F2400091F8B/postgraduate/ML_APP/locating/data'
    data_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../data'
    )
    LOGGER.info(data_dir)
    user_shop_filename = 'ccf_first_round_user_shop_behavior.csv'
    shop_info_filename = 'ccf_first_round_shop_info.csv'
    evaluation_filename = 'evaluation_public.csv'

    train_filename = 'train.csv'
    validation_filename = 'validation.csv'

    # train or predict
    is_train = True
    # if predict
    selected_model_suffix = '2017_11_13_18_26'
  
    bad_accuracy_mall_list = ['m_7800', 'm_4187', 'm_4079']

    # set parameters for xgboost
    XGB_param = {
        # use softmax multi-class classification
        'objective' : 'multi:softmax',
        # scale weight of positive examples
        'eta' : 0.1,
        'max_depth': 3,
        'silent': 1,
        'min_child_weight': 0.5,
        # subsample make accuracy decrese 0.07
        # param['subsample'] = 0.6
        'lambda': 0.5,
        # param['nthread'] = 2
    }

    # set parameters for LightGBM
    LGB_param = {
        'application': 'multiclass', # 'multiclassova'
        'boosting': 'gbdt', # 'dart'
        'num_iterations': 300,
        'learning_rate': 0.1,
        'max_depth': -1, # no limit?
        'min_data_in_leaf': 10,
        'metric': 'multi_error',
    }
