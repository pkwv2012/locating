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

    # train or predict
    is_train = False
    # if predict
    selected_model_suffix = '2017_11_13_18_26'
