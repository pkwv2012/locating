# coding: utf-8

import logging
import os

from datetime import datetime


now = datetime.now()
LOGGER = logging.getLogger('locating_monitor')
LOGGER.setLevel(logging.INFO)
# print(os.path.realpath(__file__))
reuters_handler = logging.FileHandler(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../logs_{}'.format(now.strftime("%Y_%m_%d_%H_%M_%S"))
))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
reuters_handler.setFormatter(formatter)
LOGGER.addHandler(reuters_handler)