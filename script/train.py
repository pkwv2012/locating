# coding : utf-8

import pickle
import os
import sys

from scipy.sparse import csr_matrix

from .config import Config
from .logger import LOGGER
from .mall_shop_map import MallShopMap
from .wifi_map import WifiMap



if __name__ == '__main__':
    data_dir = Config.data_dir
    wifi_hashmap = WifiMap(data_dir)
    mall_shop_hashmap = MallShopMap(data_dir)



