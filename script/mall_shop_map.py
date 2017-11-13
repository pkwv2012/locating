# coding: utf-8

import csv
import pickle
import os

from collections import defaultdict
from .logger import LOGGER

class MallShopMap(object):
    def __init__(self, data_dir):
        self.mall_list = defaultdict(list)
        self.mall_shop_map = defaultdict(dict)
        pickle_file = os.path.join(
            data_dir,
            'mall_shop_info.pickle'
        )
        if os.path.isfile(pickle_file):
            with open(pickle_file) as fin:
                self.mall_list = pickle.load(fin)
                self.mall_shop_map = pickle.load(fin)
        else:
            self.__init(data_dir);
            with open(pickle_file, 'wb') as fout:
                pickle.dump(self.mall_list, fout)
                pickle.dump(self.mall_shop_map, fout)

    def __init(self, data_dir):
        '''
        load data to `mall_list` `mall_shop_map`
        :param data_dir:
        :return: void
        '''
        shop_info_file = os.path.join(data_dir, 'ccf_first_round_shop_info.csv')
        with open(shop_info_file, 'r') as fin:
            reader = csv.DictReader(fin)
            for line in reader:
                mall_id = line['mall_id']
                shop_id = line['shop_id']
                if shop_id not in self.mall_shop_map[mall_id]:
                    self.mall_shop_map[mall_id][shop_id] = len(self.mall_list[mall_id])
                    self.mall_list[mall_id].append(shop_id)
        LOGGER.info('shop info loaded')
