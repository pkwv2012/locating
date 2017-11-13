# coding: utf-8

import csv
import pickle
import os

from collections import defaultdict
from config import Config
from logger import LOGGER

class MallShopMap(object):
    def __init__(self, data_dir):
        self.mall_list = defaultdict(list)
        self.mall_shop_map = defaultdict(dict)
        self.shop_mall_map = defaultdict(int)
        pickle_file = os.path.join(
            data_dir,
            'mall_shop_info.pickle'
        )
        if os.path.isfile(pickle_file):
            with open(pickle_file, 'rb') as fin:
                self.mall_list = pickle.load(fin)
                self.mall_shop_map = pickle.load(fin)
                self.shop_mall_map = pickle.load(fin)
        else:
            self.__init(data_dir);
            with open(pickle_file, 'wb') as fout:
                pickle.dump(self.mall_list, fout)
                pickle.dump(self.mall_shop_map, fout)
                pickle.dump(self.shop_mall_map, fout)

    def __init(self, data_dir):
        '''
        load data to `mall_list` `mall_shop_map`
        :param data_dir:
        :return: void
        '''
        shop_info_file = os.path.join(
            data_dir,
            # 'ccf_first_round_shop_info.csv'
            Config.shop_info_filename
        )
        with open(shop_info_file, 'r') as fin:
            reader = csv.DictReader(fin)
            for line in reader:
                mall_id = line['mall_id']
                shop_id = line['shop_id']
                if shop_id not in self.mall_shop_map[mall_id]:
                    self.mall_shop_map[mall_id][shop_id] = len(self.mall_list[mall_id])
                    self.mall_list[mall_id].append(shop_id)
                if shop_id not in self.shop_mall_map:
                    self.shop_mall_map[shop_id] = mall_id
        LOGGER.info('shop info loaded')

    def GetShopIndex(self, mall_id, shop_id):
        '''
        get the shop index of the mall
        :param mall_id:
        :param shop_id:
        :return: int
        '''
        assert mall_id in self.mall_shop_map
        assert shop_id in self.mall_shop_map[mall_id]
        return self.mall_shop_map[mall_id][shop_id]

    def GetShopId(self, mall_id, index):
        '''
        get the shop_id
        :param mall_id:
        :param index: int
        :return:
        '''
        assert mall_id in self.mall_shop_map
        assert index < len(self.mall_list[mall_id])
        return self.mall_list[mall_id][index]

    def GetMallId(self, shop_id):
        '''
        return the mall which the shop is located
        :param shop_id:
        :return: mall_id
        '''
        assert shop_id in self.shop_mall_map
        return self.shop_mall_map[shop_id]

    def GetShopNumInMall(self, mall_id):
        '''
        return # of shops in the mall
        :param mall_id:
        :return:
        '''
        return len(self.mall_shop_map[mall_id])