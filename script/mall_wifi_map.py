# coding: utf-8

import csv
import os
import pickle

from config import Config
from collections import defaultdict
from mall_shop_map import MallShopMap

class MallWifiMap(object):
    def __init__(self, wifi_dir):
        self.pickle_file = os.path.join(wifi_dir, 'mall_wifi_map.pickle')
        if os.path.isfile(self.pickle_file):

            f = open(self.pickle_file, 'rb')
            self.wifi_map = pickle.load(f)
        else:
            self.wifi_map = self.__init(wifi_dir);

            with open(self.pickle_file, 'wb') as fout:
                pickle.dump(self.wifi_map, fout)
                fout.close()

        # self.wifi_num = len(self.wifi_map)


    def __init(self, wifi_dir):
        '''
        load the 'wifi' column of
        user_shop_behavior.csv & evaluation_public.csv
        to wifi_map
        :param wifi_dir: directory of wifi data
        :return: map<wifi_name, int_id>
        '''
        mall_shop_hashmap = MallShopMap(wifi_dir)
        user_shop_file = os.path.join(
            wifi_dir,
            # 'ccf_first_round_user_shop_behavior.csv'
            Config.user_shop_filename
        )
        wifi_map = defaultdict(dict)
        wifi_num = 0
        with open(user_shop_file, 'r') as fin:
            reader = csv.DictReader(fin)
            for line in reader:
                shop_id = line['shop_id']
                mall_id = mall_shop_hashmap.GetMallId(shop_id)
                for wifi_items in line['wifi_infos'].split(';'):
                    item = wifi_items.split('|')
                    assert len(item) == 3;
                    wifi_id = item[0]
                    if wifi_id not in wifi_map[mall_id]:
                        wifi_map[mall_id][wifi_id] = wifi_num
                        wifi_num += 1


        return wifi_map

    def GetIndex(self, mall_id, bssid):
        '''
        hash wifi_id to int index
        :param wifi_id: type str; like 'b_6396480'
        :return: int / -1 if bssid not in wifi_map
        '''
        if bssid in self.wifi_map[mall_id]:
            return self.wifi_map[mall_id][bssid]
        return -1

    def GetWifiInMall(self, mall_id):
        '''
        return the number of WIFI in the mall
        :param mall_id:
        :return: int
        '''
        assert mall_id in self.wifi_map
        return self.wifi_map[mall_id]