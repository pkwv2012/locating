# coding: utf-8

import csv
import os
import pickle

from config import Config

class WifiMap(object):
    def __init__(self, wifi_dir):
        self.pickle_file = os.path.join(wifi_dir, 'wifi_map.pickle')
        if os.path.isfile(self.pickle_file):

            f = open(self.pickle_file, 'rb')
            self.wifi_map = pickle.load(f)
        else:
            self.wifi_map = self.__init(wifi_dir);

            with open(self.pickle_file, 'wb') as fout:
                pickle.dump(self.wifi_map, fout)
                fout.close()

        self.wifi_num = len(self.wifi_map)


    def __init(self, wifi_dir):
        '''
        load the 'wifi' column of
        user_shop_behavior.csv & evaluation_public.csv
        to wifi_map
        :param wifi_dir: directory of wifi data
        :return: map<wifi_name, int_id>
        '''
        user_shop_file = os.path.join(
            wifi_dir,
            # 'ccf_first_round_user_shop_behavior.csv'
            Config.user_shop_filename
        )
        wifi_map = {}
        wifi_num = 0
        with open(user_shop_file) as fin:
            reader = csv.DictReader(fin)
            for line in reader:
                for wifi_items in line['wifi_infos'].split(';'):
                    item = wifi_items.split('|')
                    assert len(item) == 3;
                    wifi_id = item[0]
                    if wifi_id not in wifi_map:
                        wifi_map[wifi_id] = wifi_num
                        wifi_num += 1

        evaluation_file = os.path.join(
            wifi_dir,
            # 'evaluation_public.csv'
            Config.evaluation_filename
        )
        with open(evaluation_file) as fin:
            reader = csv.DictReader(fin)
            for line in reader:
                for wifi_items in line['wifi_infos'].split(';'):
                    item = wifi_items.split('|')
                    assert len(item) == 3
                    wifi_id = item[0]
                    if wifi_id not in wifi_map:
                        wifi_map[wifi_id] = wifi_num
                        wifi_num += 1
        return wifi_map

    def GetIndex(self, bssid):
        '''
        hash wifi_id to int index
        :param wifi_id: type str; like 'b_6396480'
        :return: int
        '''
        assert bssid in self.wifi_map
        return self.wifi_map[bssid]