# coding: utf-8

import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

from collections import defaultdict

def Plot(data_dir, filter_shop):
    user_shop_filepath = os.path.join(
        data_dir,
        'ccf_first_round_user_shop_behavior.csv'
    )
    with open(user_shop_filepath, 'r') as fin:
        reader = csv.DictReader(fin)
        lng_table = defaultdict(list)
        lat_table = defaultdict(list)
        for line in reader:
            shop_id = line['shop_id']
            if shop_id not in filter_shop:
                continue
            lng = float(line['longitude'])
            lat = float(line['latitude'])
            lng_table[shop_id].append(lng)
            lat_table[shop_id].append(lat)

        colors = cm.rainbow(np.linspace(0, 1, len(filter_shop)))
        for shop_id, color in zip(filter_shop, colors):
            plt.scatter(lng_table[shop_id], lat_table[shop_id], color)
            print('shop_id={}||size={}'.format(shop_id, len(lng_table[shop_id])))
        plt.show()



if __name__ == '__main__':
    data_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../data/'
    )
    filter_mall_shop = set([('m_7800', 's_681760'), ('m_7800', 's_682074'),
                            ('m_7800', 's_683053'), ('m_7800', 's_683671')])
    filter_shop = set(['s_586869', 's_586869'])
    Plot(data_dir, filter_shop)