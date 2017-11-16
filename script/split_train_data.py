# coding: utf-8

import os
import random

from config import Config

def SplitFile(data_file, train_file, validation_file, rate=0.8):
    with open(data_file, 'r') as fin:
        header = fin.readline()
        data_holder = fin.readlines()
        random.shuffle(data_holder)
        l = len(data_holder)
        train_data = data_holder[:int(l * rate)]
        validation_data = data_holder[int(l * rate) : ]
        with open(train_file, 'w') as fout:
            fout.write('{}'.format(header))
            for line in train_data:
                fout.write('{}'.format(line))

        with open(validation_file, 'w') as fout:
            fout.write('{}'.format(header))
            for line in validation_data:
                fout.write('{}'.format(line))

if __name__ == '__main__':
    data_dir = Config.data_dir
    data_file = os.path.join(
        data_dir,
        Config.user_shop_filename
    )
    train_file = os.path.join(
        data_dir,
        Config.train_filename
    )
    validation_file = os.path.join(
        data_dir,
        Config.validation_filename
    )
    SplitFile(data_file, train_file, validation_file)