"""
    A random data_generator for x1 + x2 < 1
    Create by: FlyingBlade
    Create Time: 2018/6/7 20:58
"""
import numpy as np
class DataGenerator():
    def __init__(self, dataset_size, seed=1, func=None):
        self.dataset_size = dataset_size
        self.seed = seed
        self.func = func
        if func is None:
            self.func = lambda x: x[0] + x[1] < 1
    def generate(self):
        # load data
        rdm = np.random.RandomState(self.seed)
        df_x = rdm.rand(self.dataset_size, 2)
        df_y = np.reshape([float(self.func(x)) for x in df_x], (-1, 1))
        return df_x, df_y


