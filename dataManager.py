from itertools import product
import torch
import numpy as np
import os
from scipy.special import hyp2f1
import torch.utils.data
import time
from params import *

working_dir = os.path.join(os.path.dirname(__file__), 'data')

if not os.path.exists(working_dir):
    os.mkdir(working_dir)
elif os.path.isfile(working_dir):
    raise FileExistsError("working directory occupied")


def abspath(fp): return os.path.join(working_dir, fp)


def data_training_generate_2(seed=0, each=32):
    "generate 1048576 data for usage"
    total = each**4
    random_generator = np.random.Generator(np.random.MT19937(seed))
    data_grid = np.zeros((total, 5), dtype=np.float32)
    a_data, b_data, c_data, x_data = np.meshgrid(
        np.linspace(-1, -5, each), np.linspace(0, 5, each),
        np.linspace(1, 7, each), np.linspace(-1, 0.9, each))
    data_sequence = np.arange(0, total, 1,  dtype=np.uint32)
    random_generator.shuffle(data_sequence)
    _v1, _v2, _v3 = each**3, each**2, each
    for i, j, k, l in product(*([range(each)]*4)):
        k_ = data_sequence[i*_v1+j*_v2+k*_v3+l]
        data_grid[k_, 0] = a_ = a_data[i, j, k, l]
        data_grid[k_, 1] = b_ = b_data[i, j, k, l]
        data_grid[k_, 2] = c_ = c_data[i, j, k, l]
        data_grid[k_, 3] = x_ = x_data[i, j, k, l]
        data_grid[k_, 4] = f_ = hyp2f1(a_, b_, c_, x_)
        assert np.isfinite(f_), \
            "point used not finite at ({},{},{},{})".format(a_, b_, c_, x_)
    with open(abspath('data_train.dat'), 'wb') as file:
        file.write(data_grid.tobytes())


class ProgressShower:
    "substitute for range which has a progress bar"

    def __init__(self, n) -> None:
        self.n = n
        self._last_time = 0

    def tick(self, index):
        if time.time()-self._last_time > 0.1:
            print(f"\r{index}/{self.n}    {self.i*100/self.n:.1f}%",
                  end="        ")
            self._last_time = time.time()

    def complete(self):
        print(f"\r{self.n}/{self.n}    100.0%", end="        ")

    def __iter__(self):
        return ProgressShowerIterator(self.n)


class ProgressShowerIterator(ProgressShower):
    def __init__(self, n) -> None:
        super().__init__(n)
        self.i = 0

    def __next__(self):
        i = self.i
        if i == self.n:
            self.complete()
            raise StopIteration
        else:
            self.i += 1
            self.tick(i)
            return i


class MyDS(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open(abspath('data_train.dat'), 'rb') as file:
            self.data = np.frombuffer(
                file.read(), dtype=np.float32).reshape((-1, 5)).copy()

    def __getitem__(self, index):
        return self.data[index, :4], self.data[index, 4:]

    def __len__(self):
        # return self.data.shape[0]
        return DATASET_NUM

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == "__main__":
    try:
        data_training_generate_2(5)
    except FileExistsError:
        pass
    ds = MyDS()
    for i in range(10):
        (a, b, c, x), (y,) = ds[i]
        j = hyp2f1(a, b, c, x)
        print('%1.6f  %1.6f  %1.6f  %1.6f  %1.6f  %1.6f' % (a, b, c, x, y, j))
