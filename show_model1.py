"""
显示待拟合的目标函数。
"""

from itertools import product
from scipy.special import hyp2f1
from matplotlib import pyplot as plt
import numpy as np
import torch
from model1 import Model1
from params import *


c_data, x_data = np.meshgrid(np.linspace(1, 7, 60), np.linspace(-1, 0.9, 20))

model = torch.load(f'model1-dataset{DATASET_NUM}-epoch{EPOCH_NUM}-lr{LR}.pth')
model.eval()


def showSlice(a, b):
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    z_data = np.zeros(c_data.shape, dtype=np.float32)
    z2_data = np.zeros(c_data.shape, dtype=np.float32)
    for (i, j) in product(range(z_data.shape[0]), range(z_data.shape[1])):
        c_, x_ = c_data[i, j], x_data[i, j]
        z_data[i, j] = hyp2f1(a, b, c_, x_)
        z2_data[i, j] = model.forward(
            torch.tensor((a, b, c_, x_), dtype=torch.float32))

    surf1 = ax1.plot_surface(x_data, c_data, z_data, alpha=0.6)
    surf2 = ax1.plot_surface(x_data, c_data, z2_data, alpha=0.6)

    ax1.set_xlabel('x')
    ax1.set_ylabel('c')
    ax1.set_zlabel('₂F₁(a,b,c;x)')
    ax1.set_title('a={:.2f} b={:.2f}'.format(a, b))

    plt.show()



while True:
    try:
        a, b = map(float, input('a,b=').split())
        assert -5 <= a <= -1, "a must be in [-5, -1]"
        assert 0 <= b <= 5, "b must be in [0, 5]"
        showSlice(a, b)
    except AssertionError as exc:
        print(exc)
    except EOFError:
        break
    except Exception as exc:
        import traceback
        traceback.print_exc()