import numpy as np


def gen_mixture():
    n_comp = 5
    size = 200

    x = map(np.random.randn, [size] * n_comp, [2] * n_comp)
