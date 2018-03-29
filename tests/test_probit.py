# coding=utf-8
from __future__ import unicode_literals, print_function, absolute_import
from tirt import BayesProbitModel
import numpy as np
from functools import partial

slop = np.array([
    [0.626, -1.003, 0],
    [-0.861, 0, 0.809],
    [0, 0.615, -0.967],
    [0.733, 0.948, 0],
    [0.919, 0, 0.584],
    [0, 0.655, 0.772]
])

threshold = np.array([
    0.593,
    -0.660,
    0.419,
    -0.821,
    0.373,
    0.657
])

# 先验分布的协方差矩阵
sigma = np.array([[1., -0.333, 0.074],
                  [-0.333, 1., 0.362],
                  [0.074, 0.362, 1.]])

data = np.loadtxt('pairs3traits.dat')
scores = data[:, :-3]
true_thetas = data[:, -3:]

thetas = np.zeros((len(scores), 3))

m = partial(BayesProbitModel, sigma=sigma)

for i, score in enumerate(scores):
    model = m(slop, threshold, score=score)
    thetas[i] = model.newton
    print(model.info(thetas[[i]]))
    print(i)
