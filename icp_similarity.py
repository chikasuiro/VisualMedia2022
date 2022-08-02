import numpy as np


def similarity(points_p, points_x):
    numer = 0
    for i in range(points_p.shape[0]):
        numer += np.linalg.norm(points_p[i] - points_x[i])**2
    numer /= points_p.shape[0]

    denom = np.sqrt(np.dot(points_x, points_x.transpose()).trace())

    return (1 - numer/denom) * 100
