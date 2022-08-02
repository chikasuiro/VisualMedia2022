import numpy as np


def cross_covariance(points_p, points_x):
    assert points_p.shape[1] == 3
    assert points_x.shape[1] == 3
    mu_p = np.mean(points_p, axis=0)
    mu_x = np.mean(points_x, axis=0)

    ret = np.zeros([3, 3])
    for i in range(points_p.shape[0]):
        ret += np.dot(points_p[0].transpose(), points_x[0])
    ret /= points_p.shape[0]

    ret += np.dot(mu_p.transpose(), mu_x)

    return ret


def Q_cross_covariance(points_p, points_x):
    mat_cov = cross_covariance(points_p, points_x)
    mat_A = mat_cov - mat_cov.transpose()
    tr = mat_cov.trace()
    delta = np.array([mat_A[1][2], mat_A[2][0], mat_A[0][1]])

    ret = np.hstack([tr, delta])
    lower = np.hstack([delta.reshape(3, 1),
                       mat_cov + mat_cov.transpose() - tr*np.identity(3)])
    ret = np.vstack([ret, lower])

    return ret
