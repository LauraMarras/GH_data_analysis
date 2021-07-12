# """ """ import numpy as np
# import matplotlib.pyplot as plt

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show() """


# import numpy as np

# def correlation_coefficient(T1, T2):
#     numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
#     denominator = T1.std() * T2.std()
#     if denominator == 0:
#         return 0
#     else:
#         result = numerator / denominator
#         return result

# x = np.array([[0.1, .32, .2, 0.4, 0.15], [.1, .38, .26, .41, .12], [1, 3, 2, 4, 8]])
# y = np.array([[0.1, .32, .2, 0.4, 0.15], [.1, .38, .26, .41, .12], [1, 3, 2, 4, 8]])*(-1)
# #y = np.array([[.16,.39,0.19, .42, .12],[.1,0.3,.23, .48, .16], [2,6,4, 8, 12]])
# a= x.flatten()
# b= y.flatten()
# pearson = np.corrcoef(a, b)
# pearson2 = np.corrcoef(x, y)
# c=0
# new=[]
# for row in range(pearson2.shape[0]):
#     for element in (pearson2[row,c:]):
#         new.append(element)
#     c+=1
# new_abs = [abs(x) for x in new]
# nm = np.mean(new_abs)
# pm = pearson[0,1]
# print(pearson2)

# cf=correlation_coefficient(x,y)


# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """Computes the distance correlation between two matrices.
# https://en.wikipedia.org/wiki/Distance_correlation
# """

# from scipy.spatial.distance import pdist, squareform


# __author__ = "Kailash Budhathoki"
# __email__ = "kbudhath@mpi-inf.mpg.de"
# __copyright__ = "Copyright (c) 2019"
# __license__ = "MIT"


# def dcov(X, Y):
#     """Computes the distance covariance between matrices X and Y.
#     """
#     n = X.shape[0]
#     XY = np.multiply(X, Y)
#     cov = np.sqrt(XY.sum()) / n
#     return cov


# def dvar(X):
#     """Computes the distance variance of a matrix X.
#     """
#     return np.sqrt(np.sum(X ** 2 / X.shape[0] ** 2))


# def cent_dist(X):
#     """Computes the pairwise euclidean distance between rows of X and centers
#      each cell of the distance matrix with row mean, column mean, and grand mean.
#     """
#     M = squareform(pdist(X))    # distance matrix
#     rmean = M.mean(axis=1)
#     cmean = M.mean(axis=0)
#     gmean = rmean.mean()
#     R = np.tile(rmean, (M.shape[0], 1)).transpose()
#     C = np.tile(cmean, (M.shape[1], 1))
#     G = np.tile(gmean, M.shape)
#     CM = M - R - C + G
#     return CM


# def dcor(X, Y):
#     """Computes the distance correlation between two matrices X and Y.
#     X and Y must have the same number of rows.
#     >>> X = np.matrix('1;2;3;4;5')
#     >>> Y = np.matrix('1;2;9;4;4')
#     >>> dcor(X, Y)
#     0.76267624241686649
#     """
#     assert X.shape[0] == Y.shape[0]

#     A = cent_dist(X)
#     B = cent_dist(Y)

#     dcov_AB = dcov(A, B)
#     dvar_A = dvar(A)
#     dvar_B = dvar(B)

#     dcor = 0.0
#     if dvar_A > 0.0 and dvar_B > 0.0:
#         dcor = dcov_AB / np.sqrt(dvar_A * dvar_B)

#     return dcor


# if __name__ == "__main__":
#     #X = np.matrix('1;2;3;4;5')
#     #Y = np.matrix('1;2;9;4;4')
#     print(dcor(x, y))
#     print(1-pearson[0,1])

#  """


A = [(1,2),(3,4),(5,6)]
B=A[0:-1][1]

print(B)
