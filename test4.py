import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

elbow1 = np.load('./elbow_1.npy')
elbow2 = np.load('./elbow_2.npy')

distance, path = fastdtw(elbow1, elbow2, dist=euclidean)
print(distance)
print(path)

# x = np.array([1, 2, 3, 3, 7])
# y = np.array([1, 2, 2, 2, 2])
#
# distance, path = fastdtw(x, y, dist=euclidean)
#
# print(distance)
# print(path)

# x = np.array([1, 7, 2, 8, 3, 9, 4, 10])
# x_trim = x < 5
# print(x_trim)