import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# elbow1 = np.load('./elbow_1.npy')
# elbow2 = np.load('./elbow_2.npy')
#
# distance, path = fastdtw(elbow1, elbow2, dist=euclidean)
# print(distance)
# print(path)

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

def DTW(A, B):
    alen = len(A)
    blen = len(B)
    dy = [[0.0 for i in range(blen)] for j in range(alen)]
    via = [[0.0 for i in range(blen)] for j in range(alen)]
    # 1 : up  2 : left  3 : diagonal
    for i in range(0, alen):
        dy[i][0] = abs(A[i] - B[0])
        via[i][0] = 1
    for i in range(0, blen):
        dy[0][i] = abs(A[0] - B[i])
        via[0][i] = 2

    for i in range(1, alen):
        for j in range(1, blen):
            dy[i][j] = dy[i - 1][j] + abs(A[i] - B[j])
            via[i][j] = 1
            if dy[i][j] > dy[i][j - 1] + abs(A[i] - B[j]):
                dy[i][j] = dy[i][j - 1] + abs(A[i] - B[j])
                via[i][j] = 2
            if dy[i][j] > dy[i - 1][j - 1] + abs(A[i] - B[j]):
                dy[i][j] = dy[i - 1][j - 1] + abs(A[i] - B[j])
                via[i][j] = 3

    x = alen - 1
    y = blen - 1
    A_changes = []
    B_changes = []
    A_changes.append(x)
    B_changes.append(y)
    # print(dy)
    # print(via)
    while True:
        if x == 0 and y == 0:
            break
        if via[x][y] == 1:
            x = x - 1
        elif via[x][y] == 2:
            y = y - 1
        else:
            x = x - 1
            y = y - 1
            A_changes.append(x)
            B_changes.append(y)

    A_changes, B_changes

    A_trim = []
    for i in A_changes:
        A_trim.append(A[i])
    B_trim = []
    for i in B_changes:
        B_trim.append(B[i])

    A_trim.reverse()
    B_trim.reverse()

    return A_trim, B_trim


# A = np.array([1, 2, 6, 5, 7, 8])
# B = np.array([1, 3, 5, 7, 6, 8, 9, 10, 8, 7])
# print(DTW(A, B))
#
# distance, path = fastdtw(A, B, dist=euclidean)
# print(distance)
# print(path)