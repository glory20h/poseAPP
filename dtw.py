# Dynamic Time Warping

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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

    A_changes.reverse()
    B_changes.reverse()

    A_trim = []
    for i in A_changes:
        A_trim.append(A[i])
    B_trim = []
    for i in B_changes:
        B_trim.append(B[i])

    score, _ = fastdtw(A, B, dist=euclidean)

    return A_trim, B_trim, score, A_changes
