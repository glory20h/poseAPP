import numpy as np

# a = np.arange(1, 29)
# a = a.reshape((4,7))
# a = a.flatten()
# print(a)

# a = np.array([1, 2, 3])
# print(a.shape)
# a = a.reshape(1,-1)
# print(a.shape)
# print(a)

######################################################
# a = np.array([1, 2, 3]).reshape((1, -1))
# b = np.array([4, 5, 6]).reshape((1, -1))
# c = np.array([7, 8, 9]).reshape((1, -1))
#
# cat = np.array([])
# cat = np.concatenate((cat, a), axis=0)
# cat = np.concatenate((cat, b), axis=0)
# cat = np.concatenate((cat, c), axis=0)
# print(cat)
#
# avg = np.average(cat, axis=0)
# print(avg)
######################################################

# cat = np.array([])
# a = np.array([1, 4])
# b = np.array([6, 2])
#
# cat = np.concatenate((cat, a))
# cat = np.concatenate((cat, b))
# print(cat)

# exp = np.arange(1, 25)
# exp = exp.reshape((4, 3, 2))
# print(exp[1][0][0])

cat = np.array([])
a = 35
b = 42

cat = np.append(cat, a)
cat = np.append(cat, b)
print(cat)