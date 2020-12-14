import numpy as np

kp_avg = np.load('./kp_avg.npy')
kp_avg = kp_avg.reshape((75,17,3))
np.save('./kp_avg', kp_avg)
print(kp_avg.shape)