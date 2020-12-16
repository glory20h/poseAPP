import numpy as np


# Example Use : kneeAngle = calculateAngle(np.array([hipX, hipY]), np.array([kneeX, kneeY]), np.array([ankleX, ankleY]))
def calculateAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)


kp_avg = np.load('./kp_avg.npy')
print(kp_avg.shape)

elbow_avg = np.array([])
knee_avg = np.array([])

# right_shoulder = np.array([kp_avg[0][6][0], kp_avg[0][6][1], kp_avg[0][6][2]])
# print(right_shoulder)

# for i in range(kp_avg.shape[0]):
#     r_shoulder = np.array([kp_avg[0][6][0], kp_avg[0][6][1]])   #7
#     r_elbow = np.array([kp_avg[0][8][0], kp_avg[0][8][1]])      #9
#     r_wrist = np.array([kp_avg[0][10][0], kp_avg[0][10][1]])    #11
#     elbow_angle = calculateAngle(r_shoulder, r_elbow, r_wrist)

r_shoulder = np.array([kp_avg[0][6][0], kp_avg[0][6][1]])  # 7
r_elbow = np.array([kp_avg[0][8][0], kp_avg[0][8][1]])  # 9
r_wrist = np.array([kp_avg[0][10][0], kp_avg[0][10][1]])  # 11
elbow_angle = calculateAngle(r_shoulder, r_elbow, r_wrist)
print(elbow_angle)