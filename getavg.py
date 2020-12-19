import os
import time

import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from requests import Session
import cv2

from tkinter import *
from tkinter import filedialog

APP_KEY = '30c1f56178f37a3d34bc89bb1efeee44'

session = Session()
session.headers.update({'Authorization': 'KakaoAK ' + APP_KEY})

def submit_job_by_file(video_file_path):
    assert os.path.getsize(video_file_path) < 5e7
    with open(video_file_path, 'rb') as f:
        response = session.post('https://cv-api.kakaobrain.com/pose/job', files=[('file', f)])
        response.raise_for_status()
        print("HTTP Status code :", response.status_code)
        return response.json()


# 실제 연동시엔 콜백을 이용한 방식으로 구현하시는 것을 권장합니다
def get_job_result(job_id):
    while True:
        response = session.get('https://cv-api.kakaobrain.com/pose/job/' + job_id)
        response.raise_for_status()
        response = response.json()
        if response['status'] in {'waiting', 'processing'}:
            time.sleep(10)
        else:
            return response


# resp -> job_result
def visualize(resp, threshold=0.2):
    # COCO API를 활용한 시각화
    coco = COCO()
    coco.dataset = {'categories': resp['categories']}
    coco.createIndex()
    width, height = resp['video']['width'], resp['video']['height']

    for frame in resp['annotations']:
        for annotation in frame['objects']:
            # 낮은 신뢰도를 가진 keypoint들은 무시
            keypoints = np.asarray(annotation['keypoints']).reshape(-1, 3)
            low_confidence = keypoints[:, -1] < threshold
            keypoints[low_confidence, :] = [0, 0, 0]
            annotation['keypoints'] = keypoints.reshape(-1).tolist()
        print("--------------")
        plt.axis('off')
        plt.title("frame: " + str(frame['frame_num'] + 1))
        plt.xlim(0, width)
        plt.ylim(height, 0)
        coco.showAnns(frame['objects'])
        plt.show()


# Example Use : kneeAngle = calculateAngle(np.array([hipX, hipY]), np.array([kneeX, kneeY]), np.array([ankleX, ankleY]))
def calculateAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)

# 5개 비디오 knee 각도, elbow 각도를 knee_1.npy, elbow_1.npy에 저장
for i in range(1, 6):
    VIDEO_FILE_PATH = str(i) + '.mp4'
    knee_cat = np.array([])
    elbow_cat = np.array([])

    submit_result = submit_job_by_file(VIDEO_FILE_PATH)
    job_id = submit_result['job_id']
    job_result = get_job_result(job_id)
    for frame in job_result['annotations']:
        kp_resp = np.asarray(frame['objects'][0]['keypoints']).reshape((17,3))

        r_shoulder = np.array([kp_resp[6][0], kp_resp[6][1]])  # 7
        r_elbow = np.array([kp_resp[8][0], kp_resp[8][1]])  # 9
        r_wrist = np.array([kp_resp[10][0], kp_resp[10][1]])  # 11
        elbow_angle = calculateAngle(r_shoulder, r_elbow, r_wrist)
        elbow_cat = np.append(elbow_cat, elbow_angle)

        r_hip = np.array([kp_resp[12][0], kp_resp[12][1]]) # 13
        r_knee = np.array([kp_resp[14][0], kp_resp[14][1]]) # 15
        r_ankle = np.array([kp_resp[16][0], kp_resp[16][1]]) # 17
        knee_angle = calculateAngle(r_hip, r_knee, r_ankle)
        knee_cat = np.append(knee_cat, knee_angle)

    np.save('./elbow_' + str(i), elbow_cat)
    np.save('./knee_' + str(i), knee_cat)


"""
#VIDEO_URL = 'http://example.com/example.mp4'
for i in range(1,6):
    VIDEO_FILE_PATH = str(i)+'.mp4'

    # URL로 영상 지정 시
    #submit_result = submit_job_by_url(VIDEO_URL)
    # 파일로 영상 업로드 시 
    submit_result = submit_job_by_file(VIDEO_FILE_PATH)

    job_id = submit_result['job_id']

    job_result = get_job_result(job_id)
    l = len(job_result['annotations'])
    for i in range(0,l):
        # excel 출력
        print(job_result['annotations'][i]['objects'][0]['keypoints'])
        #response = session.get('https://cv-api.kakaobrain.com/pose/job/' + job_id)
        #print(response.status_code, response.json())
        #print(response.status_code.knee)
    if job_result['status'] == 'success':
        print("success")
        visualize(job_result)
    else:
        print(job_result)
"""

"""
VIDEO_FILE_PATH = '1.mp4'
kp_cat = np.array([])

submit_result = submit_job_by_file(VIDEO_FILE_PATH)
job_id = submit_result['job_id']
job_result = get_job_result(job_id)
for frame in job_result['annotations']:
    kp_resp = np.asarray(frame['objects'][0]['keypoints'])
    kp_cat = np.concatenate((kp_cat, kp_resp))

kp_cat = kp_cat.reshape((1, -1))
print(kp_cat.shape)
"""

"""
VIDEO_FILE_PATH = '1.mp4'

submit_result = submit_job_by_file(VIDEO_FILE_PATH)
job_id = submit_result['job_id']
job_result = get_job_result(job_id)
if job_result['status'] == 'success':
    print("success")
else:
    print("failed")

# print("bbox : ", job_result['annotations'][0]['objects'][0]['bbox'])
print(job_result['annotations'][0]['objects'][0]['keypoints'])
# keypoints = np.asarray(job_result['annotations'][0]['objects'][0]['keypoints']).reshape((1, -1))
# print(keypoints)
"""

"""
# 5개 비디오 keypoint의 평균을 내서 kp_avg.npy에 저장
for i in range(1, 6):
    VIDEO_FILE_PATH = str(i) + '.mp4'
    kp_cat = np.array([])

    submit_result = submit_job_by_file(VIDEO_FILE_PATH)
    job_id = submit_result['job_id']
    job_result = get_job_result(job_id)
    for frame in job_result['annotations']:
        kp_resp = np.asarray(frame['objects'][0]['keypoints'])
        kp_cat = np.concatenate((kp_cat, kp_resp))

    kp_cat = kp_cat.reshape((1, -1))

    if i == 1:
        keypoints = kp_cat
    else:
        keypoints = np.concatenate((keypoints, kp_cat), axis=0)

kp_avg = np.average(keypoints, axis=0)

np.save('./kp_avg', kp_avg)
"""