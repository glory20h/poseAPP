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

# Tkinter 창
root = Tk()
root.title('Tkinter Test')


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
            # Put "Loading..." Here?
        else:
            return response


# resp -> job_result
def visualize(resp, threshold=0.2):
    # COCO API를 활용한 시각화
    coco = COCO()
    coco.dataset = {'categories': resp['categories']}
    coco.createIndex()
    width, height = resp['video']['width'], resp['video']['height']

    # 낮은 신뢰도를 가진 keypoint들은 무시
    for frame in resp['annotations']:
        for annotation in frame['objects']:
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


def analyze_video(video_file_path):
    submit_result = submit_job_by_file(video_file_path)
    job_id = submit_result['job_id']
    job_result = get_job_result(job_id)
    print(job_result)
    # keypoints = np.asarray(job_result['annotations'][0]['objects'][0]['keypoints'])
    # print(keypoints)


def exp(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    ret, img = cap.read()
    if ret:
        cv2.imshow('Frame', img)


def find_file():
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    file_path = root.filename
    analyze_video(file_path)
    # exp(file_path)


my_btn = Button(root, text="Open File", command=find_file).pack()

root.mainloop()