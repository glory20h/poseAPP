import os
import time

import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from requests import Session

APP_KEY = '573c947b3506e5a9cc39a61b2597cb71'

session = Session()
session.headers.update({'Authorization': 'KakaoAK ' + APP_KEY})

"""
def submit_job_by_url(video_url):
    response = session.post('https://cv-api.kakaobrain.com/pose/job', data={'video_url': video_url})
    response.raise_for_status()
    return response.json()
"""


def submit_job_by_file(video_file_path):
    assert os.path.getsize(video_file_path) < 5e7
    with open(video_file_path, 'rb') as f:
        response = session.post('https://cv-api.kakaobrain.com/pose/job', files=[('file', f)])
        response.raise_for_status()
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