import os
import time

import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from requests import Session
import cv2

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

APP_KEY = '30c1f56178f37a3d34bc89bb1efeee44'

session = Session()
session.headers.update({'Authorization': 'KakaoAK ' + APP_KEY})

# Tkinter 창
root = tk.Tk()
root.title('Basketball Pose Correction')
root.iconbitmap('./basketball.ico')


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


# Convert OpenCV img -> PIL ImageTk format for GUI display
def convert(img):
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cvt)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    return img_tk


#Input : frame num, job_result, file path Output : display skeleton-drawn frame image
def visualize_cv(frame_num, job_result, path):
    global img_converted
    count = 0

    cap = cv2.VideoCapture(path)
    # Read nth frame
    while count < frame_num:
        ret, img = cap.read()  # Read next frame
        count += 1

    # Draw skeleton on read image
    keypoints = np.asarray(job_result['annotations'][frame_num-1]['objects'][0]['keypoints']).reshape((17, 3))

    # Draw left arm
    # int(round(x)) -> round it and change it to integer
    left_arm_pts = np.array([[keypoints[5][0], keypoints[5][1]], [keypoints[7][0], keypoints[7][1]], [keypoints[9][0], keypoints[9][1]]], dtype=np.int32)
    cv2.polylines(img, [left_arm_pts], False, (0,255,0))

    # Draw right arm
    right_arm_pts = np.array([[keypoints[6][0], keypoints[6][1]], [keypoints[8][0], keypoints[8][1]], [keypoints[10][0], keypoints[10][1]]], dtype=np.int32)
    cv2.polylines(img, [right_arm_pts], False, (0,255,0))

    # Draw left leg
    left_leg_pts = np.array([[keypoints[11][0], keypoints[11][1]], [keypoints[13][0], keypoints[13][1]], [keypoints[15][0], keypoints[15][1]]], dtype=np.int32)
    cv2.polylines(img, [left_leg_pts], False, (0,255,0))

    # Draw right leg
    right_leg_pts = np.array([[keypoints[12][0], keypoints[12][1]], [keypoints[14][0], keypoints[14][1]], [keypoints[16][0], keypoints[16][1]]], dtype=np.int32)
    cv2.polylines(img, [right_leg_pts], False, (0,255,0))

    # Draw body
    body_pts = np.array([[keypoints[5][0], keypoints[5][1]], [keypoints[11][0], keypoints[11][1]], [keypoints[12][0], keypoints[12][1]], [keypoints[6][0], keypoints[6][1]]], dtype=np.int32)
    cv2.polylines(img, [body_pts], True, (0,128,255))

    # img_converted = convert(img)
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cvt)
    img_converted = ImageTk.PhotoImage(image=img_pil)
    img_display.config(image=img_converted)


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
    print("Video Analyze Successful")
    return job_result


def find_file():
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    file_path = root.filename
    # img_display.config(image=loading_img) Loading... 은 일단 나중에 생각
    job_result = analyze_video(file_path)
    visualize_cv(1, job_result, file_path)


init_img = ImageTk.PhotoImage(Image.open("./init.png"))
loading_img = ImageTk.PhotoImage(Image.open("./loading.png"))

info_label = tk.Label(root, text="Feedback goes here", padx=5, pady=20)
display_frame = tk.LabelFrame(root)
find_file_btn = tk.Button(root, text="Open File", command=find_file, padx=5, pady=5)

img_display = tk.Label(display_frame, image=init_img)
img_display.pack()

info_label.pack(side="top", fill="x")
display_frame.pack(expand=True, fill="both", padx=5, pady=5)
find_file_btn.pack(side="bottom", pady=10)


root.mainloop()




# class DisplayFrame(tk.Frame):
#     def __init__(self, parent, *args, **kwargs):
#         super().__init__(parent, *args, **kwargs)
#         init_img = ImageTk.PhotoImage(Image.open("./init.png"))
#         loading_img = ImageTk.PhotoImage(Image.open("./loading.png"))
#
#         img_display = tk.Label(display_frame, image=init_img)
#         img_display.pack()
#
# def main():
#     root = tk.Tk()
#     root.title('Basketball Pose Correction')
#     root.geometry("1200x700")
#     root.iconbitmap('./basketball.ico')
#     info_label = tk.Label(root, text="Feedback goes here", padx=5, pady=20)
#     display_frame = DisplayFrame(root)
#     find_file_btn = tk.Button(root, text="Open File", command=find_file, padx=5, pady=5)
#
#     root.mainloop()
#
# if __name__ = '__main__':
#     main()