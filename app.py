import os
import time

import numpy as np
from pycocotools.coco import COCO
from requests import Session
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter.font import Font
from tkinter import filedialog
from PIL import Image, ImageTk

import dtw

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
def visualize_cv(frame_num, job_result, path, elbow_correct=True, knee_correct=True):
    global img_converted
    count = 0
    # print("Frame ", frame_num , " being visualized")

    cap = cv2.VideoCapture(path)
    # Read nth frame
    while count < frame_num:
        ret, img = cap.read()  # Read next frame
        count += 1

    # Draw skeleton on read image
    keypoints = np.asarray(job_result['annotations'][frame_num-1]['objects'][0]['keypoints']).reshape((17, 3))

    # Draw left arm
    left_arm_pts = np.array([[keypoints[5][0], keypoints[5][1]], [keypoints[7][0], keypoints[7][1]], [keypoints[9][0], keypoints[9][1]]], dtype=np.int32)
    if elbow_correct:
        cv2.polylines(img, [left_arm_pts], False, (0,255,0))
    else:
        cv2.polylines(img, [left_arm_pts], False, (0,0,255))

    # Draw right arm
    right_arm_pts = np.array([[keypoints[6][0], keypoints[6][1]], [keypoints[8][0], keypoints[8][1]], [keypoints[10][0], keypoints[10][1]]], dtype=np.int32)
    if elbow_correct:
        cv2.polylines(img, [right_arm_pts], False, (0,255,0))
    else:
        cv2.polylines(img, [right_arm_pts], False, (0,0,255))

    # Draw left leg
    left_leg_pts = np.array([[keypoints[11][0], keypoints[11][1]], [keypoints[13][0], keypoints[13][1]], [keypoints[15][0], keypoints[15][1]]], dtype=np.int32)
    if knee_correct:
        cv2.polylines(img, [left_leg_pts], False, (0,255,0))
    else:
        cv2.polylines(img, [left_leg_pts], False, (0,0,255))

    # Draw right leg
    right_leg_pts = np.array([[keypoints[12][0], keypoints[12][1]], [keypoints[14][0], keypoints[14][1]], [keypoints[16][0], keypoints[16][1]]], dtype=np.int32)
    if knee_correct:
        cv2.polylines(img, [right_leg_pts], False, (0,255,0))
    else:
        cv2.polylines(img, [right_leg_pts], False, (0,0,255))

    # Draw body
    body_pts = np.array([[keypoints[5][0], keypoints[5][1]], [keypoints[11][0], keypoints[11][1]], [keypoints[12][0], keypoints[12][1]], [keypoints[6][0], keypoints[6][1]]], dtype=np.int32)
    # cv2.polylines(img, [body_pts], True, (0,128,255))
    cv2.polylines(img, [body_pts], True, (0,255,0))

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


def get_average(lst):
    return sum(lst) / len(lst)


def analyze_video(video_file_path):
    submit_result = submit_job_by_file(video_file_path)
    job_id = submit_result['job_id']
    job_result = get_job_result(job_id)

    print(job_result)

    elbow_input = np.array([])
    knee_input = np.array([])

    for frame in job_result['annotations']:
        kp_resp = np.asarray(frame['objects'][0]['keypoints']).reshape((17,3))

        r_shoulder = np.array([kp_resp[6][0], kp_resp[6][1]])  # 7
        r_elbow = np.array([kp_resp[8][0], kp_resp[8][1]])  # 9
        r_wrist = np.array([kp_resp[10][0], kp_resp[10][1]])  # 11
        elbow_angle = calculateAngle(r_shoulder, r_elbow, r_wrist)
        elbow_input = np.append(elbow_input, elbow_angle)

        r_hip = np.array([kp_resp[12][0], kp_resp[12][1]]) # 13
        r_knee = np.array([kp_resp[14][0], kp_resp[14][1]]) # 15
        r_ankle = np.array([kp_resp[16][0], kp_resp[16][1]]) # 17
        knee_angle = calculateAngle(r_hip, r_knee, r_ankle)
        knee_input = np.append(knee_input, knee_angle)

    print("Video Analyze Successful")
    return job_result, elbow_input, knee_input


def compare(elbow_input, knee_input, job_result, file_path):
    #Compare elbow angles and plot
    elbow_ans = np.load('elbow_2.npy')
    elbow_input_trim, elbow_ans_trim, elbow_score, elbow_trim_frame = dtw.DTW(elbow_input, elbow_ans)

    ax1.plot(elbow_input_trim, label='Uploaded')
    ax1.plot(elbow_ans_trim, label='Suggested')
    ax1.legend(loc='best')
    ax1.get_xaxis().set_visible(False)
    ax1.set_title('Elbow Angle Comparison')

    #Compare knee angles and plot
    knee_ans = np.load('knee_2.npy')
    knee_input_trim, knee_ans_trim, knee_score, knee_trim_frame = dtw.DTW(knee_input, knee_ans)

    ax2.plot(knee_input_trim, label='Uploaded')
    ax2.plot(knee_ans_trim, label='Suggested')
    ax2.legend(loc='best')
    ax2.get_xaxis().set_visible(False)
    ax2.set_title('Knee Angle Comparison')

    bar.get_tk_widget().pack(side="right")

    # print(elbow_score)
    # print(knee_score)
    # print(get_average(elbow_input_trim))
    # print(get_average(elbow_ans_trim))
    # print(get_average(knee_input_trim))
    # print(get_average(knee_ans_trim))

    if elbow_score < 1000 and knee_score < 1000:
        print_text = "Overall, Good Job!"
    else:
        print_text = ""
        if elbow_score > 1000:
            if get_average(elbow_input_trim) > get_average(elbow_ans_trim):
                print_text += "Overall, bend your elbows a little more. "
            else:
                print_text += "Overall, bend your elbows a little less. "
        if knee_score > 1000:
            if get_average(knee_input_trim) > get_average(knee_ans_trim):
                print_text += "Overall, bend your knees a little more. "
            else:
                print_text += "Overall, bend your knees a little less. "

    info_label.config(text=print_text)

    timer_display_elbow(0, elbow_trim_frame, elbow_input_trim, elbow_ans_trim, knee_trim_frame, knee_input_trim, knee_ans_trim, job_result, file_path)


def timer_display_elbow(idx, elbow_trim_frame, elbow_input_trim, elbow_ans_trim, knee_trim_frame, knee_input_trim, knee_ans_trim, job_result, file_path):
    # Loop for elbow
    elbow_error = (elbow_input_trim[idx]-elbow_ans_trim[idx])/elbow_ans_trim[idx]
    if elbow_error > 0.075:
        elbow_correct = False
    else:
        elbow_correct = True

    visualize_cv(elbow_trim_frame[idx] + 1, job_result, file_path, elbow_correct=elbow_correct)
    idx += 1
    if idx < len(elbow_trim_frame):
        root.after(500, lambda: timer_display_elbow(idx, elbow_trim_frame, elbow_input_trim, elbow_ans_trim, knee_trim_frame, knee_input_trim, knee_ans_trim, job_result, file_path))
    else:
        root.after(500, lambda: timer_display_knee(0, knee_trim_frame, knee_input_trim, knee_ans_trim, job_result, file_path))


def timer_display_knee(idx, knee_trim_frame, knee_input_trim, knee_ans_trim, job_result, file_path):
    # Loop for knee
    knee_error = (knee_input_trim[idx] - knee_ans_trim[idx]) / knee_ans_trim[idx]
    if knee_error > 0.075:
        knee_correct = False
    else:
        knee_correct = True

    visualize_cv(knee_trim_frame[idx] + 1, job_result, file_path, knee_correct=knee_correct)
    idx += 1
    if idx < len(knee_trim_frame):
        root.after(500, lambda: timer_display_knee(idx, knee_trim_frame, knee_input_trim, knee_ans_trim, job_result, file_path))
    else:
        return


def analyze_file():
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    file_path = root.filename
    # img_display.config(image=loading_img)
    job_result, elbow_input, knee_input = analyze_video(file_path)
    compare(elbow_input, knee_input, job_result, file_path)


init_img = ImageTk.PhotoImage(Image.open("./init2.png"))
loading_img = ImageTk.PhotoImage(Image.open("./loading.png"))

my_font = Font(family="Times", size=20, weight="bold", slant="roman")
btn_font = Font(size=16, weight="bold")

info_label = tk.Label(root, text="", font=my_font, padx=5, pady=20)
display_frame = tk.LabelFrame(root)
find_file_btn = tk.Button(root, text="Open File", font=btn_font, command=analyze_file, padx=5, pady=5)

# Widgets that go inside display_frame
img_display = tk.Label(display_frame, image=init_img)
img_display.pack(side="left")

fig = plt.Figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.set_ylabel('angle')
ax2.set_ylabel('angle')
bar = FigureCanvasTkAgg(fig, display_frame)

info_label.pack(side="top", fill="x")
display_frame.pack(expand=True, fill="both", padx=5, pady=5)
find_file_btn.pack(side="bottom", pady=10)

root.mainloop()