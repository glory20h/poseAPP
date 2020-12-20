import cv2
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk


def convert(img):
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cvt)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    return img_tk


def do_stuff():
    img_display.config(image=img_converted)

cap = cv2.VideoCapture('1.mp4')

ret, img = cap.read()  # Read next frame

# Draw left arm
# left_arm_pts = np.array([[keypoints[5][0], keypoints[5][1]], [keypoints[7][0], keypoints[7][1]], [keypoints[9][0], keypoints[9][1]]], dtype=np.int32)
pts = np.array([[207,302], [221,344], [268,350]], dtype=np.int32)
cv2.polylines(img, [pts], False, (0,255,0))



# img_converted.show()

root = tk.Tk()
root.title('Basketball Pose Correction')
root.iconbitmap('./basketball.ico')

init_img = ImageTk.PhotoImage(Image.open("./init.png"))

img_display = tk.Label(root, image=init_img)
img_display.pack()

img_converted = convert(img)
do_stuff()

root.mainloop()

# import cv2
# from PIL import Image, ImageTk
# import numpy as np
# import tkinter as tk
#
# def convert(img):
#     img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(img_cvt)
#     img_tk = ImageTk.PhotoImage(image=img_pil)
#     return img_tk
#
# cap = cv2.VideoCapture('1.mp4')
#
# ret, img = cap.read()  # Read next frame
#
#
# # Draw left arm
# # left_arm_pts = np.array([[keypoints[5][0], keypoints[5][1]], [keypoints[7][0], keypoints[7][1]], [keypoints[9][0], keypoints[9][1]]], dtype=np.int32)
# pts = np.array([[207,302], [221,344], [268,350]], dtype=np.int32)
# cv2.polylines(img, [pts], False, (0,255,0))
#
# img_converted = convert(img)
#
# # img_converted.show()
#
# root = tk.Tk()
# root.title('Basketball Pose Correction')
# root.iconbitmap('./basketball.ico')
#
# display = tk.Label(root, image=img_converted)
# display.pack()
#
# root.mainloop()