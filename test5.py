import cv2
import tkinter as tk
from PIL import Image, ImageTk

def convert(img):
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cvt)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    return img_tk


def next_frame():
    return "do_what?"


VIDEO_FILE_PATH = '1.mp4'

cap = cv2.VideoCapture(VIDEO_FILE_PATH)
ret, img = cap.read()  # Read next frame
# if ret:
#     cv2.imshow('Frame', img)
img_converted = convert(img)

# my_label = Label(image=img_converted).pack()
# my_btn = Button(root, text="Next Frame", command=next_frame).pack()
#
# root.mainloop()

class MyWidgets(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.src = cv2.imread("opg-CEPH-1003x1024.jpg")
        self.src = cv2.resize(self.src, (640, 400))

        img = cv2.cvtColor(self.src, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.label = tk.Label(self, image=imgtk)
        self.label.image = imgtk #class 내에서 작업할 경우에는 이 부분을 넣어야 보인다.
        self.label.pack(side="top")

        self.button = tk.Button(self, text="이진화 처리", command=self.convert_to_tkimage)
        self.button.pack(side="bottom", expand=True, fill='both')

    def convert_to_tkimage(self):
        gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        img = Image.fromarray(binary)
        imgtk = ImageTk.PhotoImage(image=img)

        self.label.config(image=imgtk)
        self.label.image = imgtk



class MainApp(tk.Tk):
    """Application root window"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Hello World!")
        self.resizable(width=True, height=True)

        self.widgetform = MyWidgets(self)
        self.widgetform.grid(row=3, padx=10, sticky=(tk.W + tk.E))

        self.geometry("640x480+200+200")

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
