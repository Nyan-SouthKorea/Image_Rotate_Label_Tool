import os
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np

class RotateLabelTool:
    def __init__(self, dataset_path, img_size=640, max_angle=45):
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'images')
        self.labels_path = os.path.join(dataset_path, 'labels')
        os.makedirs(self.labels_path, exist_ok=True)

        self.img_size = img_size
        self.max_angle = max_angle
        self.angle = 0
        self.current_idx = 0

        self.image_list = [f for f in os.listdir(self.images_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
        self.image_list.sort()
        self.total_images = len(self.image_list)

        self.config_path = os.path.join(os.getcwd(), 'config.json')
        self.load_bookmark()

        self.root = tk.Tk()
        self.root.title("Rotate Labeling Tool")

        self.canvas = tk.Canvas(self.root, width=img_size, height=img_size)
        self.canvas.pack()

        self.gauge_frame = tk.Frame(self.root, height=50)
        self.gauge_frame.pack(fill='x')

        self.gauge = tk.Canvas(self.gauge_frame, height=50, bg='lightgray')
        self.gauge.pack(fill='x')
        self.gauge.bind("<B1-Motion>", self.drag_gauge)
        self.gauge.bind("<Button-1>", self.drag_gauge)

        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack()

        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.prev_image)

        self.load_image()
        self.root.mainloop()

    def load_bookmark(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            bookmark_name = config.get('bookmark', None)
            if bookmark_name:
                for idx, fname in enumerate(self.image_list):
                    if os.path.splitext(fname)[0] == bookmark_name:
                        self.current_idx = idx
                        break

    def save_bookmark(self):
        config = {"bookmark": self.img_name}
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

    def load_image(self):
        img_file = self.image_list[self.current_idx]
        self.img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(self.images_path, img_file)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize_keep_ratio(img, self.img_size)
        self.original_img = img.astype(np.float32)

        label_path = os.path.join(self.labels_path, f"{self.img_name}.json")
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
            self.angle = label.get('rotate', 0)
        else:
            self.angle = 0

        self.update_display()

    def resize_keep_ratio(self, img, size):
        h, w = img.shape[:2]
        scale = size / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        padded = np.full((size, size, 3), 125, dtype=np.uint8)
        y_offset = (size - resized.shape[0]) // 2
        x_offset = (size - resized.shape[1]) // 2
        padded[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
        return padded

    def rotate_image(self, img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=(125, 125, 125))
        return rotated.astype(np.uint8)


    def update_display(self):
        rotated_img = self.rotate_image(self.original_img, self.angle)
        img_pil = Image.fromarray(rotated_img)
        self.tk_img = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")  # ★★★ 여기 추가 ★★★
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)

        grid_interval = 80
        for i in range(0, self.img_size, grid_interval):
            self.canvas.create_line(i, 0, i, self.img_size, fill='white', dash=(2, 2))
            self.canvas.create_line(0, i, self.img_size, i, fill='white', dash=(2, 2))

        self.gauge.delete("all")
        w = self.gauge.winfo_width()
        if w == 1:
            w = self.root.winfo_width()
        cursor_x = int((self.angle + self.max_angle) / (2 * self.max_angle) * w)
        self.gauge.create_rectangle(0, 20, w, 30, fill='white')
        self.gauge.create_line(cursor_x, 0, cursor_x, 50, fill='red', width=3)

        self.progress_label.config(text=f"{self.current_idx + 1} / {self.total_images} - {self.img_name}")


    def drag_gauge(self, event):
        w = self.gauge.winfo_width()
        x = min(max(event.x, 0), w)
        ratio = x / w
        self.angle = (ratio * 2 - 1) * self.max_angle
        self.update_display()

    def save_label(self):
        label = {'rotate': round(self.angle, 2)}
        label_path = os.path.join(self.labels_path, f"{self.img_name}.json")
        with open(label_path, 'w') as f:
            json.dump(label, f)

    def next_image(self, event=None):
        self.save_label()
        self.save_bookmark()
        if self.current_idx < self.total_images - 1:
            self.current_idx += 1
            self.load_image()

    def prev_image(self, event=None):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

if __name__ == "__main__":
    dataset_path = filedialog.askdirectory(title='Select Dataset Folder')
    if dataset_path:
        RotateLabelTool(dataset_path)