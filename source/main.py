import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np


class ImageProcessor:
    def __init__(self):
        self.image_path = None
        self.original_frame = None
        self.lower_hue_slider = None
        self.upper_hue_slider = None
        self.lower_saturation_slider = None
        self.upper_saturation_slider = None
        self.lower_value_slider = None
        self.upper_value_slider = None

        self.frame_width = 640

        self.root = tk.Tk()
        self.root.title("Image Masking")

        select_frame = tk.Frame(self.root)
        select_frame.pack(pady=10)
        btn_select = tk.Button(select_frame, text="Select Image", command=self.select_image)
        btn_select.pack(side="left", padx=10)

        masking_frame = tk.Frame(self.root)
        masking_frame.pack(pady=10)

        self.lower_hue_slider = tk.Scale(masking_frame, from_=0, to=255, orient="vertical", label="Lower Hue",
                                         command=self.apply_mask, length=300)
        self.lower_hue_slider.pack(side="left", padx=10)
        self.upper_hue_slider = tk.Scale(masking_frame, from_=0, to=255, orient="vertical", label="Upper Hue",
                                         command=self.apply_mask, length=300)
        self.upper_hue_slider.pack(side="left", padx=10)
        self.lower_saturation_slider = tk.Scale(masking_frame, from_=0, to=255, orient="vertical",
                                                label="Lower Saturation", command=self.apply_mask, length=300)
        self.lower_saturation_slider.pack(side="left", padx=10)
        self.upper_saturation_slider = tk.Scale(masking_frame, from_=0, to=255, orient="vertical",
                                                label="Upper Saturation", command=self.apply_mask, length=300)
        self.upper_saturation_slider.pack(side="left", padx=10)
        self.lower_value_slider = tk.Scale(masking_frame, from_=0, to=255, orient="vertical", label="Lower Value",
                                           command=self.apply_mask, length=300)
        self.lower_value_slider.pack(side="left", padx=10)
        self.upper_value_slider = tk.Scale(masking_frame, from_=0, to=255, orient="vertical", label="Upper Value",
                                           command=self.apply_mask, length=300)
        self.upper_value_slider.pack(side="left", padx=10)

        self.save_config_btn = tk.Button(self.root, text="Save Config", command=self.save_config, state="disabled")
        self.save_config_btn.pack(pady=5)

        self.load_config_btn = tk.Button(self.root, text="Load Config", command=self.load_config, state="disabled")
        self.load_config_btn.pack(pady=5)

        self.bbox_btn = tk.Button(self.root, text="Make Bbox", command=self.make_bbox, state="disabled")
        self.bbox_btn.pack(pady=5)

    @staticmethod
    def resize_shape(source_shape, width=None, height=None):
        (h, w) = source_shape

        if width is None and height is None:
            return h, w

        if width is None:
            r = height / float(h)
            shape = (height, int(w * r))
        else:
            r = width / float(w)
            shape = (int(h * r), width)

        return shape

    def resize_image(self, image, width=None, height=None, inter=cv2.INTER_AREA) -> np.ndarray:
        if width is None and height is None:
            return image

        h, w = self.resize_shape(source_shape=image.shape[:2], width=width, height=height)
        resized = cv2.resize(image, (w, h), interpolation=inter)

        return resized

    def apply_mask(self, event=None):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first." + str(event))
            return

        try:
            image = self.original_frame
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_hue = self.lower_hue_slider.get()
            upper_hue = self.upper_hue_slider.get()
            lower_saturation = self.lower_saturation_slider.get()
            upper_saturation = self.upper_saturation_slider.get()
            lower_value = self.lower_value_slider.get()
            upper_value = self.upper_value_slider.get()

            lower_bound = np.array([lower_hue, lower_saturation, lower_value])
            upper_bound = np.array([upper_hue, upper_saturation, upper_value])

            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            result = cv2.bitwise_and(image, image, mask=mask)

            result = self.resize_image(result, width=self.frame_width)
            cv2.imshow("Masked Image", result)

        except Exception as e:
            messagebox.showerror("Error", str(e) + str(event))

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;.png;.gif;*.bmp")])
        if self.image_path:
            self.load_config_btn.config(state="normal")
            self.save_config_btn.config(state="normal")
            self.bbox_btn.config(state="normal")
            image = cv2.imread(self.image_path)
            image = self.resize_image(image, width=self.frame_width)
            cv2.imshow("Image", image)
            self.original_frame = image
            self.apply_mask()

            self.upper_hue_slider.set(255)
            self.upper_saturation_slider.set(255)
            self.upper_value_slider.set(255)

    def save_config(self):
        config_data = {
            "lower_hue": self.lower_hue_slider.get(),
            "upper_hue": self.upper_hue_slider.get(),
            "lower_saturation": self.lower_saturation_slider.get(),
            "upper_saturation": self.upper_saturation_slider.get(),
            "lower_value": self.lower_value_slider.get(),
            "upper_value": self.upper_value_slider.get()
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "w") as file:
                for key, value in config_data.items():
                    file.write(f"{key}: {value}\n")
            messagebox.showinfo("Save Config", "Config saved successfully.")

    def load_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "r") as file:
                    config_data = {}
                    for line in file:
                        key, value = line.strip().split(": ")
                        config_data[key] = int(value)

                self.lower_hue_slider.set(config_data["lower_hue"])
                self.upper_hue_slider.set(config_data["upper_hue"])
                self.lower_saturation_slider.set(config_data["lower_saturation"])
                self.upper_saturation_slider.set(config_data["upper_saturation"])
                self.lower_value_slider.set(config_data["lower_value"])
                self.upper_value_slider.set(config_data["upper_value"])
                messagebox.showinfo("Load Config", "Config loaded successfully.")
            except FileNotFoundError:
                messagebox.showerror("Load Config", "Config file not found.")

    def make_bbox(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        try:
            image = self.original_frame.copy()
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_hue = self.lower_hue_slider.get()
            upper_hue = self.upper_hue_slider.get()
            lower_saturation = self.lower_saturation_slider.get()
            upper_saturation = self.upper_saturation_slider.get()
            lower_value = self.lower_value_slider.get()
            upper_value = self.upper_value_slider.get()

            lower_bound = np.array([lower_hue, lower_saturation, lower_value])
            upper_bound = np.array([upper_hue, upper_saturation, upper_value])

            mask = cv2.inRange(hsv, lower_bound, upper_bound)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            num_of_contour = 0

            result = self.original_frame.copy()
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

                num_of_contour = i + 1

            messagebox.showinfo("Info", f"Found {num_of_contour} contours.")

            result = self.resize_image(result, width=self.frame_width)
            cv2.imshow("Image", result)

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    learn_image_masking = ImageProcessor()
    learn_image_masking.root.mainloop()
