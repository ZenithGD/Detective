import os
import cv2
import mimetypes
import numpy as np

import tkinter as tk
from tkinter import filedialog

from typing import List, Tuple
from detective.logger import Logger

def read_image(path : str) -> np.array:
    """Read an image from a path.

    Args:
        path (str): The path to the image.

    Returns:
        The image object.
    """
    Logger.info(f"Reading file '{path}'")
    # determine file type
    ftype = mimetypes.guess_type(path)
    if ftype[0].startswith('image/'):
        # load if it's an image
        image = cv2.imread(path)

        if not image is None:
            return image

    return None

def read_images(path : str) -> List[Tuple[str, np.array]]:
    """Read images from a directory and store them in a list

    Args:
        path (str): The path to the directory that contains the images

    Returns:
        The list of images.
    """
    images = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)

        # read and add to list
        image = read_image(full_path)
        if not image is None:
            images.append( (file, image) )

    return images

class ImageCarousel(tk.Frame):
    def __init__(self, root, images):

        # define state
        self.root = root
        self.root.title("Image Carousel")

        # OpenCV images
        self.images = images
        self.current_image = 0

        # Convert OpenCV images to Tkinter PhotoImage objects
        self.image_tkinter = [self.convert_opencv_to_tkinter(image[1]) for image in self.images]

        # Create label for image information
        font_info = ("Helvetica", 16)
        self.image_info_label = tk.Label(root, text=f"Image {self.current_image + 1}/{len(self.images)}", font=font_info)
        self.image_info_label.pack(pady=5)

        self.image_info_subtitle = tk.Label(root, text=self.images[self.current_image][0])
        self.image_info_subtitle.pack(pady=5)

        # Create label to display images
        self.image_label = tk.Label(root, image=self.image_tkinter[self.current_image])
        self.image_label.pack(padx=10, pady=10)

        # Create buttons for navigation
        prev_button = tk.Button(root, text="Previous", command=self.show_previous)
        prev_button.pack(side=tk.LEFT, padx=5)

        next_button = tk.Button(root, text="Next", command=self.show_next)
        next_button.pack(side=tk.RIGHT, padx=5)

    def convert_opencv_to_tkinter(self, image):

        # Get the original image dimensions
        height, width, _ = image.shape

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Scale the image to fit within the window (adjust 800 and 600 as needed)
        new_width = 800
        new_height = int(new_width / aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Convert to PhotoImage
        return tk.PhotoImage(data=cv2.imencode('.ppm', resized_image)[1].tobytes())

    def show_previous(self):
        self.current_image = (self.current_image - 1) % len(self.images)
        self.update_image()

    def show_next(self):
        self.current_image = (self.current_image + 1) % len(self.images)
        self.update_image()

    def update_image(self):
        self.image_label.configure(image=self.image_tkinter[self.current_image])
        self.image_info_label.configure(text=f"Image {self.current_image + 1}/{len(self.images)}")
        self.image_info_subtitle.configure(text=self.images[self.current_image][0])

def display_images(images : List[np.array]):
    root = tk.Tk()

    if images:
        app = ImageCarousel(root, images)
        root.mainloop()