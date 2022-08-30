import os

import cv2
import numpy as np

# TODO: Replace this with imageio as to avoid BGR2RGB
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def read_img(in_file, apply_alpha=True, grayscale=False):
    flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH if str(in_file).endswith(".exr") else 0
    img = cv2.imread(str(in_file), flags | cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        img = img[..., None]

    img = img / (1.0 if str(in_file).endswith(".exr") else 255.0)
    alpha = img[:, :, -1][..., None] if img.shape[2] == 4 else 1.0
    img = img if not apply_alpha else img * alpha

    if grayscale:
        # Manually grayscale as we've already converted to floating point pixel values
        # Values from http://en.wikipedia.org/wiki/Grayscale
        b, g, r, _ = np.transpose(img, (2, 0, 1))
        img = 0.0722 * b + 0.7152 * g + 0.2126 * r
        img = img[..., None]

    return img[:, :, :3][:, :, ::-1], alpha


def write_img(out_file, img):
    return cv2.imwrite(out_file[:, :, ::-1], img)
