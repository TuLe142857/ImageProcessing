import cv2
import matplotlib.pyplot as plt
import numpy as np

def iterative_thresholding(img, theta=0.001):
    total_gray_val = np.sum(img).astype(np.uint32)
    count_pixel = img.shape[0] * img.shape[1]
    t = total_gray_val/count_pixel
    while True:
        f_mask = img < t
        count_f = np.sum(f_mask).astype(np.uint32)
        count_b = count_pixel - count_f
        total_val_f = np.sum(img * f_mask).astype(np.uint32)
        total_val_b = total_gray_val - total_val_f
        m_f = total_val_f/count_f
        m_b = total_val_b/count_b
        new_t = (m_f + m_b)/2
        if abs(t - new_t) < theta:
            return int(new_t)
        t = new_t

img = cv2.imread('brain.jpg', cv2.IMREAD_GRAYSCALE)
print(iterative_thresholding(img))