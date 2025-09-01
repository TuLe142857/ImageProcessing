import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
CONTRAST LIMIT
"""

img = cv2.imread("images/moon.jpg", cv2.IMREAD_GRAYSCALE)


plt.figure("Contrast-limit Histogram equalization", (10, 5))

plt.subplot(2, 2, 1)
plt.title("no clipping")
plt.imshow(cv2.equalizeHist(img), cmap='gray')
plt.axis('off')

# contrast limit
limits = [0.1, 0.4, 0.7]
for (idx, i) in enumerate(limits, 2):
    plt.subplot(2, 2, idx)
    clahe = cv2.createCLAHE(clipLimit=i, tileGridSize=(8, 8))
    res = clahe.apply(img)
    plt.imshow(res, cmap='gray', vmin = 0, vmax = 255)
    plt.title(f"Limit {i}")
    plt.axis('off')

plt.tight_layout()
plt.show()

