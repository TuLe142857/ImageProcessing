import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("peter.png", cv2.IMREAD_GRAYSCALE)

threshold = 112

bin_img = img < threshold

plt.figure("Gray level thresholding", (10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("original")

plt.subplot(1, 3, 2)
plt.imshow(bin_img, cmap='gray')
plt.title(f'threshold  = {threshold}')

plt.subplot(1, 3, 3)
plt.imshow(img * bin_img, cmap='gray')
plt.title("ori * thres")

plt.tight_layout()
plt.show()