import cv2
import matplotlib.pyplot as plt
import numpy as np



img = cv2.imread("petter.png", cv2.IMREAD_GRAYSCALE)

t, mask = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

print(t)
plt.figure(f"threshold = {t}", (10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("original")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("thresholded")
plt.show()