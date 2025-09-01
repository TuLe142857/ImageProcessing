import cv2
# import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/brain.jpg', cv2.IMREAD_GRAYSCALE)

h, w = img.shape
# n_pixel = w * h

x = [_ for _ in range(256)]
y = [0 for _ in range(256)]

for r in range(h):
    for c in range(w):
        y[img[r][c]] += 1
plt.figure("Histogram", (10, 5))

plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram")
plt.plot(x, y, color='b', linestyle='-')

plt.show()