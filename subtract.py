import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("sub1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("sub2.png", cv2.IMREAD_GRAYSCALE)

result = img1.astype(np.float32) - img2.astype(np.float32)
result = np.abs(result.astype(np.uint8))

plt.figure("subtract image", (10, 5))
plt.imshow(result, cmap='gray')
plt.show()