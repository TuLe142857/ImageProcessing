import cv2
import matplotlib.pyplot as plt
import numpy as np

imgs = [
    cv2.imread("avg1.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("avg1.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("avg1.png", cv2.IMREAD_GRAYSCALE),
]

sum_img = imgs[0].astype(np.int32) + imgs[1].astype(np.int32) + imgs[2].astype(np.int32)
sum_img //= 3
sum_img = sum_img.astype(np.uint8)

plt.figure("avg image", (10, 5))
plt.imshow(sum_img, cmap='gray', vmin=0, vmax=255)
# plt.show()

plt.figure("ori image", (10, 5))
plt.imshow(imgs[0], cmap='gray')
plt.show()