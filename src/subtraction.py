import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("images/ic1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/ic2.jpg", cv2.IMREAD_GRAYSCALE)

# subtract = np.abs(img1- img2)
subtract = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
subtract = subtract.astype(np.uint8)

plt.figure("Image Subtraction")

plt.subplot(1, 3, 1)
plt.title("IC1")
plt.imshow(img1, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("IC2")
plt.imshow(img2, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Subtract")
plt.imshow(subtract, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
