import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("petter.png", cv2.IMREAD_GRAYSCALE)

t = 100

mask = (img < t).astype(np.uint8)

print(img)
print(mask)

res = img * mask

plt.figure("demo 1", (10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("original")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title("thresholded")

plt.subplot(1, 3, 3)
plt.imshow(res, cmap='gray')
plt.title("ori * thres")


plt.tight_layout()
plt.show()



