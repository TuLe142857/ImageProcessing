import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("s.png", cv2.IMREAD_GRAYSCALE)


plt.figure("demo")

plt.imshow(img, cmap='gray')
plt.title("Source")

plt.tight_layout()
plt.show()