import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/brain.jpg", cv2.IMREAD_GRAYSCALE)


# return x[], y[]
def make_histogram(image):
    h, w = image.shape
    x = [_ for _ in range(256)]
    y = [0 for _ in range(256)]
    for r in range(h):
        for c in range(w):
            y[image[r][c]] += 1
    return x, y

def histogram_eq(image):
    return cv2.equalizeHist(image)

plt.figure("Histogram Equalization", (10, 5))

# before
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
x, y = make_histogram(img)
plt.plot(x, y, color='b', linestyle='-')


# after
eq_img = histogram_eq(img)
plt.subplot(2, 2, 3)
plt.imshow(eq_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
x, y = make_histogram(eq_img)
plt.plot(x, y, color='g', linestyle='-')


plt.tight_layout()
plt.show()

