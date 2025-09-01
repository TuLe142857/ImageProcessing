import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

def random_noise(img, percent):
    h, w = img.shape
    noise_img = np.copy(img)
    for i in range(int(h * w * percent)):
        i = random.randint(0, h-1)
        j = random.randint(0, w-1)
        noise_img[i][j] = 255
    return noise_img

def calc_avg_image(imgs):
    sum_img = imgs[0].astype(np.int32)
    for i in range(1, len(imgs)):
        sum_img += imgs[i].astype(np.int32)
    return (sum_img//len(imgs)).astype(np.uint8)

# original image
image = cv2.imread("images/doraemon.jpg", cv2.IMREAD_GRAYSCALE)

# make random n noise image
n = 10
noise_imgs = [random_noise(image, 0.1) for i in range(n)]

avg_img = calc_avg_image(noise_imgs)


plt.figure("Average")

plt.subplot(1, 2, 1)
plt.imshow(noise_imgs[0], cmap='gray')
plt.title("noise image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(avg_img, cmap='gray')
plt.title("average image")
plt.axis('off')

plt.tight_layout()
plt.show()
