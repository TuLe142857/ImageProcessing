import cv2
import matplotlib.pyplot as plt
import numpy as np


IMG_PATH = 'doraemon.jpg'
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)


# Quantization
def quantization(img, nbit):
    step = 256 // (2 ** nbit)
    return (img // step) * step

bits = [8, 5, 4, 3, 2, 1]
plt.figure(num="Quantization", figsize=(10, 5))
for idx, i in enumerate(bits, 1):
    plt.subplot(2, 3, idx)
    plt.title(f"{i} bit{"s" if i > 1 else ""}")
    plt.imshow(quantization(img, i), cmap='gray')
    plt.axis('off')
plt.tight_layout()

# Brightness Scaling
def bri_scale(img, alpha):
    return np.clip(img * alpha, 0, 255).astype(np.uint8)

alphas = [1, 0.5, 1.5]
plt.figure(num="Brightness Scaling", figsize=(10, 5))
for idx, i in enumerate(alphas, 1):
    plt.subplot(1, 3, idx)
    plt.imshow(bri_scale(img, i), cmap='gray', vmin=0, vmax=255)
    plt.title(f"alpha = {i}")
    plt.axis('off')
plt.tight_layout()

plt.show()