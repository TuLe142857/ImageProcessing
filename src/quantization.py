import cv2
import matplotlib.pyplot as plt

def quantization(img, nbit):
    # select (8-n) bits
    return img >> (8 - nbit)

image = cv2.imread("images/doraemon.jpg", cv2.IMREAD_GRAYSCALE)

bits = [8, 5, 4, 3 , 2, 1]
plt.figure("Image Quantization", (10, 5))
for (idx, i) in enumerate(bits, 1):
    plt.subplot(2, 3, idx)
    plt.imshow(quantization(image, i), cmap='gray')
    plt.title(f"{i} bit{'s' if i > 1 else ''}")
    plt.axis('off')

plt.tight_layout()
plt.show()