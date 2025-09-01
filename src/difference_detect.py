import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as ply
import numpy as np

def subtract_gray_image(f1, f2, thresholded=None):
    result = np.abs(f1.astype(np.int32) - f2.astype(np.int32)).astype(np.uint8)
    if thresholded is not None:
        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                result[r][c] = 255 if result[r][c] > thresholded else 0
    return result

def matching_image(im1, im2, block_size = 50):
    min_size = min(im1.shape + im2.shape)
    if block_size >= min_size:
        block_size = min_size

    # choose mid block on im1
    r1, c1 = (im1.shape[0]//2) - (block_size//2), (im1.shape[1]//2) - (block_size//2)
    block1 = im1[r1:r1+block_size, c1:c1+block_size]

    # sad: sum absolute diff
    best_sad = float('inf')
    r2, c2 = 0, 0
    for r in range(im2.shape[0]):
        for c in range(im2.shape[1]):
            block2 = im2[r : r + block_size, c : c + block_size]
            if block2.shape != block1.shape:
                continue
            sad = np.sum(subtract_gray_image(block1, block2)**2)
            if sad < best_sad:
                best_sad = sad
                r2, c2 = r, c
    print(best_sad)
    if best_sad == float('inf'):
        return False, None, None

    up, left, down, right = min(r1, r2), min(c1, c2), min(im1.shape[0] - r1, im2.shape[0] - r2), min(im1.shape[1] - c1, im2.shape[1] - c2)
    return True, (r1 - up, c1 - left, r1 + down, c1 + right), (r2 - up, c2 - left, r2 + down, c2 + right)

img1 = cv2.imread("images/block1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/block2.png", cv2.IMREAD_GRAYSCALE)
# img1 = cv2.imread("images/k1.jpg", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("images/k2.jpg", cv2.IMREAD_GRAYSCALE)
ret, rect1, rect2 = matching_image(img1, img2)
if not ret:
    print(":(((")
    exit(1)

copied_1 = img1.copy()
copied_2 = img2.copy()

block1 = img1[rect1[0]:rect1[2], rect1[1]:rect1[3]]
block2 = img2[rect2[0]:rect2[2], rect2[1]:rect2[3]]

cv2.rectangle(copied_1, (rect1[1], rect1[0]), (rect1[3], rect1[2]), 255, 2, cv2.LINE_AA)
cv2.rectangle(copied_2, (rect2[1], rect2[0]), (rect2[3], rect2[2]), 255, 2, cv2.LINE_AA)

plt.figure("Image difference detection")

def show_img(img, idx, title):
    plt.subplot(3, 3, idx)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

show_img(img1, 1, "Image 1")
show_img(img2, 2, "Image 2")
show_img(copied_1, 4, "Block matching 1")
show_img(copied_2, 5, "Block matching 2")

show_img(block1, 7, "block1")
show_img(block2, 8, "block2")
show_img(subtract_gray_image(block1, block2, 30), 9, "subtraction")
plt.tight_layout()
plt.show()