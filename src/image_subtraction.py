import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("images/ic1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/ic2.jpg", cv2.IMREAD_GRAYSCALE)

def subtract_gray_image(f1, f2, thresholded=None):
    result = np.abs( f1.astype(np.int32) - f2.astype(np.int32) )
    result = result.astype(np.uint8)

    if thresholded is not None:
        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                result[r][c] = 255 if result[r][c] > thresholded else 0
    return result

# return top-lef coordinate of similar block(row, column)
def block_matching(block, frame)-> tuple[int, int]:
    bh, bw = block.shape
    best_sad = float('inf')
    r, c = 0, 0
    h, w = frame.shape

    for i in range(h):
        for j in range(w):
            blk = frame[i:i+bh, j:j+bw]
            if blk.shape != block.shape:
                continue
            sad = np.sum(np.abs(block - blk))
            if sad < best_sad:
                r, c = i, j
                best_sad = sad
    return r, c


diff = subtract_gray_image(img1, img2, thresholded=10)

shifted_ic1 = img1[0:500:, 0::]
shifted_ic2 = img2[0::, 0:-100:]
# shifted_ic1 = img1[0:, 0:]
# shifted_ic2 = img2[0:, 0:]

# choose block from shifted image 1
def mid_block(frame, size):
    h, w = frame.shape
    mid_point = (h//2, w//2)
    top_left_point = (mid_point[0] - size//2, mid_point[1] - size//2)
    return frame[top_left_point[0]:top_left_point[0] + size, top_left_point[1]:top_left_point[1] + size], top_left_point

block, tl_point = mid_block(shifted_ic1, 100)

r1, c1 = tl_point
r2, c2 = block_matching(block, shifted_ic2)

clone1 = shifted_ic1.copy()
clone2 = shifted_ic2.copy()

# cv2.rectangle(clone1, (tl_point[1], tl_point[0]), (tl_point[1] + 100, tl_point[0] + 100), 255, 2, cv2.LINE_AA)
# cv2.rectangle(clone2, (c2, r2), (c2 + 100, r2 + 100), 255, 2, cv2.LINE_AA)


h1, w1 = min(r1, r2), min(c1, c2)
h2, w2 = min(clone1.shape[0] - r1, clone2.shape[0] - r2), min(clone1.shape[1] - c1, clone2.shape[1] - c2)

pt11 = (c1 - w1, r1 - h1)
pt12 = (c1 + w2, r1 + h2)

pt21 = (c2 - w1, r2 - h1)
pt22 = (c2 + w2, r2 + h2)

b1 = clone1[pt11[1]:pt12[1], pt11[1]:pt12[1]]
b2 = clone2[pt21[1]:pt22[1], pt21[1]:pt22[1]]

cv2.rectangle(clone1, (tl_point[1], tl_point[0]), (tl_point[1] + 100, tl_point[0] + 100), 255, 2, cv2.LINE_AA)
cv2.rectangle(clone2, (c2, r2), (c2 + 100, r2 + 100), 255, 2, cv2.LINE_AA)
cv2.rectangle(clone1, pt11, pt12, 255, 2, cv2.LINE_AA)
cv2.rectangle(clone2, pt21, pt22, 255, 2, cv2.LINE_AA)



plt.figure("Image subtraction + Block matching", (10, 5))

def show_img(img,idx, title):
    plt.subplot(3, 3, idx)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

show_img(img1, 1, "ic1")
show_img(img2, 2, "ic2")
show_img(diff, 3, "diff")

show_img(clone1, 4, "shifted ic1")
show_img(clone2, 5 ,"shifted ic2")

show_img(b1, 7, "b1")
show_img(b2, 8, "b2")
show_img(subtract_gray_image(b1, b2, thresholded=10), 9, "res")


# show_img()

# plt.tight_layout()
plt.show()