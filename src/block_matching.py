import cv2
import matplotlib.pyplot as plt
import numpy as np

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
    print(f"best: {best_sad} - {r}, {c}")
    return r, c


def shift_image(img, dy, dx):
    h, w = img.shape
    shifted = np.zeros_like(img)  # tạo ảnh trống cùng kích thước

    # Xét dịch chuyển dương/âm
    if dy >= 0 and dx >= 0:
        shifted[dy:, dx:] = img[:h - dy, :w - dx]
    elif dy >= 0 and dx < 0:
        shifted[dy:, :w + dx] = img[:h - dy, -dx:]
    elif dy < 0 and dx >= 0:
        shifted[:h + dy, dx:] = img[-dy:, :w - dx]
    else:  # dy <0, dx <0
        shifted[:h + dy, :w + dx] = img[-dy:, -dx:]

    return shifted

img1 = cv2.imread("images/block1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("images/block2.png", cv2.IMREAD_GRAYSCALE)

block_coord = ((100, 100), (200, 200))
block = img1[100:200, 100:200]



plt.figure("Block matching", (10, 5))

r, c = block_matching(block, img2)
cv2.rectangle(img2, (c, r), (c+100, r+100), (255), 2, cv2.LINE_AA)
cv2.rectangle(img1, (100, 100), (200, 200), (255), 2, cv2.LINE_AA)

dx, dy = c-100, r-100
shifted_img = shift_image(img2, -dy, -dx)



plt.subplot(2, 2, 1)
plt.title("Image 1")
plt.imshow(img1, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Image2")
plt.imshow(img2, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Image 2 shifted")
plt.imshow(shifted_img, cmap='gray')
plt.axis('off')

h1, w1 = img1.shape
h2, w2 = img2.shape
h, w = min(h1, h2), min(w1, w2)
sub_img1 = img1[0:h, 0:w]
sub_img2 = shifted_img[0:h, 0:w]
diff_img = np.abs(sub_img1.astype(np.int32) - sub_img2.astype(np.int32)).astype(np.uint8)


plt.subplot(2, 2, 4)
plt.title("diff")
plt.imshow(diff_img, cmap = 'gray', vmin=0, vmax=255)
plt.axis('off')

plt.tight_layout()
plt.show()