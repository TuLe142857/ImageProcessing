import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------------------
# 1. Hàm tạo ma trận Haar NxN
# ------------------------------
def haar_matrix(N):
    """Tạo ma trận Haar size N x N"""
    if N == 2:
        return np.array([[1,1],
                         [1,-1]]) / np.sqrt(2)
    else:
        Hn_2 = haar_matrix(N//2)
        top = np.hstack([Hn_2, Hn_2])
        bottom = np.hstack([Hn_2, -Hn_2])
        return np.vstack([top, bottom]) / np.sqrt(2)

# ------------------------------
# 2. Hàm áp dụng 2D Haar transform
# ------------------------------
def haar_2d(img, N):
    """
    Áp dụng 2D Haar transform cho ảnh xám
    img: numpy array 2D
    N: kích thước transform (2,4,8,...)
    """
    # Chỉ lấy phần ảnh vừa chia hết cho N
    h, w = img.shape
    h_crop = (h//N)*N
    w_crop = (w//N)*N
    img_crop = img[:h_crop, :w_crop].astype(np.float32)

    H = haar_matrix(N)
    # Áp dụng block Haar transform
    out = np.zeros_like(img_crop)
    for i in range(0, h_crop, N):
        for j in range(0, w_crop, N):
            block = img_crop[i:i+N, j:j+N]
            out[i:i+N, j:j+N] = H @ block @ H.T
    return out

# ------------------------------
# 3. Đọc ảnh xám
# ------------------------------
img = cv2.imread("croppedBike.png", cv2.IMREAD_GRAYSCALE)

# ------------------------------
# 4. Áp dụng Haar transform các cấp N=2,4,8
# ------------------------------
G2 = haar_2d(img, 2)
G4 = haar_2d(img, 4)
G8 = haar_2d(img, 8)

# ------------------------------
# 5. Hiển thị ảnh gốc và các kết quả
# ------------------------------
plt.figure(figsize=(12,3))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Haar N=2")
plt.imshow(G2, cmap='gray')
plt.axis("off")

plt.subplot(1,4,3)
plt.title("Haar N=4")
plt.imshow(G4, cmap='gray')
plt.axis("off")

plt.subplot(1,4,4)
plt.title("Haar N=8")
plt.imshow(G8, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
