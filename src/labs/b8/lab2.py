import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh xám
img = cv2.imread("croppedBike.png", cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# --- Tạo ma trận H_x và H_y 2:1 trung bình ---
hx = np.zeros((h//2, h))
for i in range(h//2):
    hx[i, 2*i] = 0.5
    hx[i, 2*i+1] = 0.5

hy = np.zeros((w//2, w))
for i in range(w//2):
    hy[i, 2*i] = 0.5
    hy[i, 2*i+1] = 0.5

# Áp dụng H_x, H_y
f = img.astype(np.float32)
subsampled = hx @ f @ hy.T
subsampled = np.clip(subsampled, 0, 255).astype(np.uint8)

# Hiển thị
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Subsampled 2:1 (Average, theo H)")
plt.imshow(subsampled, cmap='gray')
plt.axis("off")

plt.show()
