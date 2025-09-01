import cv2
import matplotlib.pyplot as plt
import numpy as np
# read bgr
images = [
    cv2.imread("images/hdr1.jpg"),
    cv2.imread("images/hdr2.jpg"),
    cv2.imread("images/hdr3.jpg"),
    cv2.imread("images/hdr4.jpg")
]

# convert to rgb
images = [cv2.cvtColor(_, cv2.COLOR_BGR2RGB) for _ in images]


plt.figure("HDR", (10, 5))

for (idx, i) in enumerate(images, 1):
    plt.subplot(2, 4, idx)
    plt.imshow(i)
    plt.title(f"Image {idx}")
    plt.axis('off')

# Merge HDR với Mertens
merge_mertens = cv2.createMergeMertens()
hdr = merge_mertens.process([img.astype('float32')/255.0 for img in images])  # float32 0-1

# Normalize để hiển thị matplotlib
hdr_display = cv2.normalize(hdr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hdr_display = (hdr_display*255).astype('uint8')
plt.subplot(2, 4, 5)
plt.title("Result")
plt.imshow(hdr_display)
plt.axis('off')

plt.tight_layout()
plt.show()
