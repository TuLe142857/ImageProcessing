import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_bin_img(file_path, size:tuple[int, int]):
    img = np.zeros(size, np.uint8)
    with open(file_path, 'br') as f:
        data = f.read()
        for r in range(256):
            for c in range(256):
                # start = (row * 256) + column
                # end = start + 1
                start = (r * 256) + c
                end = start + 1
                img[r][c] = int.from_bytes(data[start:end])
    return img

def draw_rect(img, p1, p2, c):
    a = np.copy(img)
    cv2.rectangle(a, p1, p2, c)
    return a

img = read_bin_img("actontBin.bin", (256, 256))
template = img[108:138, 67:83]


def m2_matching(source, temp, pos:tuple[int, int])->float:
    match_zone = source[pos[0]:pos[0]+temp.shape[0], pos[1]:pos[1]+temp.shape[1]]
    if match_zone.shape != temp.shape:
        return 0.0
    match_count = np.sum(match_zone == temp)
    return match_count / (temp.shape[0] * temp.shape[1])


def best_match(source, temp):
    best = 0.0
    pos = None
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            m = m2_matching(source, temp, (i, j))
            if m > best:
                best = m
                pos = (i, j)
    return best, pos

rendered= None
for _ in range(0, 1):
    best = 0.0
    pos = None
    size = None

    scaled = cv2.resize(template, [template.shape[1] +_, template.shape[0]+_])
    for i, j in ((x, y) for x in range(img.shape[0]) for y in range(img.shape[1])):
        _this_m2 = m2_matching(img, scaled, (i, j))
        print(_this_m2, best)
        if best < _this_m2:
            best = _this_m2
            pos = (i, j)
            size = scaled.shape




            rendered = draw_rect(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (j, i), (j + scaled.shape[1], i + scaled.shape[0]), (255, 0, 0))
        # cv2.imshow("hi", rendered)
        # key = cv2.waitKey(1)
        # if key == 32:
        #     while True:
        #         if cv2.waitKey(1) == 32:
        #             break

plt.imshow(rendered)
plt.show()