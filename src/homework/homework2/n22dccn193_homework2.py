import cv2
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
print(matplotlib.get_backend())

def read_bin_img(file_path, width=256, height=256, chanel=1):
    data = None
    with open(file_path, 'rb') as file:
        data = file.read()
    data = np.frombuffer(data, dtype=np.uint8)
    return np.reshape(data, (width, height, chanel))


"""
--------------------------------
            2.1
--------------------------------
"""
def homework2_1():
    img = read_bin_img("Mammogram.bin")
    threshold = 128
    bin_img = img > threshold
    contour = np.zeros(bin_img.shape, np.uint8)
    for i in range(1, contour.shape[0]-1):
        for j in range(1, contour.shape[1]-1):
            if (bin_img[i][j] == 1)and(bin_img[i+1][j] == 0 or \
                    bin_img[i-1][j] == 0 or\
                    bin_img[i][j+1] == 0 or\
                    bin_img[i][j-1] == 0):
                contour[i][j] = 1

    plt.figure("Homework 2.1", (10, 5))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")

    plt.subplot(2, 2, 2)
    plt.imshow(bin_img, cmap='gray')
    plt.title(f"Binary(threshold = {threshold})")

    plt.subplot(2, 2, 3)
    plt.imshow(bin_img * img, cmap='gray')
    plt.title("Original * binary")

    plt.subplot(2, 2, 4)
    plt.imshow(contour, cmap='gray')
    plt.title("Contour")

    plt.tight_layout()

"""
--------------------------------
            2.2
--------------------------------
"""
def homework2_2():
    # original
    img = read_bin_img('lady.bin')
    colors = [_ for _ in range(256)]
    colors_count = [np.sum(img == _) for _ in range(256)]

    # full scale contrast stretch
    min_color = np.min(img).astype(np.uint8)
    max_color = np.max(img).astype(np.uint8)
    scaled_img = np.copy(img)
    scaled_img = ((scaled_img - min_color)/(max_color - min_color)) * 255
    scaled_img = scaled_img.astype(np.uint8)
    scaled_color_count = [np.sum(scaled_img == _) for _ in range(256)]

    # plot
    plt.figure("Homework 2.2", (10, 5))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")

    plt.subplot(2, 2, 2)
    plt.plot(colors, colors_count)

    plt.subplot(2, 2, 3)
    plt.imshow(scaled_img, cmap='gray')
    plt.title("Full scale contrast stretch")

    plt.subplot(2, 2, 4)
    plt.plot(colors, scaled_color_count)

    plt.tight_layout()

"""
--------------------------------
            2.3
--------------------------------
"""
def m2_matching(source, temp, pos):
    h, w, _ = temp.shape
    if not((h + pos[0] <= source.shape[0] ) and (w + pos[1] < source.shape[1])):
        return 0.0
    match_zone = source[pos[0]:pos[0]+h, pos[1]:pos[1]+w]
    equal_count = np.sum(match_zone == temp)
    return equal_count/(h*w)

def homework2_3():
    img = read_bin_img('actontBin.bin')
    # letter T
    # template = img[111:138, 66:84]
    template = np.zeros((47, 15, 1), np.uint8)
    template[10:16, :] = 255
    template[16:37, 6:10] = 255
    #letter N
    # template = img[114:138, 105:127]

    # letter O
    # template = img[111:137, 85:104]


    windows = []
    threshold = 0.9
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = m2_matching(img, template, (i, j))
            if val > threshold:
                windows.append((val, i, j))


    windows.sort(key=lambda x:x[0], reverse=True)
    i = 0
    while i < len(windows):
        j = i + 1
        while j < len(windows):
            if (abs(windows[i][1] - windows[j][1]) < template.shape[0]) and \
                (abs(windows[i][2] - windows[j][2]) < template.shape[1]):
                if windows[i][0] < windows[j][0]:
                    windows.pop(i)
                    i -= 1
                    break
                else:
                    windows.pop(j)
                    j -= 1
            j += 1
        i += 1

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    c_idx = 0
    for w in windows:
        color = colors[c_idx]
        c_idx = (c_idx+1)%len(color)

        pos = w[1], w[2]
        cv2.rectangle(img, (pos[1], pos[0]), (pos[1]+template.shape[1], pos[0]+template.shape[0]),color)
        cv2.putText(img, f"{w[0]*100:.02f}%", (pos[1], pos[0]), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

    # cv2.rectangle(img, (pos[1], pos[0]), (pos[1] + template.shape[1], pos[0]+template.shape[0]), (255, 0, 0), 2)
    plt.figure("Homework3", (10, 5))

    plt.imshow(img)
    plt.tight_layout()

"""
--------------------------------
            2.4
--------------------------------
"""

def homework2_4():
    # original
    img = read_bin_img('johnny.bin')
    colors = [_ for _ in range(256)]
    colors_count = [np.sum(img == _)for _ in range(256)]


    #equalization
    equal_img = cv2.equalizeHist(img)
    equal_color_count = [np.sum(equal_img == _) for _ in range(256)]


    #plot
    plt.figure("Homework 2.4")

    plt.subplot(2, 2,1)
    plt.imshow(img, cmap='gray')
    plt.title("original")

    plt.subplot(2, 2, 2)
    plt.plot(colors, colors_count)

    plt.subplot(2, 2, 3)
    plt.imshow(equal_img, cmap='gray')
    plt.title("Equalized")

    plt.subplot(2, 2, 4)
    plt.plot(colors, equal_color_count)
    plt.tight_layout()

if __name__ == '__main__':
    homework2_1()
    homework2_2()
    homework2_3()
    homework2_4()
    plt.show()