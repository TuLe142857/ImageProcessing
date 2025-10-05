import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
        1.1
        Read image from binary file: lena.bin, peppers.bin (8 bits image)
        Size: 256 * 256
"""
def read_bin_img(file_path):
    img = np.zeros((256, 256), np.uint8)
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

def homework_1_1():
    img_lena = read_bin_img('lena.bin')
    img_peppers = read_bin_img('peppers.bin')
    img_j = np.hstack([img_lena[0:, 0:128], img_peppers[0:, 128:]])
    img_k = img_j[0:, -1::-1]

    plt.figure("homework 1.1", (10, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(img_lena, cmap='gray')
    plt.title("lena.bin")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_peppers, cmap='gray')
    plt.title('peppers.bin')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_j, cmap='gray')
    plt.title('Image J')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(img_k, cmap='gray')
    plt.title("Image K")
    plt.axis('off')


"""
        1.2
        Read 8 bits image, make photographic negative
"""
def homework_1_2():
    img_j1 = cv2.imread("lenagray.jpg", cv2.IMREAD_GRAYSCALE)
    img_j2 = 255 - img_j1

    plt.figure("homework 1.2", (10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_j1, cmap='gray')
    plt.title('Image J1')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_j2, cmap='gray')
    plt.title('Image J2')
    plt.axis('off')

"""
        1.3
        Read 24 bits image
        Swap color chanel
"""
def homework_1_3():
    # by default, cv2 read image in bgr color
    img_j1 = cv2.imread('lena512color.jpg')
    # convert to rgb
    img_j1 = cv2.cvtColor(img_j1, cv2.COLOR_BGR2RGB)

    # swap color band
    # j2 [r, g, b] = j1[b, r, g]

    red, green, blue = 0, 1, 2

    img_j2 = img_j1[0:, 0:, [blue, red, green]]
    # img_j2 = img_j1.copy()
    # for r in range(img_j2.shape[0]):
    #     for c in range(img_j2.shape[1]):
    #         img_j2[r][c][red], img_j2[r][c][green], img_j2[r][c][blue] = img_j2[r][c][blue], img_j2[r][c][red], img_j2[r][c][green]

    plt.figure("homework 1.3", (10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_j1)
    plt.title("Image J1")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_j2)
    plt.title('Image J2')
    plt.axis('off')

    # save new image j2
    cv2.imwrite("lena512color_new.jpg", img_j2)



"""
        MAIN
"""
if __name__ == '__main__':
    homework_1_1()
    homework_1_2()
    homework_1_3()
    plt.tight_layout()
    plt.show()
