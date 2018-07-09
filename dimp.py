import numpy as np
import cv2
from time import time
import os
from base64 import b64decode, b64encode


def readb64(base64_string):
    prefix = "data:image/jpg;base64,"
    base64_string = base64_string.replace(prefix, "")

    prefix = "data:image/png;base64,"
    base64_string = base64_string.replace(prefix, "")

    prefix = "data:image/jpeg;base64,"
    base64_string = base64_string.replace(prefix, "")
    decoded_image = b64decode(base64_string)

    if not os.path.exists("./repository"):
        os.mkdir("./repository")

    save_file_name = "./repository/{}_{}.jpg".format(
        np.random.randint(0, 1e4),
        int(time())
    )

    with open(save_file_name, "wb") as f:
        f.write(decoded_image)

    return cv2.imread(save_file_name)


def interactive_segmentation(cv_im, box, ean):
    w, h, d = cv_im.shape
    if w != 224 or h != 224:
        raise ValueError('Image must be resized to 224x224x3 using bicubic interpolation')

    mask = np.zeros(cv_im.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(cv_im, mask, box, bgdModel, fgdModel, 15, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # alpha blending 4th channel
    b_channel, g_channel, r_channel = cv2.split(cv_im)
    img = cv2.merge((b_channel, g_channel, r_channel, mask2 * 255))

    if not os.path.exists("dumb"):
        os.mkdir("dumb")

    sub_dir = os.path.join("dumb", ean.zfill(15))
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    file_path = os.path.join(sub_dir, "{}{}{}.png".format(int(time()),
                                                          np.random.randint(0, 1e3),
                                                          np.random.randint(1e3, 1e6)
                                                          )
                             )

    cv2.imwrite(file_path, img)

    return file_path


def encode64_im(file_path):
    with open(file_path, "rb") as image_file:
        return b64encode(image_file.read()).decode()


def wrap_segmentation(base64_im, box, ean):
    return encode64_im(interactive_segmentation(readb64(base64_im), box, ean))
