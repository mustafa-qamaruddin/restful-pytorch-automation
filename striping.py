import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from time import time
from random import shuffle
import random
from rq import get_current_job


MP_CARDINALITY = 2
BG_CARDINALITY = 5
TGT_SIZE = (512, 512)
RAW_DIR = "./RAW_data_v7"
TRAINSET_DIR = "./TRAIN_data_224"
TESTSET_DIR = "./TEST_data_224"

bgs = []
for bg in os.listdir("./bgs"):
    bgs.append(os.path.join("./bgs", bg))
shuffle(bgs)


def verify_image(path):
    if not os.path.exists(path):
        print("File Write Failed @ {}".format(path))
        return
    try:
        Image.open(path).load()
    except:
        try:
            os.remove(path)
            print("Corrupt File @ {}".format(path))
        except:
            print("File does not exist @ {}".format(path))
            print("Corrupt File @ {}".format(path))


def non_overlap_rand_bg(save_dir):
    sz = int(len(bgs) * 0.5)
    if save_dir.find("TRAIN") != -1:
        low = 0  # BG_CARDINALITY
        high = sz
    else:
        low = sz
        high = len(bgs)
    return np.random.randint(low, high)


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def merge_bg_fg(bg_idx, fg):
    bg = Image.open(bgs[bg_idx]).convert("RGBA")
    bg = bg.resize(TGT_SIZE)
    ret = Image.new("RGBA", bg.size)
    ret.paste(bg, (0, 0), bg)
    # Translation
    random_x = random_y = 0
    if TGT_SIZE[0] - fg.size[0] != 0 and TGT_SIZE[1] - fg.size[1] != 0:
        random_x = np.random.randint(0, abs(TGT_SIZE[0] - fg.size[0]))
        random_y = np.random.randint(0, abs(TGT_SIZE[1] - fg.size[1]))
    ret.paste(fg, (random_x, random_y), fg)
    return ret


def aux_gen_distortions(save_dir,
                        file_path,
                        timestamp,
                        rand_int,
                        scale,
                        bg_idx,
                        variance=None,
                        flip_ver=True,
                        flip_hor=True,
                        flip_both=True,
                        angels=[90, 180, 270]
                        ):
    # load image again for forking can not pickle pil images
    # image load
    temp_im = Image.open(file_path)
    temp_im.thumbnail((scale, scale), Image.ANTIALIAS)
    rsz = temp_im.convert("RGBA")

    if variance == "gaussian":
        rsz = rsz.filter(ImageFilter.GaussianBlur(2))
    elif variance == "brightness":
        enhancer_brightness = ImageEnhance.Brightness(rsz)
        factors = [0.3, 0.4, 0.5, 0.6]
        rsz = enhancer_brightness.enhance(random.choice(factors))
    elif variance == "contrast":
        enhancer_contrast = ImageEnhance.Contrast(rsz)
        factors = [0.3, 0.4, 0.5, 0.6]
        rsz = enhancer_contrast.enhance(random.choice(factors))
    elif variance == "color":
        enhancer_color = ImageEnhance.Color(rsz)
        factors = [0.3, 0.4, 0.5, 0.6]
        rsz = enhancer_color.enhance(random.choice(factors))

    # for i in range(len(bgs)):
    final = merge_bg_fg(bg_idx=bg_idx, fg=rsz)
    file_name = "{}{}{}{}.png".format(rand_int, timestamp, int(time()), scale)
    file_path = os.path.join(save_dir, file_name)
    final.save(file_path)
    verify_image(file_path)

    # FLIP
    if flip_hor:
        flip_hor = rsz.transpose(method=Image.FLIP_LEFT_RIGHT)
        final = merge_bg_fg(bg_idx=non_overlap_rand_bg(save_dir), fg=flip_hor)
        file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), 10, scale)
        file_path = os.path.join(save_dir, file_name)
        final.save(file_path)
        verify_image(file_path)

    if flip_ver:
        flip_ver = rsz.transpose(method=Image.FLIP_TOP_BOTTOM)
        final = merge_bg_fg(bg_idx=non_overlap_rand_bg(save_dir), fg=flip_ver)
        file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), 20, scale)
        file_path = os.path.join(save_dir, file_name)
        final.save(file_path)
        verify_image(file_path)

    if flip_both:
        flip_both = rsz.transpose(method=Image.TRANSPOSE)
        final = merge_bg_fg(bg_idx=non_overlap_rand_bg(save_dir), fg=flip_both)
        file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), 30, scale)
        file_path = os.path.join(save_dir, file_name)
        final.save(file_path)
        verify_image(file_path)

    # ROTATE
    for angle in angels:
        rotated = rsz.rotate(angle, expand=1)
        final = merge_bg_fg(bg_idx=non_overlap_rand_bg(save_dir), fg=rotated)
        file_name = "{}{}{}{}{}.png".format(rand_int, timestamp, int(time()), angle, scale)
        file_path = os.path.join(save_dir, file_name)
        final.save(file_path)
        verify_image(file_path)


def split_image(train_dir,
                test_dir,
                src,
                i,
                bg_cardinality,
                training_transforms,
                testing_transforms,
                scales,
                flip_ver,
                flip_hor,
                flip_both,
                angels,
                total_count=10,
                sub_total_count=0,
                test_angels=[45]
                ):
    # image load
    im = Image.open(src)
    imgwidth, imgheight = im.size

    counter = i + 1

    view = im.crop((0, 0, imgwidth, imgheight))

    # NORMAL SCALE
    view.thumbnail(TGT_SIZE, Image.ANTIALIAS)
    rsz = view.convert("RGBA")

    # save image
    file_name = "{}{}{}{}{}.png".format(i, int(time()), i + 1, 0, 1)
    file_path = os.path.join(train_dir, file_name)
    rsz.save(file_path)
    verify_image(file_path)

    for x in range(bg_cardinality):
        rand_int = np.random.randint(0, 1e6)
        timestamp = int(time())

        # NORMAL
        sz = TGT_SIZE[0]
        if x % 2 == 0:
            aux_gen_distortions(
                train_dir,
                src,
                timestamp,
                rand_int,
                sz,
                x,
                None,
                flip_ver,
                flip_hor,
                flip_both,
                angels
            )
            for trans in training_transforms:
                aux_gen_distortions(
                    train_dir,
                    src,
                    timestamp,
                    rand_int,
                    sz,
                    non_overlap_rand_bg(test_dir),
                    trans,
                    flip_ver,
                    flip_hor,
                    flip_both,
                    angels
                )
        else:
            aux_gen_distortions(
                test_dir,
                src,
                timestamp,
                rand_int,
                sz,
                x,
                None,
                flip_ver,
                flip_hor,
                flip_both,
                test_angels
            )
            for trans in testing_transforms:
                aux_gen_distortions(
                    test_dir,
                    src,
                    timestamp,
                    rand_int,
                    sz,
                    non_overlap_rand_bg(test_dir),
                    trans,
                    flip_ver,
                    flip_hor,
                    flip_both,
                    test_angels
                )

        if TGT_SIZE[0] < 64:
            return

        # SCALES
        for sz in scales:
            if x % 2 == 0:
                aux_gen_distortions(
                    train_dir,
                    src,
                    timestamp,
                    rand_int,
                    sz,
                    x,
                    None,
                    flip_ver,
                    flip_hor,
                    flip_both,
                    angels
                )
                for trans in training_transforms:
                    aux_gen_distortions(
                        train_dir,
                        src,
                        timestamp,
                        rand_int,
                        sz,
                        non_overlap_rand_bg(test_dir),
                        trans,
                        flip_ver,
                        flip_hor,
                        flip_both,
                        angels
                    )
            else:
                aux_gen_distortions(
                    test_dir,
                    src,
                    timestamp,
                    rand_int,
                    sz,
                    x,
                    None,
                    flip_ver,
                    flip_hor,
                    flip_both,
                    test_angels
                )
                for trans in testing_transforms:
                    aux_gen_distortions(
                        test_dir,
                        src,
                        timestamp,
                        rand_int,
                        sz,
                        non_overlap_rand_bg(test_dir),
                        trans,
                        flip_ver,
                        flip_hor,
                        flip_both,
                        test_angels
                    )

        counter += 1
        pstr = "{:.2f}".format(x/(len(bgs) * sub_total_count * total_count))
        # @todo log to file
        job = get_current_job()
        job.meta['progress'] = pstr
        job.save_meta()


def custom_crop(path, input, height, width, stride):
    if not os.path.exists(path):
        os.mkdir(path)
    im = Image.open(input)
    imgwidth, imgheight = im.size
    ret = []
    for i in range(0, imgheight, stride):
        for j in range(0, imgwidth, stride):
            box = (j, i, j + width, i + height)
            a = im.crop(box)
            file_path = os.path.join(
                path, "{}_{}.png".format(
                    i,
                    j
                )
            )
            a.save(file_path)
            ret.append(file_path)
    return ret


def center_crop(path, input, crop_width, crop_height):
    if not os.path.exists(path):
        os.mkdir(path)
    im = Image.open(input)
    imgwidth, imgheight = im.size

    center_x = imgwidth // 2
    center_y = imgheight // 2

    x1 = center_x - (crop_width // 2)
    y1 = center_y - (crop_height // 2)

    box = (x1, y1, x1 + crop_width, y1 + crop_height)
    a = im.crop(box)
    file_path = os.path.join(
        path, "{}_{}.png".format(
            x1,
            y1
        )
    )
    a.save(file_path)

    return file_path


# def main():
#     create_dir_if_not_exists(TRAINSET_DIR)
#     create_dir_if_not_exists(TESTSET_DIR)
#     read_panorama(
#         bg_cardinality=2,
#         training_transforms=[],
#         testing_transforms=[
#             "gaussian",
#             "contrast",
#             "brightness",
#             "color"
#         ],
#         scales=[
#             190,
#             200
#         ],
#         flip_ver=True,
#         flip_hor=True,
#         flip_both=True,
#         angels=[
#             90,
#             270
#         ],
#         grids=True,
#         grid_width=112,
#         grid_stride=56
#     )
#     print("TERMINATED")


# if __name__ == '__main__':
#     main()
