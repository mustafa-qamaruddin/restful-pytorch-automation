from striping import split_image, custom_crop, center_crop
import os
import shutil


RAW_DIR = "./RAW_data_v7"
TRAINSET_DIR = "./TRAIN_data_224"
TESTSET_DIR = "./TEST_data_224"


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def read_panorama(
        q,
        bg_cardinality,
        training_transforms,
        testing_transforms,
        scales,
        flip_ver,
        flip_hor,
        flip_both,
        angels,
        grids,
        grid_width,
        grid_stride,
        bool_center_crop,
        center_crop_windows,
        test_angels
):
    ret = []
    dirs = os.listdir(os.path.abspath(RAW_DIR))
    total_count = len(dirs)
    for dir in dirs:
        path = os.path.abspath(os.path.join(RAW_DIR, dir))
        files = os.listdir(path)

        class_train_dir = os.path.abspath(os.path.join(TRAINSET_DIR, dir))
        class_test_dir = os.path.abspath(os.path.join(TESTSET_DIR, dir))
        create_dir_if_not_exists(class_train_dir)
        create_dir_if_not_exists(class_test_dir)

        sub_total_count = len(files)
        for j in range(len(files)):
            # print("{}::{}".format(j, dir))
            file_path = os.path.abspath(os.path.join(path, files[j]))
            qj = q.enqueue(split_image,
                      args=(
                          class_train_dir,
                          class_test_dir,
                          file_path,
                          j,
                          bg_cardinality,
                          training_transforms,
                          testing_transforms,
                          scales,
                          flip_ver,
                          flip_hor,
                          flip_both,
                          angels,
                          total_count,
                          sub_total_count,
                          test_angels
                      ),
                      timeout="3000h"
                      )
            ret.append(qj.id)

            if grids:
                splits = custom_crop(
                    "dummy",
                    file_path,
                    grid_width,
                    grid_width,
                    grid_stride
                )

                for part in splits:
                    qj = q.enqueue(split_image,
                                  args=(
                                      class_train_dir,
                                      class_test_dir,
                                      os.path.abspath(part),
                                      j,
                                      bg_cardinality,
                                      training_transforms,
                                      testing_transforms,
                                      scales,
                                      flip_ver,
                                      flip_hor,
                                      flip_both,
                                      angels,
                                      total_count,
                                      sub_total_count,
                                      test_angels
                                  ),
                                  timeout="3000h"
                                  )
                    ret.append(qj.id)

            if bool_center_crop:
                for win in center_crop_windows:
                    cropped_path = center_crop(
                        "dummy",
                        file_path,
                        int(win),
                        int(win)
                    )

                    qj = q.enqueue(split_image,
                                  args=(
                                      class_train_dir,
                                      class_test_dir,
                                      os.path.abspath(cropped_path),
                                      j,
                                      bg_cardinality,
                                      training_transforms,
                                      testing_transforms,
                                      scales,
                                      flip_ver,
                                      flip_hor,
                                      flip_both,
                                      angels,
                                      total_count,
                                      sub_total_count,
                                      test_angels
                                  ),
                                  timeout="3000h"
                                  )
                    ret.append(qj.id)
    return ret


def async_split(
    q,
    bg_cardinality=2,
    training_transforms=[],
    testing_transforms=[
    "gaussian",
    "contrast",
    "brightness",
    "color"
    ],
    scales=[
    190,
    200
    ],
    flip_ver=True,
    flip_hor=True,
    flip_both=True,
    angels=[
    90,
    270
    ],
    grids=True,
    grid_width=112,
    grid_stride=56,
    bool_center_crop=True,
    center_crop_windows=[
        110
    ],
    test_angels=[45]
):
    shutil.rmtree(os.path.abspath(TRAINSET_DIR), ignore_errors=True)
    shutil.rmtree(os.path.abspath(TESTSET_DIR), ignore_errors=True)
    create_dir_if_not_exists(os.path.abspath(TRAINSET_DIR))
    create_dir_if_not_exists(os.path.abspath(TESTSET_DIR))
    return read_panorama(
        q,
        bg_cardinality,
        training_transforms,
        testing_transforms,
        scales,
        flip_ver,
        flip_hor,
        flip_both,
        angels,
        grids,
        grid_width,
        grid_stride,
        bool_center_crop,
        center_crop_windows,
        test_angels
    )
