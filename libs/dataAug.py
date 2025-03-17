# Example usage: data_aug = dataAugmentation(file_dir, save_dir)

# ├── file_dir
#     └── images
#         └── all image files...
#     └── labels
#         └── all labels files...
# └── save_dir
#     └──

import os
import shutil
from random import shuffle
from PIL import Image
import cv2 as cv
import random

import numpy as np
from tqdm import tqdm

names = ['电流', '当', '前', '上', '1', '8', '月', '组', '合', '正', '反', '向', '总', '尖', '峰', '平', '谷', '剩',
         '余', '常', '数', '1234', '电话', '房子', '阶', '梯', '透', '支', '用', '电', '量', '价', '户', '时', '间',
         '段', '金', '额', '表', '号', '圆形尖', '圆形峰', '圆形平', '圆形谷', '三角1', '三角2', 'L', 'N', '方块',
         'COS', 'VA', '元', 'kWh', '左箭头', '圆形1', '圆形2', '电池',
         '拨号', '锁', '读', '卡', '中', '成', '功', '失', '败', '请', '购', '拉', '闸', '囤', '积', '费', '率', 'T',
         '点', '象限', '无', '有', 'Ⅲ', 'V', 'A', 'B', 'C', 'O', 'S', 'fai', '需', '压', '流', '方块', 'kWAh', 'kvarh',
         'Ua', 'Ub', 'Uc', '-Ia', '-Ib', '-Ic', '信号', '电话12', '报警', '缺电1', '缺电2', '逆', '相', '序', "象限",
         "方框1", '方框2', '方框3', '方框4', '万']
selected = ['当', '前', '上', '月', '组', '合', '正', '反', '向',
            '总', '尖', '峰', '平', '谷', '剩', '余', '常', '数', '阶', '梯', '透', '支', '用', '电', '量', '价', '户',
            '时', '间', '段', '金', '额', '表', '号', '读', '卡', '中', '成', '功', '失', '败', '请', '购', '拉', '闸',
            '囤', '积', '逆', '相', '序', '房子', '电话', 'L', 'N', '圆形1', '圆形2', '元', '圆形尖', '圆形峰',
            '圆形平', '圆形谷', "方框1", '方框2', '方框3', '方框4', '万', 'fai', '需', '压', '流']


def shuffle_selected_classes(image_path, names=names, selected=selected):
    label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
    img = Image.open(image_path)
    img_width, img_height = img.size

    selected_lines = []
    new_positions_images = []
    original_positions = []
    new_lines = []

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls, x_center, y_center, width, height = map(float, parts)

            # 如果这个类别是选定的类别，我们将其添加到要打乱的列表中
            if names[int(cls)] in selected:
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)
                cropped_img = img.crop((x1, y1, x2, y2))
                selected_lines.append((cls, x_center, y_center, width, height))
                new_positions_images.append(
                    (cls, width, height, cropped_img))

            else:
                new_lines.append(
                    (cls, x_center, y_center, width, height))

    shuffle(new_positions_images)

    # 创建一个新的空白图像
    new_img = Image.open(image_path)

    # 将选定的类别粘贴到新的位置
    for i, (_, old_x_center, old_y_center, old_width, old_height) in enumerate(selected_lines):
        cls, width, height, new_images = new_positions_images[i]
        new_x1 = int((old_x_center - old_width / 2) * img_width)
        new_y1 = int((old_y_center - old_height / 2) * img_height)
        new_images = new_images.resize(
            (int(img_width * old_width), int(img_height * old_height)))
        new_img.paste(new_images, (new_x1, new_y1))
        new_lines.append(
            (cls, old_x_center, old_y_center, old_width, old_height))

    # 保存新的图像
    new_img_path = image_path
    new_img.save(new_img_path)

    # 写入新的位置和标签到YOLO格式的标签文件中
    with open(label_path, 'w') as f:
        # 先写入未选定的类别的原始位置和标签
        for cls, x_center, y_center, width, height in new_lines:
            f.write(f"{int(cls)} {x_center} {y_center} {width} {height}\n")

    return new_img_path


class dataAugmentation:
    def __init__(self, file_dir, save_dir, shuffle_char=True, valid_ratio=0.2, test_ratio=0.1, increased=160):
        self.file_dir = file_dir
        self.images_dir = os.path.join(self.file_dir, 'images')
        self.label_dir = os.path.join(self.file_dir, 'labels')
        self.shuffle_char = shuffle_char
        assert os.path.exists(self.images_dir), 'make sure there is a folder named "images" which contains all images ' \
                                                'in the file_dir '
        assert os.path.exists(
            self.images_dir), 'make sure there is a folder named "labels" which contains all label files ' \
                              'in the file_dir '

        self.save_dir = save_dir
        self.dataset_images_dir = os.path.join(self.save_dir, 'images')
        self.dataset_labels_dir = os.path.join(self.save_dir, 'labels')
        self.images_test_dir = os.path.join(self.dataset_images_dir, 'test')
        self.images_valid_dir = os.path.join(self.dataset_images_dir, 'valid')
        self.images_train_dir = os.path.join(self.dataset_images_dir, 'train')
        self.labels_test_dir = os.path.join(self.dataset_labels_dir, 'test')
        self.labels_valid_dir = os.path.join(self.dataset_labels_dir, 'valid')
        self.labels_train_dir = os.path.join(self.dataset_labels_dir, 'train')
        self.offset_step = 1.6 / increased
        self.image_path = None
        self.labelp_path = None

        # 如果保存地址存在，说明是在原数据集基础上添加，要修改命名计数为已有的最大值
        if os.path.exists(self.dataset_images_dir) and os.path.exists(self.dataset_labels_dir):
            self.name_cnt = 0
            self.get_name_cnt()
        else:
            os.makedirs(self.dataset_images_dir)
            os.makedirs(self.dataset_labels_dir)
            os.makedirs(self.images_test_dir)
            os.makedirs(self.images_valid_dir)
            os.makedirs(self.images_train_dir)
            os.makedirs(self.labels_train_dir)
            os.makedirs(self.labels_test_dir)
            os.makedirs(self.labels_valid_dir)
            self.name_cnt = 0
        self.cur_augmented_image = None
        self.cur_original_image = None
        self.cur_label_file_path = None
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.test = 0
        self.valid = 0
        self.train = 0
        self.images_count = 0
        assert valid_ratio + test_ratio <= 1
        self.image_increment()
        self.summary()

    def image_increment(self):
        for root, dirs, files in os.walk(self.images_dir):
            for images_name in tqdm(files):
                base_name = os.path.splitext(images_name)[0]
                label_name = base_name + '.txt'
                self.cur_label_file_path = os.path.join(self.label_dir, label_name)
                offsets = np.arange(0.4, 1.2, self.offset_step)
                self.cur_original_image = cv.imread(os.path.join(root, images_name), 0)
                random_num = random.random()
                if random_num < self.test_ratio:
                    self.image_path = self.images_test_dir
                    self.label_path = self.labels_test_dir
                    self.test += 1
                elif self.test_ratio < random_num < self.test_ratio + self.valid_ratio:
                    self.image_path = self.images_valid_dir
                    self.label_path = self.labels_valid_dir
                    self.valid += 1
                else:
                    self.image_path = self.images_train_dir
                    self.label_path = self.labels_train_dir
                    self.train += 1
                self.save_one(isOriginal=True)
                self.images_count += 1

                for offset in offsets:
                    self.cur_augmented_image = cv.convertScaleAbs(self.cur_original_image, alpha=offset, beta=0)

                    self.save_one()
                    self.cur_augmented_image = cv.convertScaleAbs(
                        cv.normalize(np.log(self.cur_original_image.astype(np.float32) + 1), None, 0, 255,
                                     cv.NORM_MINMAX, cv.CV_8U), alpha=offset, beta=0)
                    self.save_one()

    def get_name_cnt(self):
        for root, dirs, files in os.walk(self.save_dir):
            for file in files:
                base_name = os.path.splitext(file)[0]
                try:
                    self.name_cnt = max(self.name_cnt, int(base_name))
                except ValueError:
                    continue
        self.name_cnt += 1

    def save_one(self, isOriginal=False):
        if isOriginal:
            cv.imwrite(os.path.join(self.image_path, '%d.jpg' % self.name_cnt), self.cur_original_image)
        else:
            self.do()
            cv.imwrite(os.path.join(self.image_path, '%d.jpg' % self.name_cnt), self.cur_augmented_image)
        try:
            shutil.copy(self.cur_label_file_path, os.path.join(self.label_path, '%d.txt' % self.name_cnt))
        except FileNotFoundError:
            pass
        img_path = os.path.join(self.image_path, '%d.jpg' % self.name_cnt)
        if self.shuffle_char:
            shuffle_selected_classes(image_path=img_path)
        print(f'Image:{self.name_cnt}.jpg -> Path:{img_path} and Label:{self.name_cnt}.txt -> Path:{self.label_path}')
        self.name_cnt += 1

    def do(self):
        if random.random() < 0.5:
            self.cur_augmented_image = cv.equalizeHist(self.cur_augmented_image)
        if random.random() < 0.5:
            self.cur_augmented_image = cv.blur(self.cur_augmented_image, (3, 3))
        if random.random() < 0.5:
            self.cur_augmented_image = cv.medianBlur(self.cur_augmented_image, 3)

    def summary(self):
        print(f'{self.test} test data generated\n{self.valid} valid data generated\n{self.train} train data generated')
        print(f'{self.train + self.valid + self.test - self.images_count} images has been increased')


def merge_yolo_datasets(source_dirs, target_dir):
    """
    仅移动 source_dirs 下的 images 和 labels 目录内的文件，并确保文件名匹配。
    :param source_dirs: list，包含多个 YOLO 数据集路径
    :param target_dir: str，目标数据集存放路径
    """
    target_images = os.path.join(target_dir, "images")
    target_labels = os.path.join(target_dir, "labels")
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)

    for source_dir in source_dirs:
        images_dir = os.path.join(source_dir, "images")
        labels_dir = os.path.join(source_dir, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Warning: {source_dir} does not contain images or labels directory, skipping.")
            continue

        image_files = set(os.listdir(images_dir))
        label_files = set(os.listdir(labels_dir))

        common_files = {f for f in image_files if os.path.splitext(f)[0] + ".txt" in label_files}

        for file in common_files:
            img_source = os.path.join(images_dir, file)
            lbl_source = os.path.join(labels_dir, os.path.splitext(file)[0] + ".txt")

            img_target = os.path.join(target_images, file)
            lbl_target = os.path.join(target_labels, os.path.splitext(file)[0] + ".txt")

            shutil.copy2(img_source, img_target)
            shutil.copy2(lbl_source, lbl_target)
            print(f"Copied: {img_source} -> {img_target}")
            print(f"Copied: {lbl_source} -> {lbl_target}")


if __name__ == "__main__":
    source_dirs = [
        r'F:\毕业论文实验\银河数据集_裁剪',
        r'F:\毕业论文实验\1ph_char_13',
        r'F:\毕业论文实验\1ph_char_20',
        r'F:\毕业论文实验\3ph_char_20',
        r'F:\毕业论文实验\3ph_char_13',
        r'F:\毕业论文实验\20版三角2识别失败_裁剪',
        r'F:\毕业论文实验\20240305'
    ]

    source_dirs = [
        r'F:\德创\全图标注\完整图片',
        r'F:\德创\全图标注'
    ]

    # increased = [8,8,8,16*0.8,16*0.8,8,2]
    increased = [30, 1]

    target_dir = r'F:\毕业论文实验\粗粒度模型'  # 合并后的目标文件夹

    for i in range(len(source_dirs)):
        dataAugmentation(source_dirs[i], target_dir, test_ratio=0.1, increased=increased[i])

    # merge_yolo_datasets(source_dirs, target_dir)
    # print("Dataset merging completed!")
