import os
import shutil
import random
import numpy as np
import yaml
from tqdm import tqdm
import cv2 as cv
from PIL import Image

# 你原先全局的 names
names = [
    '电流', '当', '前', '上', '1', '8', '月', '组', '合', '正', '反', '向', '总', '尖', '峰', '平', '谷', '剩',
    '余', '常', '数', '1234', '电话', '房子', '阶', '梯', '透', '支', '用', '电', '量', '价', '户', '时', '间',
    '段', '金', '额', '表', '号', '圆形尖', '圆形峰', '圆形平', '圆形谷', '三角1', '三角2', 'L', 'N', '方块',
    'COS', 'VA', '元', 'kWh', '左箭头', '圆形1', '圆形2', '电池',
    '拨号', '锁', '读', '卡', '中', '成', '功', '失', '败', '请', '购', '拉', '闸', '囤', '积', '费', '率', 'T',
    '点', '象限', '无', '有', 'Ⅲ', 'V', 'A', 'B', 'C', 'O', 'S', 'fai', '需', '压', '流', '方块', 'kWAh', 'kvarh',
    'Ua', 'Ub', 'Uc', '-Ia', '-Ib', '-Ic', '信号', '电话12', '报警', '缺电1', '缺电2', '逆', '相', '序', "象限",
    "方框1", '方框2', '方框3', '方框4', '万'
]


def shuffle_multi_groups_classes(image_path, names, shuffle_groups):
    """
    多分组打乱:
    1. 读取标签文件，
    2. 找到所属分组的框裁图，
    3. 在各分组内随机 shuffle，
    4. 逐个贴回原位置，
    5. 最后标签文件坐标不变，图像中相应区域顺序对调。
    """
    label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
    if not os.path.exists(label_path):
        return image_path

    img = Image.open(image_path)
    img_width, img_height = img.size

    # 存放「非分组」(找不到分组)的标注或不打乱的标注
    remaining_lines = []

    # 收集每个分组的图像块
    group_data = {}
    for gname in shuffle_groups.keys():
        group_data[gname] = []

    # 第一次读取标签，把 bbox 分进各分组
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_idx, x_center, y_center, w, h = map(float, parts)
        cls_idx = int(cls_idx)
        class_name = names[cls_idx]

        found_group = None
        for gname, ginfo in shuffle_groups.items():
            if class_name in ginfo["items"]:
                found_group = gname
                break

        if found_group is not None:
            x1 = int((x_center - w / 2) * img_width)
            y1 = int((y_center - h / 2) * img_height)
            x2 = int((x_center + w / 2) * img_width)
            y2 = int((y_center + h / 2) * img_height)
            cropped_img = img.crop((x1, y1, x2, y2))
            group_data[found_group].append(
                (cls_idx, x_center, y_center, w, h, cropped_img)
            )
        else:
            # 没有分组就直接保持不动
            remaining_lines.append((cls_idx, x_center, y_center, w, h))

    # 对每个分组内的图像块随机打乱
    for gname in group_data:
        if len(group_data[gname]) > 1:
            random.shuffle(group_data[gname])

    # 第二次读取标签，按行顺序贴回
    # 以便把「同一个分组」的打乱图像块，按照原行顺序 one-by-one 地贴回
    new_img = img.copy()
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 记录分组里已经贴了多少个框了
    group_paste_index = {gname: 0 for gname in shuffle_groups}

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_idx, x_center, y_center, w, h = map(float, parts)
        cls_idx = int(cls_idx)
        class_name = names[cls_idx]

        found_group = None
        for gname, ginfo in shuffle_groups.items():
            if class_name in ginfo["items"]:
                found_group = gname
                break

        if found_group is not None:
            i = group_paste_index[found_group]
            # 打乱后的该分组的第 i 个图像块
            if i < len(group_data[found_group]):
                _, _, _, _, _, block_img = group_data[found_group][i]
                block_resized = block_img.resize(
                    (int(img_width * w), int(img_height * h))
                )
                x1 = int((x_center - w / 2) * img_width)
                y1 = int((y_center - h / 2) * img_height)
                new_img.paste(block_resized, (x1, y1))
                group_paste_index[found_group] += 1
        else:
            # 没有分组，不打乱
            pass

    # 保存更新后的图像
    new_img.save(image_path)

    # 标签文件坐标不变，直接原样写回
    with open(label_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)

    return image_path


# 读取 YAML 配置文件
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

class dataAugmentation:
    def __init__(
        self,
        yaml_file,
        save_dir,
        shuffle_char=True,
        increased=160,
        shuffle_groups=None  # 新增：多分组信息
    ):
        """
        :param yaml_file: YOLO 格式的 YAML 配置文件，内容包括 train 和 val 数据路径
        :param save_dir: 增广后数据要保存到的根目录，结构会自动创建为：
                         save_dir/
                           images/
                             train/
                             valid/
                             test/
                           labels/
                             train/
                             valid/
                             test/
        :param shuffle_char: 是否在增广后进行打乱字符（多分组交换）处理
        :param increased: 影响亮度调整步数的因子
        :param shuffle_groups: 多分组信息
        """
        # 加载 YAML 配置文件中的路径
        self.config = load_yaml_config(yaml_file)

        # 使用 YAML 文件中的路径
        self.train_images_dir = self.config['train']
        self.val_images_dir = self.config['val']
        self.test_images_dir = self.config.get('test', None)  # 可选的测试集路径
        self.label_dir = self.config['train'].replace('images', 'labels')  # 假设标签和图像在相同目录结构下

        self.shuffle_char = shuffle_char
        self.shuffle_groups = shuffle_groups if shuffle_groups else {}

        # 断言原始数据集必须存在
        assert os.path.exists(self.train_images_dir), (
            'make sure train images path exists and contains train subfolder'
        )
        assert os.path.exists(self.label_dir), (
            'make sure labels path exists and contains train subfolder'
        )

        # 增广后保存位置
        self.save_dir = save_dir
        self.dataset_images_dir = os.path.join(self.save_dir, 'images')
        self.dataset_labels_dir = os.path.join(self.save_dir, 'labels')

        # 对应 train/valid/test 子目录
        self.images_train_dir = os.path.join(self.dataset_images_dir, 'train')
        self.images_valid_dir = os.path.join(self.dataset_images_dir, 'valid')
        self.images_test_dir = os.path.join(self.dataset_images_dir, 'test')
        self.labels_train_dir = os.path.join(self.dataset_labels_dir, 'train')
        self.labels_valid_dir = os.path.join(self.dataset_labels_dir, 'valid')
        self.labels_test_dir = os.path.join(self.dataset_labels_dir, 'test')

        # 增强因子
        self.offset_step = 1.6 / increased

        # 如果保存地址已存在，就根据已经存在的文件名确定 name_cnt 的起点
        if os.path.exists(self.dataset_images_dir) and os.path.exists(self.dataset_labels_dir):
            self.name_cnt = 0
            self.get_name_cnt()
        else:
            # 否则创建
            os.makedirs(self.images_train_dir, exist_ok=True)
            os.makedirs(self.images_valid_dir, exist_ok=True)
            os.makedirs(self.images_test_dir, exist_ok=True)
            os.makedirs(self.labels_train_dir, exist_ok=True)
            os.makedirs(self.labels_valid_dir, exist_ok=True)
            os.makedirs(self.labels_test_dir, exist_ok=True)
            self.name_cnt = 0

        # 临时保存增广过程中的一些信息
        self.cur_augmented_image = None
        self.cur_original_image = None
        self.cur_label_file_path = None

        # 统计信息
        self.train = 0
        self.valid = 0
        self.test = 0
        self.images_count = 0

    def generate_new_yaml(self):
        """
        在增广后的数据目录（self.save_dir）下生成新的 yaml 文件，并返回其路径。
        """
        # 构建新的yaml内容
        # 这里的 train/val/test 指向 self.save_dir 下的 images
        # 你也可以根据需要添加 'names', 'nc' 等字段
        new_yaml = {}
        new_yaml["train"] = os.path.join(self.save_dir, "images", "train").replace("\\", "/")
        new_yaml["val"] = os.path.join(self.save_dir, "images", "valid").replace("\\", "/")

        # 如果 test 目录里确实有数据，也可以加上
        test_images_path = os.path.join(self.save_dir, "images", "test")
        if os.path.exists(test_images_path) and len(os.listdir(test_images_path)) > 0:
            new_yaml["test"] = test_images_path.replace("\\", "/")

        # 如果你想写入类别数或类别名，可以根据自己的逻辑
        # 例如： new_yaml["nc"] = len(names)
        # 例如： new_yaml["names"] = names
        new_yaml["nc"] = self.config['nc']
        new_yaml["names"] = self.config['names']

        # 目标文件写在 save_dir 里
        new_yaml_path = os.path.join(self.save_dir, "data.yaml")

        with open(new_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(new_yaml, f, sort_keys=False, allow_unicode=True)

        return new_yaml_path

    def start(self, progress_callback=None):
        """
        扫描 train/val/test 下所有 JPG 文件，
        调用回调函数或发射信号来汇报进度。
        """
        subsets = ['train', 'valid']
        if self.test_images_dir:
            subsets.append('test')

        # 先统计总图像数（原图数量），准备给进度条用
        self.total_images_to_process = 0
        for subset in subsets:
            images_sub_dir = self._get_subset_image_dir(subset)  # 下面我们自定义了一个小函数
            img_files = os.listdir(images_sub_dir)
            # 仅统计真实的图像文件
            for img_name in img_files:
                ext = os.path.splitext(img_name)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.total_images_to_process += 1

        processed_count = 0

        # 开始执行真正的增强
        for subset in subsets:
            images_sub_dir = self._get_subset_image_dir(subset)
            labels_sub_dir = self.label_dir  # 假设和原逻辑一致
            self._set_paths(subset)          # 设置 self.image_path / self.label_path 等

            img_files = os.listdir(images_sub_dir)
            for img_name in img_files:
                ext = os.path.splitext(img_name)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue

                base_name = os.path.splitext(img_name)[0]
                label_name = base_name + '.txt'

                self.cur_label_file_path = os.path.join(labels_sub_dir, label_name)
                img_path = os.path.join(images_sub_dir, img_name)

                # 读原图(灰度)
                self.cur_original_image = cv.imread(img_path, 0)
                if self.cur_original_image is None:
                    continue

                # ====== 正常的保存&增强操作 ======
                self.save_one(isOriginal=True)
                self.images_count += 1

                # 生成增广图
                offsets = np.arange(0.4, 1.2, self.offset_step)
                for offset in offsets:
                    # 第一种增广
                    self.cur_augmented_image = cv.convertScaleAbs(
                        self.cur_original_image, alpha=offset, beta=0
                    )
                    self.save_one()

                    # 第二种增广
                    temp = cv.normalize(
                        np.log(self.cur_original_image.astype(np.float32) + 1),
                        None,
                        0, 255,
                        cv.NORM_MINMAX, cv.CV_8U
                    )
                    self.cur_augmented_image = cv.convertScaleAbs(temp, alpha=offset, beta=0)
                    self.save_one()
                # ====== 正常的保存&增强操作 ======

                # 每处理完一张“原图”则进度+1
                processed_count += 1
                if progress_callback:  # 如果传了回调，就调用
                    progress_callback(processed_count, self.total_images_to_process, img_path)

        self.summary()

    def _get_subset_image_dir(self, subset):
        if subset == 'train':
            return self.train_images_dir
        elif subset == 'valid':
            return self.val_images_dir
        else:
            return self.test_images_dir

    def _set_paths(self, subset):
        """
        根据 subset(train/valid/test)，设置
        self.image_path, self.label_path 用于保存增强后的数据
        """
        if subset == 'train':
            self.image_path = self.images_train_dir
            self.label_path = self.labels_train_dir
            self.train += 1
        elif subset == 'valid':
            self.image_path = self.images_valid_dir
            self.label_path = self.labels_valid_dir
            self.valid += 1
        else:
            self.image_path = self.images_test_dir
            self.label_path = self.labels_test_dir
            self.test += 1

    def get_name_cnt(self):
        """
        若 save_dir 下已存在一些文件，就计算它们的文件名(数字)的最大值 + 1，作为新的开始。
        """
        for root, dirs, files in os.walk(self.save_dir):
            for file in files:
                base_name = os.path.splitext(file)[0]
                try:
                    self.name_cnt = max(self.name_cnt, int(base_name))
                except ValueError:
                    # 文件名不是纯数字就忽略
                    continue
        self.name_cnt += 1

    def save_one(self, isOriginal=False):
        """
        将当前 self.cur_augmented_image 或 self.cur_original_image
        保存到 self.image_path 中，并复制标签文件到对应位置。
        如果开启多分组打乱，则调用 shuffle_multi_groups_classes。
        """
        # 1. 保存图像
        if isOriginal:
            cv.imwrite(
                os.path.join(self.image_path, f'{self.name_cnt}.jpg'),
                self.cur_original_image
            )
        else:
            self.do_augment_ops()  # 可选的一些二次处理
            cv.imwrite(
                os.path.join(self.image_path, f'{self.name_cnt}.jpg'),
                self.cur_augmented_image
            )

        # 2. 复制标签
        dst_label = os.path.join(self.label_path, f'{self.name_cnt}.txt')
        try:
            shutil.copy(self.cur_label_file_path, dst_label)
        except FileNotFoundError:
            # 如果没有标签文件，就略过
            pass

        # 3. 如果需要在这里对已保存的图像再进行“多分组字符打乱”，则调用
        if self.shuffle_char and self.shuffle_groups:
            new_img_path = os.path.join(self.image_path, f'{self.name_cnt}.jpg')
            # 这里需要你事先实现 shuffle_multi_groups_classes
            shuffle_multi_groups_classes(
                image_path=new_img_path,
                names=names,
                shuffle_groups=self.shuffle_groups
            )

        # 打印日志
        print(f'Image:{self.name_cnt}.jpg -> Path:{self.image_path} | Label:{self.name_cnt}.txt -> {self.label_path}')

        # 自增
        self.name_cnt += 1

    def do_augment_ops(self):
        """
        对 self.cur_augmented_image 做一些随机处理，如直方图均衡化、模糊等。
        可根据需求随时增减。
        """
        if random.random() < 0.5:
            self.cur_augmented_image = cv.equalizeHist(self.cur_augmented_image)
        if random.random() < 0.5:
            self.cur_augmented_image = cv.blur(self.cur_augmented_image, (3, 3))
        if random.random() < 0.5:
            self.cur_augmented_image = cv.medianBlur(self.cur_augmented_image, 3)

    def summary(self):
        """
        输出统计信息。
        """
        print(f'{self.train} original images processed in train')
        print(f'{self.valid} original images processed in valid')
        print(f'{self.test} original images processed in test')
        # 如果你想统计“增广后总计多少图”也可自行增加统计。
        print(f'Total original images: {self.train + self.valid + self.test}')
        print(f'After augmentation, total images in new dataset: {self.name_cnt - 1}')


def merge_yolo_datasets(source_dirs, target_dir):
    """
    仅移动 source_dirs 下的 images 和 labels 目录内的文件，并确保文件名匹配。
    """
    target_images = os.path.join(target_dir, "images")
    target_labels = os.path.join(target_dir, "labels")
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)

    for source_dir in source_dirs:
        images_dir = os.path.join(source_dir, "images")
        labels_dir = os.path.join(source_dir, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(
                f"Warning: {source_dir} does not contain images or labels directory, skipping.")
            continue

        image_files = set(os.listdir(images_dir))
        label_files = set(os.listdir(labels_dir))

        common_files = {
            f for f in image_files
            if os.path.splitext(f)[0] + ".txt" in label_files
        }

        for file in common_files:
            img_source = os.path.join(images_dir, file)
            lbl_source = os.path.join(
                labels_dir, os.path.splitext(file)[0] + ".txt")

            img_target = os.path.join(target_images, file)
            lbl_target = os.path.join(
                target_labels, os.path.splitext(file)[0] + ".txt")

            shutil.copy2(img_source, img_target)
            shutil.copy2(lbl_source, lbl_target)
            print(f"Copied: {img_source} -> {img_target}")
            print(f"Copied: {lbl_source} -> {lbl_target}")

from PyQt5 import QtCore

class AugWorker(QtCore.QObject):
    """
    用于在子线程中执行 dataAugmentation，
    通过信号把进度、中途出错、完成等信息发回主线程。
    """
    progressChanged = QtCore.pyqtSignal(int, int, str)  # 当前进度，最大值，当前处理的图像路径
    finished = QtCore.pyqtSignal()                      # 完成
    errorOccurred = QtCore.pyqtSignal(str)              # 出错时发送错误信息

    def __init__(self, aug_instance:dataAugmentation, parent=None):
        """
        :param aug_instance: 传入一个已经配置好的 dataAugmentation 对象
        """
        super().__init__(parent)
        self.aug = aug_instance
        self._is_cancelled = False  # 如果想支持中途取消，可以加个标记

    @QtCore.pyqtSlot()
    def run(self):
        """
        线程开始工作时会自动调用这里。
        """
        try:
            # 调用 aug.start，并传递一个回调，用于发射 progressChanged 信号
            self.aug.start(progress_callback=self.on_progress)
            # 完成后发射 finished 信号
            self.finished.emit()
            self.aug.generate_new_yaml()
        except Exception as e:
            self.errorOccurred.emit(str(e))

    def on_progress(self, current, total, img_path):
        """
        这是给 dataAugmentation.start() 调用的回调函数。
        """
        if self._is_cancelled:
            raise RuntimeError("数据增强被取消。")
        self.progressChanged.emit(current, total, img_path)

    def cancel(self):
        """
        如果想在主线程主动取消，可以调用此方法。
        """
        self._is_cancelled = True


if __name__ == "__main__":

    # 假设你的 shuffle_groups 结构形如：
    example_shuffle_groups = {
        "group1": {
            "color": "#FF0000",  # 仅供UI显示，这里不会真正用到
            "items": ["当", "前", "电", "量"]  # 该组包含这些字符
        },
        "group2": {
            "color": "#00FF00",
            "items": ["房子", "电话", "时", "间"]
        },
        "default": {
            "color": "#CCCCCC",
            "items": ["剩", "余", "金", "额", "表", "号", ...]  # 假设其余都放在default
        }
    }

    source_dirs = [
        r'F:\德创\全图标注\完整图片',
        r'F:\德创\全图标注'
    ]
    increased = [30, 1]
    target_dir = r'F:\毕业论文实验\粗粒度模型'

    for i in range(len(source_dirs)):
        dataAugmentation(
            source_dirs[i],
            target_dir,
            test_ratio=0.1,
            increased=increased[i],
            shuffle_char=True,
            shuffle_groups=example_shuffle_groups  # 传入我们需要的分组信息
        )

    # merge_yolo_datasets(source_dirs, target_dir)
    # print("Dataset merging completed!")
