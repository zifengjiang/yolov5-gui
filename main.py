from functools import lru_cache

import yaml
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QHBoxLayout, QRadioButton, QComboBox, QLineEdit

from libs.LineWidget import LineWidget
from libs.model_size_selector import ModelSizeSelector
from libs.settings import Settings
from libs.resources import *
import subprocess
import os
import sys
import tempfile


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings('.yolov5-gui.pkl')
        self.settings.load()
        self.data_line = LineWidget('数据集配置路径', self.settings.get('data_config_path'),
                                    "YoloV5 Data Config (%s)" % ' '.join(['*.yaml']))
        self.model_cfg_line = LineWidget('模型配置路径', self.settings.get('model_config_path'),
                                         "YoloV5 Model Config (%s)" % ' '.join(['*.yaml']))
        self.pretrained_line = LineWidget('预训练模型路径', self.settings.get('pretrained_model_path'),
                                          "YoloV5 Pretrained Model (%s)" % ' '.join(['*.pt']))
        self.train_py_line = LineWidget('训练脚本路径', self.settings.get('train_script_path'),
                                        "Python Script (%s)" % ' '.join(['*.py']))
        self.save_dir_line = LineWidget('训练结果保存路径', self.settings.get('save_dir', os.path.expanduser('~')))
        # 创建一个横向布局，左边是文本，右边是一个下拉框，用于选择Conda环境
        self.conda_env_combobox = QComboBox(self)
        # 获取所有Conda环境
        self.envs = get_all_conda_envs()
        if self.envs:
            for env in self.envs:
                self.conda_env_combobox.addItem(env[0])
        conda_env_hbox = QHBoxLayout()
        conda_env_hbox.addWidget(QLabel('Conda环境'))
        conda_env_hbox.addWidget(self.conda_env_combobox)
        self.conda_env_line = QWidget()
        conda_env_hbox.setAlignment(QtCore.Qt.AlignLeft)
        self.conda_env_line.setLayout(conda_env_hbox)
        
        name_hbox = QHBoxLayout()
        name_hbox.addWidget(QLabel('项目名称'))
        self.name_line_edit = QLineEdit()
        self.name_line = QWidget()
        name_hbox.addWidget(self.name_line_edit)
        self.name_line.setLayout(name_hbox)
        self.name_line.setFixedHeight(40)
       
        conda_name_hbox = QHBoxLayout()
        conda_env_hbox.addWidget(self.conda_env_line)
        conda_env_hbox.addWidget(self.name_line)
        conda_env_hbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing)

        self.name_line_edit.setText(self.settings.get('project_name', 'yolov5_project'))

        self.conda_env_line.setFixedHeight(40)

        self.conda_name_line = QWidget()
        
        self.conda_name_line.setLayout(conda_env_hbox)


        self.conda_env_line.setFixedHeight(40)
        self.conda_env_combobox.setCurrentText(self.settings.get('conda_env_name'))
        self.data_line.line_edit.setText(self.settings.get('data_config_path'))
        self.model_cfg_line.line_edit.setText(self.settings.get('model_config_path'))
        self.pretrained_line.line_edit.setText(self.settings.get('pretrained_model_path'))
        self.train_py_line.line_edit.setText(self.settings.get('train_script_path'))
        self.save_dir_line.line_edit.setText(self.settings.get('save_dir', os.path.expanduser('~')))
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pretrained_line)
        layout.addWidget(self.data_line)
        layout.addWidget(self.model_cfg_line)
        layout.addWidget(self.train_py_line)
        layout.addWidget(self.save_dir_line)

        layout.addWidget(self.conda_name_line)
        

        layout.setSpacing(10)

        self.train_parm_group = QtWidgets.QGroupBox('训练参数设置')
        train_parm_layout = QtWidgets.QVBoxLayout()
        self.batch_size_line = QtWidgets.QLineEdit(self.settings.get('batch_size', '16'))
        self.epochs_line = QtWidgets.QLineEdit(self.settings.get('epochs', '300'))
        self.img_size_line = QtWidgets.QLineEdit(self.settings.get('img_size', '640'))
        self.patience_line = QtWidgets.QLineEdit(self.settings.get('patience', '100'))
        resume_layout = QHBoxLayout()
        self.not_resume_button = QRadioButton('否')
        resume_layout.addWidget(self.not_resume_button)
        resume_layout.addWidget(QRadioButton('是'))
        self.resume_line = QWidget()
        self.resume_line.setLayout(resume_layout)
        self.not_resume_button.setChecked(True)
        device_layout = QHBoxLayout()
        self.use_gpu_button = QRadioButton('显卡')
        device_layout.addWidget(self.use_gpu_button)
        device_layout.addWidget(QRadioButton('CPU'))
        self.use_gpu_button.setChecked(True)
        self.device_line = QWidget()
        self.device_line.setLayout(device_layout)
        labels = [['每批图像数量', '最大训练次数'], ['模型输入尺寸', '无改善停止数'], ['继续训练', '训练设备']]
        lines = [[self.batch_size_line, self.epochs_line], [self.img_size_line, self.patience_line],
                 [self.resume_line, self.device_line]]
        parm_gird = QtWidgets.QGridLayout()
        for i, line in enumerate(lines):
            for j, line_widget in enumerate(line):
                hbox = QtWidgets.QHBoxLayout()
                hbox.addWidget(QLabel(labels[i][j]))
                hbox.addWidget(line_widget)
                parm_gird.addLayout(hbox, i, j)

        self.train_parm_group.setLayout(train_parm_layout)
        train_parm_layout.addLayout(parm_gird)
        self.model_size_selector = ModelSizeSelector('n,s,m,l,x')
        model_size_layout = QHBoxLayout()
        model_size_layout.addWidget(QLabel('模型大小'))
        model_size_layout.addWidget(self.model_size_selector)
        train_parm_layout.addLayout(model_size_layout)

        train_parm_layout.setSpacing(10)
        layout.addWidget(self.train_parm_group)

        # 在现有训练参数组之后添加数据增强参数组
        self.aug_group = QtWidgets.QGroupBox('数据增强设置')
        aug_layout = QtWidgets.QVBoxLayout()

        # 添加复选框
        check_widget = QWidget()
        check_layout = QtWidgets.QHBoxLayout()
        self.enable_aug_check = QtWidgets.QCheckBox('启用数据增强')
        self.enable_aug_check.setChecked(self.settings.get('enable_aug', False))
        self.enable_shuffle_check = QtWidgets.QCheckBox('启用字符打乱')
        self.enable_shuffle_check.setChecked(self.settings.get('enable_shuffle',False))
        check_layout.addWidget(self.enable_aug_check)
        check_layout.addWidget(self.enable_shuffle_check)
        check_widget.setLayout(check_layout)
        aug_layout.addWidget(check_widget)

        # 参数输入行
        params_grid = QtWidgets.QGridLayout()

        # 增强强度
        self.aug_strength = QtWidgets.QLineEdit(self.settings.get('aug_strength', '160'))
        params_grid.addWidget(QLabel('增强强度'), 0, 0)
        params_grid.addWidget(self.aug_strength, 0, 1)

        # 验证集比例
        self.valid_ratio = QtWidgets.QLineEdit(self.settings.get('valid_ratio', '0.2'))
        params_grid.addWidget(QLabel('验证集比例'), 1, 0)
        params_grid.addWidget(self.valid_ratio, 1, 1)

        # 测试集比例
        self.test_ratio = QtWidgets.QLineEdit(self.settings.get('test_ratio', '0.1'))
        params_grid.addWidget(QLabel('测试集比例'), 2, 0)
        params_grid.addWidget(self.test_ratio, 2, 1)

        aug_layout.addLayout(params_grid)
        # 在数据增强设置的参数网格之后添加按钮
        self.shuffle_config_btn = QtWidgets.QPushButton("配置打乱分组")
        self.shuffle_config_btn.clicked.connect(self.show_shuffle_group_dialog)
        aug_layout.addWidget(self.shuffle_config_btn)
        self.aug_group.setLayout(aug_layout)

        # 将数据增强组添加到主布局（在训练参数组之后）
        layout.addWidget(self.aug_group)

        # 运行命令框,用于显示生成的运行命令
        self.run_command_line = QtWidgets.QLineEdit()
        self.run_command_line.setReadOnly(True)
        run_command_layout = QtWidgets.QHBoxLayout()
        run_command_layout.addWidget(QLabel('运行命令'))
        run_command_layout.addWidget(self.run_command_line)
        run_command_generate_button = QtWidgets.QPushButton('生成运行命令')
        run_command_generate_button.clicked.connect(self.generate_run_command)
        start_training_button = QtWidgets.QPushButton('开始训练')
        start_training_button.clicked.connect(self.start_training)
        run_command_layout.addWidget(run_command_generate_button)
        run_command_layout.addWidget(start_training_button)
        layout.addLayout(run_command_layout)
        widget = QWidget()
        widget.setLayout(layout)




        self.setCentralWidget(widget)
        self.setWindowTitle('YOLO训练工具')
        self.setFixedSize(450, 640)
        self.generate_run_command()

    # 在MainWindow类中添加以下方法
    def show_shuffle_group_dialog(self):
        # 读取数据集配置
        try:
            with open(self.data_line.line_edit.text(), 'r', encoding='utf-8') as f:
                import yaml
                data_config = yaml.safe_load(f)
                all_names = sorted(list(set(data_config['names'])))  # 去重排序
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"读取数据集配置失败: {str(e)}")
            return

        # 加载保存的分组配置
        saved_groups = self.settings.get('shuffle_groups', {})

        # 创建配置对话框
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("字符分组配置")
        dialog.setMinimumSize(900, 600)

        # 主布局：左侧分组管理，右侧所有字符列表
        main_layout = QtWidgets.QHBoxLayout()

        # 左侧分组面板
        group_panel = QtWidgets.QWidget()
        group_layout = QtWidgets.QVBoxLayout()

        # 分组列表
        self.group_widgets = []
        group_scroll = QtWidgets.QScrollArea()
        group_scroll.setWidgetResizable(True)
        group_container = QtWidgets.QWidget()
        self.group_container_layout = QtWidgets.QVBoxLayout()

        # 初始化分组
        for group_name, group_info in saved_groups.items():
            self._add_group_ui(
                group_container=group_container,
                group_name=group_name,
                color=group_info['color'],
                members=group_info['members']
            )

        group_container.setLayout(self.group_container_layout)
        group_scroll.setWidget(group_container)

        # 添加分组按钮
        add_group_btn = QtWidgets.QPushButton("新建分组")
        add_group_btn.clicked.connect(lambda: self._add_group_ui(group_container))

        group_layout.addWidget(add_group_btn)
        group_layout.addWidget(group_scroll)
        group_panel.setLayout(group_layout)

        # 右侧所有字符列表
        all_chars_panel = QtWidgets.QWidget()
        all_chars_layout = QtWidgets.QVBoxLayout()

        # 搜索框
        search_box = QtWidgets.QLineEdit()
        search_box.setPlaceholderText("搜索字符...")
        all_chars_layout.addWidget(search_box)

        # 字符列表
        self.all_chars_list = QtWidgets.QListWidget()
        self.all_chars_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)

        # 填充字符
        self.char_items = {}
        for name in all_names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.all_chars_list.addItem(item)
            self.char_items[name] = item

        # 应用保存的分组颜色
        self._update_char_colors(saved_groups)

        all_chars_layout.addWidget(self.all_chars_list)
        all_chars_panel.setLayout(all_chars_layout)

        main_layout.addWidget(group_panel, 3)
        main_layout.addWidget(all_chars_panel, 2)
        dialog.setLayout(main_layout)

        # 底部按钮
        btn_layout = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("保存")
        save_btn.clicked.connect(lambda: self._save_groups(dialog, all_names))
        btn_layout.addWidget(save_btn)

        cancel_btn = QtWidgets.QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)

        main_layout.addLayout(btn_layout)

        # 实时搜索
        search_box.textChanged.connect(self._filter_chars)

        dialog.exec_()

    def _add_group_ui(self, group_container, group_name=None, color=None, members=None):
        # 生成默认分组信息
        group_name = group_name or f"分组{len(self.group_widgets) + 1}"
        color = color or self._get_random_color()
        members = members or []

        # 分组Widget
        group_widget = QtWidgets.QGroupBox()
        group_layout = QtWidgets.QVBoxLayout()

        # 头部：名称和颜色选择
        header_layout = QtWidgets.QHBoxLayout()

        # 颜色选择按钮
        color_btn = QtWidgets.QPushButton()
        color_btn.setStyleSheet(f"background-color: {color};")
        color_btn.setFixedSize(20, 20)
        color_btn.clicked.connect(lambda: self._change_group_color(color_btn, group_widget))

        # 分组名称
        name_edit = QtWidgets.QLineEdit(group_name)

        # 删除按钮
        del_btn = QtWidgets.QPushButton("×")
        del_btn.setFixedSize(20, 20)
        del_btn.clicked.connect(lambda: self._remove_group(group_widget))

        header_layout.addWidget(color_btn)
        header_layout.addWidget(name_edit)
        header_layout.addWidget(del_btn)

        # 成员列表
        member_list = QtWidgets.QListWidget()
        member_list.setAcceptDrops(True)
        member_list.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)

        # 初始化成员
        for name in members:
            item = QtWidgets.QListWidgetItem(name)
            member_list.addItem(item)
            if name in self.char_items:
                self.char_items[name].setCheckState(QtCore.Qt.Checked)

        # 保存分组引用
        group_data = {
            'widget': group_widget,
            'color_btn': color_btn,
            'name_edit': name_edit,
            'member_list': member_list,
            'color': color
        }
        self.group_widgets.append(group_data)

        group_layout.addLayout(header_layout)
        group_layout.addWidget(member_list)
        group_widget.setLayout(group_layout)
        self.group_container_layout.addWidget(group_widget)

        # 设置样式
        self._update_group_style(group_data)

        # 绑定事件
        member_list.itemChanged.connect(self._update_char_colors)

    def _change_group_color(self, color_btn, group_widget):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            color_btn.setStyleSheet(f"background-color: {hex_color};")
            # 更新对应分组的颜色数据
            for group in self.group_widgets:
                if group['widget'] == group_widget:
                    group['color'] = hex_color
                    self._update_group_style(group)
                    break

    def _update_group_style(self, group_data):
        style = f"""
        QGroupBox {{
            border: 2px solid {group_data['color']};
            border-radius: 5px;
            margin-top: 1ex;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }}
        """
        group_data['widget'].setStyleSheet(style)

    def _remove_group(self, group_widget):
        # 放回未分组字符
        for group in self.group_widgets:
            if group['widget'] == group_widget:
                while group['member_list'].count() > 0:
                    item = group['member_list'].takeItem(0)
                    self._return_to_ungrouped(item.text())
                break

        # 移除UI组件
        group_widget.deleteLater()
        self.group_widgets = [g for g in self.group_widgets if g['widget'] != group_widget]

    def _return_to_ungrouped(self, name):
        if name in self.char_items:
            self.char_items[name].setCheckState(QtCore.Qt.Unchecked)

    def _filter_chars(self, text):
        for i in range(self.all_chars_list.count()):
            item = self.all_chars_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def _save_groups(self, dialog, all_names):
        groups = {}
        all_used = set()

        for group in self.group_widgets:
            group_name = group['name_edit'].text().strip()
            color = group['color']
            members = []

            # 收集成员
            for i in range(group['member_list'].count()):
                item = group['member_list'].item(i)
                name = item.text()
                if name not in all_names:
                    continue
                if name in all_used:
                    QtWidgets.QMessageBox.warning(self, "错误", f"字符 '{name}' 被重复分配到多个分组！")
                    return
                members.append(name)
                all_used.add(name)

            if group_name and members:
                groups[group_name] = {
                    'color': color,
                    'members': members
                }

        self.settings['shuffle_groups'] = groups
        dialog.accept()

    def _update_char_colors(self, groups=None):
        # 根据当前分组状态更新字符颜色
        color_map = {}
        if groups:
            # 初始化时使用保存的分组
            for group_name, info in groups.items():
                for name in info['members']:
                    color_map[name] = info['color']
        else:
            # 运行时根据当前分组更新
            for group in self.group_widgets:
                color = group['color']
                for i in range(group['member_list'].count()):
                    name = group['member_list'].item(i).text()
                    color_map[name] = color

        for name, item in self.char_items.items():
            color = color_map.get(name, "#FFFFFF")
            item.setBackground(QtGui.QColor(color))

    def _get_random_color(self):
        colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD9BA']
        return colors[len(self.group_widgets) % len(colors)]

    def generate_run_command(self):
        plus = '/bin/python' if sys.platform == 'darwin' else '/python.exe'
        env_path = self.envs[self.conda_env_combobox.currentIndex()][1] + plus
        train_script_path = self.train_py_line.line_edit.text()

        run_command = '%s %s --data %s --cfg %s --weights %s --batch-size %s --epochs %s --img-size %s --patience %s --device %s %s --project %s --name %s' % (
            env_path, train_script_path, self.data_line.line_edit.text(),
            self.model_cfg_line.line_edit.text(),
            self.pretrained_line.line_edit.text(),
            self.batch_size_line.text(), self.epochs_line.text(), self.img_size_line.text(), self.patience_line.text(),
            '0' if self.use_gpu_button.isChecked() else 'cpu', '' if self.not_resume_button.isChecked() else '--resume',self.save_dir_line.line_edit.text(),self.name_line_edit.text())
        self.run_command_line.setText(run_command)

    def start_training(self):
        self.generate_run_command()

        # 执行数据增强
        if self.enable_aug_check.isChecked():
            try:
                file_dir = self.data_line.line_edit.text()
                save_dir = os.path.join(self.save_dir_line.line_edit.text(), "augmented_data")

                # 创建数据增强实例
                from libs.dataAug import dataAugmentation  # 导入你的数据增强类
                aug = dataAugmentation(
                    file_dir=file_dir,
                    save_dir=save_dir,
                    shuffle_char=self.enable_shuffle_check.isChecked(),
                    valid_ratio=float(self.valid_ratio.text()),
                    test_ratio=float(self.test_ratio.text()),
                    increased=int(self.aug_strength.text())
                )

                # 更新训练数据路径为增强后的数据
                self.data_line.line_edit.setText(os.path.join(save_dir, "data.yaml"))

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "数据增强错误", f"执行数据增强时出错：{str(e)}")
                return

        # 运行训练命令
        self.run_in_terminal(self.run_command_line.text())

    def closeEvent(self, a0):
        self.settings['data_config_path'] = self.data_line.line_edit.text()
        self.settings['model_config_path'] = self.model_cfg_line.line_edit.text()
        self.settings['pretrained_model_path'] = self.pretrained_line.line_edit.text()
        self.settings['batch_size'] = self.batch_size_line.text()
        self.settings['epochs'] = self.epochs_line.text()
        self.settings['img_size'] = self.img_size_line.text()
        self.settings['patience'] = self.patience_line.text()
        self.settings['resume'] = self.not_resume_button.isChecked()
        self.settings['device'] = self.use_gpu_button.isChecked()
        self.settings['conda_env_name'] = self.conda_env_combobox.currentText()
        self.settings['train_script_path'] = self.train_py_line.line_edit.text()
        self.settings['save_dir'] = self.save_dir_line.line_edit.text()
        self.settings['project_name'] = self.name_line_edit.text()
        self.settings['enable_aug'] = self.enable_aug_check.isChecked()
        self.settings['enable_shuffle'] = self.enable_shuffle_check.isChecked()
        self.settings['aug_strength'] = self.aug_strength.text()
        self.settings['valid_ratio'] = self.valid_ratio.text()
        self.settings['test_ratio'] = self.test_ratio.text()
        self.settings.save()


    def run_in_terminal(self,command):
        # 获取脚本所在的目录
        script_dir = os.path.dirname(self.train_py_line.line_edit.text())
        # 在macOS中
        if sys.platform == 'darwin':
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.sh') as f:
                f.write('#!/bin/bash\n')
                f.write('cd ' + script_dir + '\n')  # 切换到脚本所在的目录
                f.write(command + '\n')
            os.chmod(f.name, 0o700)
            subprocess.Popen(['open', '-a', 'Terminal.app', f.name])
        # 在Windows中
        elif sys.platform == 'win32':
            subprocess.Popen(['start', 'cmd', '/k', 'cd /d ' + script_dir + ' && ' + command], shell=True)
        # 在Linux中（需要xterm）
        elif 'linux' in sys.platform:
            subprocess.Popen(['xterm', '-e', 'cd ' + script_dir + ' && ' + command])
        else:
            print("Unsupported platform")


@lru_cache(maxsize=5)
def get_conda_env_python_path(env_name):
    try:
        print('1111')
        # 使用conda命令行接口获取环境信息
        result = subprocess.run(['conda', 'env', 'list'], stdout=subprocess.PIPE, check=True)
        # 解析结果
        lines = result.stdout.decode().split('\n')
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == env_name:
                return parts[-1] + '/bin/python'
    except Exception as e:
        print(f"Error while getting conda env python path: {e}")
    return None


def get_all_conda_envs():
    try:
        # 使用批处理脚本获取环境信息
        result = subprocess.run(['cmd', '/c', 'get_conda_envs.bat'], stdout=subprocess.PIPE, check=True)
        # 解析结果
        lines = result.stdout.decode().split('\n')
        envs = []
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                envs.append([parts[0], parts[-1]])
        return envs
    except Exception as e:
        print(f"Error while getting conda envs: {e}")
    return None


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(':/app'))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
