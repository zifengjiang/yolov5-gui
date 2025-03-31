from functools import lru_cache
import random

import yaml
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QHBoxLayout, QRadioButton, QComboBox, QLineEdit

from libs.LineWidget import LineWidget
from libs.dataAug import AugWorker
from libs.model_size_selector import ModelSizeSelector
from libs.settings import Settings
from libs.resources import *
import subprocess
import os
import sys
import tempfile

from ShuffleGroupDialog import ShuffleGroupDialog, get_yaml_keys


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
        self.save_dir_line = LineWidget(
            '训练结果保存路径', self.settings.get('save_dir', os.path.expanduser('~')))
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
        conda_name_hbox.addWidget(self.conda_env_line)
        conda_name_hbox.addWidget(self.name_line)
        conda_name_hbox.setAlignment(QtCore.Qt.AlignTrailing)
        conda_env_hbox.addWidget(self.conda_env_line)
        conda_env_hbox.addWidget(self.name_line)
        conda_env_hbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignTrailing)

        self.name_line_edit.setText(
            self.settings.get('project_name', 'yolov5_project'))

        self.conda_env_line.setFixedHeight(40)

        self.conda_name_line = QWidget()

        self.conda_name_line.setLayout(conda_name_hbox)

        self.conda_env_line.setFixedHeight(40)
        self.conda_env_combobox.setCurrentText(
            self.settings.get('conda_env_name'))
        self.data_line.line_edit.setText(self.settings.get('data_config_path'))
        self.model_cfg_line.line_edit.setText(
            self.settings.get('model_config_path'))
        self.pretrained_line.line_edit.setText(
            self.settings.get('pretrained_model_path'))
        self.train_py_line.line_edit.setText(
            self.settings.get('train_script_path'))
        self.save_dir_line.line_edit.setText(
            self.settings.get('save_dir', os.path.expanduser('~')))
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
        self.batch_size_line = QtWidgets.QLineEdit(
            self.settings.get('batch_size', '16'))
        self.epochs_line = QtWidgets.QLineEdit(
            self.settings.get('epochs', '300'))
        self.img_size_line = QtWidgets.QLineEdit(
            self.settings.get('img_size', '640'))
        self.patience_line = QtWidgets.QLineEdit(
            self.settings.get('patience', '100'))
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
        self.enable_aug_check.setChecked(
            self.settings.get('enable_aug', False))
        self.enable_shuffle_check = QtWidgets.QCheckBox('启用字符打乱')
        self.enable_shuffle_check.setChecked(
            self.settings.get('enable_shuffle', False))
        check_layout.addWidget(self.enable_aug_check)
        check_layout.addWidget(self.enable_shuffle_check)
        check_widget.setLayout(check_layout)
        aug_layout.addWidget(check_widget)

        # 参数输入行
        params_grid = QtWidgets.QGridLayout()

        # 增强强度
        self.aug_strength = QtWidgets.QLineEdit(
            self.settings.get('aug_strength', '160'))
        params_grid.addWidget(QLabel('增强强度'), 0, 0)
        params_grid.addWidget(self.aug_strength, 0, 1)

        aug_layout.addLayout(params_grid)
        # 在数据增强设置的参数网格之后添加按钮
        self.shuffle_config_btn = QtWidgets.QPushButton("配置打乱分组")
        self.shuffle_config_btn.clicked.connect(self.show_shuffle_group_dialog)
        aug_layout.addWidget(self.shuffle_config_btn)
        self.progress_bar = QtWidgets.QProgressBar()
        aug_layout.addWidget(self.progress_bar)
        self.progress_bar.setValue(0)  # 初始为0
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
        self.setFixedSize(450, 700)
        self.generate_run_command()

    def generate_run_command(self):
        plus = '/bin/python' if sys.platform == 'darwin' else '/python.exe'
        env_path = self.envs[self.conda_env_combobox.currentIndex()][1] + plus
        train_script_path = self.train_py_line.line_edit.text()

        run_command = '%s %s --data %s --cfg %s --weights %s --batch-size %s --epochs %s --img-size %s --patience %s --device %s %s --project %s --name %s' % (
            env_path, train_script_path, self.data_line.line_edit.text(),
            self.model_cfg_line.line_edit.text(),
            self.pretrained_line.line_edit.text(),
            self.batch_size_line.text(), self.epochs_line.text(
            ), self.img_size_line.text(), self.patience_line.text(),
            '0' if self.use_gpu_button.isChecked() else 'cpu', '' if self.not_resume_button.isChecked() else '--resume', self.save_dir_line.line_edit.text(), self.name_line_edit.text())
        self.run_command_line.setText(run_command)

    def start_training(self):
        # 0. 如果你之前有别的校验、获取路径、生成命令等逻辑，这里都不变...
        #    这里只演示如何把数据增强放到子线程中执行 + 进度条

        if self.enable_aug_check.isChecked():
            # 1) 创建并配置 dataAugmentation 对象
            file_dir = self.data_line.line_edit.text()
            save_dir = os.path.join(
                self.save_dir_line.line_edit.text(), self.name_line_edit.text(), "augmented_data"
            )
            from libs.dataAug import dataAugmentation
            aug = dataAugmentation(
                yaml_file=file_dir,
                save_dir=save_dir,
                shuffle_char=self.enable_shuffle_check.isChecked(),
                increased=int(self.aug_strength.text()),
                shuffle_groups=self.settings.get(
                    get_yaml_keys(self.data_line.line_edit.text()), {}
                )
            )

            # 2) 创建工作者和线程
            self.aug_thread = QtCore.QThread(self)  # 注意:要给 self 以免被垃圾回收
            self.aug_worker = AugWorker(aug)  # 把上面写的 Worker 类放进来
            self.aug_worker.moveToThread(self.aug_thread)  # Worker移动到子线程

            # 3) 连接信号槽
            # 当子线程启动时，调用 Worker.run
            self.aug_thread.started.connect(self.aug_worker.run)
            # 当 Worker 报告进度时，调用 on_aug_progress
            self.aug_worker.progressChanged.connect(self.on_aug_progress)
            # 当 Worker 出错时，调用 on_aug_error
            self.aug_worker.errorOccurred.connect(self.on_aug_error)
            # 当 Worker 完成时，调用 on_aug_finished
            self.aug_worker.finished.connect(self.on_aug_finished)
            # 完成后，退出并清理线程
            self.aug_worker.finished.connect(self.aug_thread.quit)
            self.aug_worker.finished.connect(self.aug_worker.deleteLater)
            self.aug_thread.finished.connect(self.aug_thread.deleteLater)

            # 4) 启动线程
            self.aug_thread.start()

            # 让进度条归零，并且避免用户多次点击
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("开始数据增强...")
            self.progress_bar.setMaximum(100)  # 先假设 100，后面在 on_aug_progress 里再动态修正
            # 如果你想禁用按钮防止重复点击，也可以：
            # start_training_button.setEnabled(False)

        else:
            # 没勾选数据增强，那就正常继续训练即可
            self.run_train_command()

    def on_aug_progress(self, current, total, img_path):
        """子线程实时发回的数据增强进度。"""
        if total == 0:
            return
        percent = int(current * 100 / total)
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"数据增强中：{percent}% （{current}/{total}）")

    def on_aug_error(self, error_msg):
        """数据增强出现异常时。"""
        QtWidgets.QMessageBox.critical(self, "数据增强错误", f"执行数据增强时出错：{error_msg}")

    def on_aug_finished(self):
        """数据增强完成后，继续后续训练流程。"""
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("数据增强完成！")
        # 这里就可以执行后续训练命令了
        # 比如：
        tmp = self.data_line.line_edit.text()
        save_dir = os.path.join(self.save_dir_line.line_edit.text(),
                                self.name_line_edit.text(), "augmented_data")
        self.data_line.line_edit.setText(os.path.join(save_dir, "data.yaml"))
        self.generate_run_command()
        self.data_line.line_edit.setText(tmp)
        self.run_in_terminal(self.run_command_line.text())

        # 如果有按钮需要恢复可用，也可以：
        # start_training_button.setEnabled(True)

    def run_train_command(self):
        """不做数据增强时，直接执行训练。"""
        self.generate_run_command()
        self.run_in_terminal(self.run_command_line.text())

    def show_shuffle_group_dialog(self):
        # 打开对话框之前，先确保能拿到 data.yaml 中的 names
        yaml_path = self.data_line.line_edit.text()
        dialog = ShuffleGroupDialog(self, self.settings, yaml_path)
        dialog.exec_()  # 模态方式打开
        # 在对话框关闭后，分组信息已经保存在 self.settings["shuffle_groups"] 中
        # 你可以在这之后做一些更新 UI 或者其他操作
        # print(self.settings.get("shuffle_groups", {}))

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
        self.settings.save()

    def run_in_terminal(self, command):
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
            subprocess.Popen(['start', 'cmd', '/k', 'cd /d ' +
                             script_dir + ' && ' + command], shell=True)
        # 在Linux中（需要xterm）
        elif 'linux' in sys.platform:
            subprocess.Popen(
                ['xterm', '-e', 'cd ' + script_dir + ' && ' + command])
        else:
            print("Unsupported platform")


@lru_cache(maxsize=5)
def get_conda_env_python_path(env_name):
    try:
        print('1111')
        # 使用conda命令行接口获取环境信息
        result = subprocess.run(
            ['conda', 'env', 'list'], stdout=subprocess.PIPE, check=True)
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
        if sys.platform == 'win32':
            result = subprocess.run(
                ['cmd', '/c', 'get_conda_envs.bat'], stdout=subprocess.PIPE, check=True)
        else:
            result = subprocess.run(
                ['bash', 'get_conda_envs.sh'], stdout=subprocess.PIPE, check=True)
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
