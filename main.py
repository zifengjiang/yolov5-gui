from functools import lru_cache

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QHBoxLayout, QRadioButton, QComboBox

from LineWidget import LineWidget
from model_size_selector import ModelSizeSelector
from settings import Settings
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
        self.conda_env_line.setLayout(conda_env_hbox)
        self.conda_env_line.setFixedHeight(40)
        self.conda_env_combobox.setCurrentText(self.settings.get('conda_env_name'))
        self.data_line.line_edit.setText(self.settings.get('data_config_path'))
        self.model_cfg_line.line_edit.setText(self.settings.get('model_config_path'))
        self.pretrained_line.line_edit.setText(self.settings.get('pretrained_model_path'))
        self.train_py_line.line_edit.setText(self.settings.get('train_script_path'))
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pretrained_line)
        layout.addWidget(self.data_line)
        layout.addWidget(self.model_cfg_line)
        layout.addWidget(self.train_py_line)
        layout.addWidget(self.conda_env_line)

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
        self.setWindowTitle('YOLOv5训练工具')
        self.setFixedSize(450, 500)

    def generate_run_command(self):
        plus = '/bin/python' if sys.platform == 'darwin' else '/python.exe'
        env_path = self.envs[self.conda_env_combobox.currentIndex()][1] + plus
        train_script_path = self.train_py_line.line_edit.text()

        run_command = '%s %s --data %s --cfg %s --weights %s --batch-size %s --epochs %s --img-size %s --patience %s --device %s %s' % (
            env_path, train_script_path, self.data_line.line_edit.text(),
            self.model_cfg_line.line_edit.text(),
            self.pretrained_line.line_edit.text(),
            self.batch_size_line.text(), self.epochs_line.text(), self.img_size_line.text(), self.patience_line.text(),
            '0' if self.use_gpu_button.isChecked() else 'cpu', '' if self.not_resume_button.isChecked() else '--resume')
        self.run_command_line.setText(run_command)

    def start_training(self):
        self.generate_run_command()
        run_in_terminal(self.run_command_line.text())

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
        self.settings.save()


def run_in_terminal(command):
    # 在macOS中
    if sys.platform == 'darwin':
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.sh') as f:
            f.write('#!/bin/bash\n')
            f.write(command + '\n')
        os.chmod(f.name, 0o700)
        subprocess.Popen(['open', '-a', 'Terminal.app', f.name])
    # 在Windows中
    elif sys.platform == 'win32':
        subprocess.Popen(['start', 'cmd', '/k', command], shell=True)
    # 在Linux中（需要xterm）
    elif 'linux' in sys.platform:
        subprocess.Popen(['xterm', '-e', command])
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
        # 使用conda命令行接口获取环境信息
        result = subprocess.run(['conda', 'env', 'list'], stdout=subprocess.PIPE, check=True)
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
    app.setWindowIcon(QtGui.QIcon('icons/app.png'))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
