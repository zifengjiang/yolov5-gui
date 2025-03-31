import multiprocessing
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QMessageBox
import os
import subprocess
from PyQt5 import QtCore
import sys

def check_environment(python_path, required_packages, conn):
    """在子进程中运行的环境检查逻辑"""
    missing_packages = []
    for package in required_packages:
        try:
            subprocess.run(
                [python_path, '-c', f'import {package}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError:
            missing_packages.append(package)

    # 检查 torch 是否为 CUDA 版本
    try:
        result = subprocess.run(
            [python_path, '-c', 'import torch; print(torch.cuda.is_available())'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        is_cuda_available = result.stdout.decode().strip() == 'True'
        if not is_cuda_available:
            missing_packages.append('torch (CUDA version required)')
    except subprocess.CalledProcessError:
        missing_packages.append('torch')

    # 将结果发送回主进程
    conn.send(missing_packages)
    conn.close()


class LoadingDialog(QDialog):
    """加载对话框，用于显示检查进度"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("检查环境中...")
        self.setModal(True)
        layout = QVBoxLayout()
        self.label = QLabel("正在检查环境，请稍候...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 设置为无限进度条
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)





def poll_check_result(self, conn):
    """轮询检查子进程的结果"""
    if conn.poll():  # 如果子进程有结果返回
        missing_packages = conn.recv()
        self.timer.stop()
        self.check_process.join()
        self.loading_dialog.close()

        if missing_packages:
            QMessageBox.critical(
                self, "环境检查失败",
                f"当前 Conda 环境缺少以下必要库或配置不正确：\n{', '.join(missing_packages)}\n"
                "请安装后再尝试运行训练。"
            )
        else:
            QMessageBox.information(
                self, "环境检查成功", "所有必要库已安装，可以开始训练！"
            )
            self.start_training()  # 开始训练