from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel

class YoloParamsDialog(QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO 参数设置")
        self.settings = settings or {}

        # 参数定义
        self.params = {
            "model": {"type": "str", "default": None, "description": "指定用于训练的模型文件"},
            "data": {"type": "str", "default": None, "description": "数据集配置文件的路径"},
            "epochs": {"type": "int", "default": 100, "description": "训练历元总数"},
            "batch": {"type": "int", "default": 16, "description": "批量大小"},
            "imgsz": {"type": "int", "default": 640, "description": "目标图像尺寸"},
            "device": {"type": "str", "default": "cuda", "description": "指定用于训练的计算设备"},
            "optimizer": {"type": "str", "default": "auto", "description": "选择优化器"},
            "lr0": {"type": "float", "default": 0.01, "description": "初始学习率"},
            "momentum": {"type": "float", "default": 0.937, "description": "动量因子"},
            "weight_decay": {"type": "float", "default": 0.0005, "description": "L2正则化项"},
            "val": {"type": "bool", "default": True, "description": "是否进行验证"},
        }

        # 布局
        layout = QVBoxLayout()
        self.form_layout = QtWidgets.QFormLayout()

        # 动态生成控件
        self.controls = {}
        for param, info in self.params.items():
            label = QLabel(f"{param} ({info['description']})")
            if info["type"] == "bool":
                control = QtWidgets.QCheckBox()
                control.setChecked(self.settings.get(param, info["default"]))
            elif info["type"] in ["int", "float", "str"]:
                control = QtWidgets.QLineEdit()
                control.setText(str(self.settings.get(param, info["default"])))
            else:
                control = QtWidgets.QLineEdit()
            self.controls[param] = control
            self.form_layout.addRow(label, control)

        layout.addLayout(self.form_layout)

        # 按钮
        button_layout = QHBoxLayout()
        save_button = QtWidgets.QPushButton("保存")
        save_button.clicked.connect(self.save_params)
        cancel_button = QtWidgets.QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def save_params(self):
        """保存参数到 settings"""
        for param, control in self.controls.items():
            if isinstance(control, QtWidgets.QCheckBox):
                self.settings[param] = control.isChecked()
            elif isinstance(control, QtWidgets.QLineEdit):
                value = control.text()
                param_type = self.params[param]["type"]
                if param_type == "int":
                    self.settings[param] = int(value)
                elif param_type == "float":
                    self.settings[param] = float(value)
                else:
                    self.settings[param] = value
        self.accept()