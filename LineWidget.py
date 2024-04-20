from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QLineEdit, QLabel


class LineWidget(QWidget):
    def __init__(self, text, default_path=None, filters=None):
        super().__init__()
        self.filters = filters
        self.default_path = default_path
        self.text = QLabel(text)
        self.line_edit = QLineEdit(f'请选择{text}')
        self.line_edit.setReadOnly(True)
        self.button = QtWidgets.QPushButton()
        self.button.setIcon(QtGui.QIcon('icons/folder.jpg'))
        self.button.clicked.connect(self.open_file_dialog)
        self.button.setFixedSize(30, 30)
        # 设置按钮边缘为0，背景透明
        self.button.setStyleSheet('QPushButton{border:0px;background-color:transparent}')
        self.text.setFixedSize(120, 30)
        self.text.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.line_edit.setFixedHeight(30)
        self.line_edit.setAlignment(QtCore.Qt.AlignLeft)
        self.line_edit.setStyleSheet(
            'QLabel{background-color:white;color:black;border-width:1px;border-style:solid;border-color:rgb(37, 99, 235);border-radius:5px}')
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.text)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button)
        layout.setAlignment(QtCore.Qt.AlignLeft)
        self.setLayout(layout)
        self.setFixedHeight(40)

    def open_file_dialog(self):
        if self.filters:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', self.default_path, self.filters)
        else:
            # 没有过滤器，只能选择文件夹
            file_path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择文件夹', self.default_path, QtWidgets.QFileDialog.ShowDirsOnly)
        self.line_edit.setText(file_path)
