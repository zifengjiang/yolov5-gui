from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QApplication, QWidget, QRadioButton, QHBoxLayout, QButtonGroup)


class ModelSizeSelector(QWidget):
    def __init__(self, options):
        super().__init__()

        # 解析传入的逗号分隔字符串为选项列表
        self.options = options.split(',')

        # 创建水平布局
        layout = QHBoxLayout()

        # 创建一个按钮组，确保只能选择一个选项
        self.buttonGroup = QButtonGroup(self)

        # 为每个选项创建一个单选按钮
        for option in self.options:
            button = QRadioButton(option.strip())
            self.buttonGroup.addButton(button)
            layout.addWidget(button)

        # 设置布局
        self.setLayout(layout)

        # 连接信号到槽函数
        self.buttonGroup.buttonClicked.connect(self.onButtonClicked)

        # 默认选择第一个选项
        if self.options:
            self.buttonGroup.buttons()[0].setChecked(True)
        self.currentSelection = self.options[0] if self.options else None

    def onButtonClicked(self, button):
        # 更新当前选中的按钮
        self.currentSelection = button.text()
        print("当前选择是:", self.currentSelection)

    def getCurrentSelection(self):
        # 返回当前选中的按钮的文本
        return self.currentSelection


# 使用方法
if __name__ == '__main__':
    app = QApplication([])

    options = 'n,s,m,l,x'  # 这里可以是任何逗号分隔的字符串
    selector = ModelSizeSelector(options)
    selector.show()


    def printSelection():
        print("当前选择是:", selector.getCurrentSelection())


    # 模拟用户交互，打印当前选择
    QTimer.singleShot(1000, printSelection)

    app.exec_()