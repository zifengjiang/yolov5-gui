from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton


class GridExample(QWidget):
    def __init__(self):
        super().__init__()

        # 创建一个网格布局
        grid = QGridLayout()

        # 创建一些按钮并添加到网格布局中
        for i in range(3):
            for j in range(3):
                button = QPushButton(f'Button {i}-{j}')
                grid.addWidget(button, i, j)

        # 设置布局
        self.setLayout(grid)


# 使用方法
if __name__ == '__main__':
    app = QApplication([])

    window = GridExample()
    window.show()

    app.exec_()
