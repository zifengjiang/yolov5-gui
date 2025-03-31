# ShuffleGroupDialog.py
import yaml
import os
import random
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QListWidget, QListWidgetItem, QLineEdit, QColorDialog,
    QMessageBox, QInputDialog
)
from PyQt5.QtGui import QColor
from PyQt5 import QtCore


def load_names_from_yaml(yaml_path):
    """从 data.yaml 中读取 names 列表，并去重、排序后返回"""
    if not os.path.exists(yaml_path):
        return []
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        # 去重并排序
        names = list(set(names))
        names.sort()
        return names
    except Exception as e:
        print("读取 data.yaml 失败:", e)
        return []


def get_yaml_keys(yaml_path):
    names = load_names_from_yaml(yaml_path)
    _names = list(set(names))
    _names.sort()
    return "char_group:"+",".join(_names)


class ShuffleGroupDialog(QDialog):
    """用于配置打乱字符的分组信息，对同一个字符只允许选一个分组。"""

    def __init__(self, parent, settings, data_yaml_path):
        super().__init__(parent)
        self.setWindowTitle("配置打乱字符分组")
        self.resize(600, 400)
        self.settings = settings
        self.data_yaml_path = data_yaml_path

        # 读取 data.yaml 中的 names
        self.all_names = load_names_from_yaml(self.data_yaml_path)
        self.all_names_str = get_yaml_keys(self.data_yaml_path)

        # 分组数据结构：dict，形如：
        # {
        #   "group1": {"color": "#FF0000", "items": ["cat", "dog"]},
        #   "group2": {"color": "#00FF00", "items": ["person"]},
        #   ...
        # }
        # 如果 settings 中已保存有 shuffle_groups，就拿过来用，否则就创建一个空的
        self.shuffle_groups = self.settings.get(self.all_names_str, {})

        # 主布局
        main_layout = QVBoxLayout(self)

        # 1. 上方：选择分组 + 新增分组 + 删除分组 + 重命名分组 + 设置颜色
        top_hbox = QHBoxLayout()
        top_hbox.addWidget(QLabel("选择分组:"))
        self.group_combobox = QComboBox()
        self.reload_group_combobox_items()  # 根据 self.shuffle_groups 刷新分组下拉
        top_hbox.addWidget(self.group_combobox)

        # 添加若干按钮
        self.add_group_btn = QPushButton("添加分组")
        self.del_group_btn = QPushButton("删除分组")
        self.rename_group_btn = QPushButton("重命名")
        self.set_color_btn = QPushButton("设置颜色")

        top_hbox.addWidget(self.add_group_btn)
        top_hbox.addWidget(self.del_group_btn)
        top_hbox.addWidget(self.rename_group_btn)
        top_hbox.addWidget(self.set_color_btn)

        main_layout.addLayout(top_hbox)

        # 2. 中间：搜索框
        search_hbox = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索字符...")
        search_hbox.addWidget(QLabel("搜索:"))
        search_hbox.addWidget(self.search_edit)
        main_layout.addLayout(search_hbox)

        # 3. 列表
        self.list_widget = QListWidget()
        main_layout.addWidget(self.list_widget)

        # 底部：保存/取消按钮（也可直接右上角关闭）
        bottom_hbox = QHBoxLayout()
        self.save_btn = QPushButton("保存")
        self.cancel_btn = QPushButton("取消")
        bottom_hbox.addWidget(self.save_btn)
        bottom_hbox.addWidget(self.cancel_btn)
        main_layout.addLayout(bottom_hbox)

        # 初始化列表显示
        self.refresh_list()

        # 信号槽
        self.group_combobox.currentIndexChanged.connect(self.on_group_changed)
        self.add_group_btn.clicked.connect(self.on_add_group)
        self.del_group_btn.clicked.connect(self.on_del_group)
        self.rename_group_btn.clicked.connect(self.on_rename_group)
        self.set_color_btn.clicked.connect(self.on_set_color)
        self.search_edit.textChanged.connect(self.on_search_changed)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)

        self.save_btn.clicked.connect(self.on_save)
        self.cancel_btn.clicked.connect(self.on_cancel)

    def reload_group_combobox_items(self):
        """根据 self.shuffle_groups 刷新 group_combobox 的下拉项"""
        self.group_combobox.blockSignals(True)
        self.group_combobox.clear()
        group_list = list(self.shuffle_groups.keys())
        group_list.sort()
        self.group_combobox.addItems(group_list)
        self.group_combobox.blockSignals(False)

    def get_current_group_name(self):
        """获取 combobox 当前选择的分组名，没有则返回 None"""
        if self.group_combobox.currentIndex() < 0:
            return None
        return self.group_combobox.currentText()

    def on_group_changed(self, index):
        """当切换分组时，刷新列表颜色显示"""
        self.refresh_list()

    def on_add_group(self):
        """添加分组"""
        base_name = "group"
        i = 1
        # 保证分组名唯一
        while True:
            new_group_name = f"{base_name}{i}"
            if new_group_name not in self.shuffle_groups:
                break
            i += 1

        # 默认颜色随机生成，或者指定一个固定色
        random_color = QColor.fromHsv(
            int(360 * random.random()),
            200 + int(55 * random.random()),
            200 + int(55 * random.random())
        )
        color_hex = random_color.name()

        self.shuffle_groups[new_group_name] = {
            "color": color_hex,
            "items": []
        }
        self.reload_group_combobox_items()
        # 自动选中新建的分组
        idx = self.group_combobox.findText(new_group_name)
        if idx >= 0:
            self.group_combobox.setCurrentIndex(idx)
        self.refresh_list()

    def on_del_group(self):
        """删除当前分组"""
        group_name = self.get_current_group_name()
        if group_name is None:
            return
        reply = QMessageBox.question(
            self, "删除分组", f"确定删除分组 {group_name} 吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.shuffle_groups.pop(group_name, None)
            self.reload_group_combobox_items()
            self.refresh_list()

    def on_rename_group(self):
        """重命名当前分组"""
        group_name = self.get_current_group_name()
        if group_name is None:
            return

        text, ok = QInputDialog.getText(
            self, "重命名分组", "新的分组名称", text=group_name
        )
        if ok and text:
            new_name = text.strip()
            # 如果新名字与旧名字不同且不冲突
            if new_name and new_name not in self.shuffle_groups:
                # 先把旧的取出
                old_data = self.shuffle_groups.pop(group_name)
                # 放入新的key
                self.shuffle_groups[new_name] = old_data

                self.reload_group_combobox_items()
                idx = self.group_combobox.findText(new_name)
                if idx >= 0:
                    self.group_combobox.setCurrentIndex(idx)
                self.refresh_list()

    def on_set_color(self):
        """设置当前分组的颜色"""
        group_name = self.get_current_group_name()
        if group_name is None:
            return
        current_color = self.shuffle_groups[group_name]["color"]
        dlg = QColorDialog(QColor(current_color), self)
        if dlg.exec_() == QColorDialog.Accepted:
            chosen = dlg.selectedColor()
            self.shuffle_groups[group_name]["color"] = chosen.name()
            self.refresh_list()

    def on_search_changed(self, text):
        """搜索框文本改变时，刷新列表"""
        self.refresh_list()

    def on_item_double_clicked(self, item):
        """
        双击某个字符时：
        1. 若该字符已在「当前分组」，则将其移出（恢复到无分组）。
        2. 若该字符在「其他分组」，则移出该分组后加入当前分组。
        3. 若该字符原本不在任何分组，则加入当前分组。
        """
        name = item.text()
        current_group_name = self.get_current_group_name()
        if not current_group_name:
            return

        # 判断是否在当前分组
        if name in self.shuffle_groups[current_group_name]["items"]:
            # 移出当前分组 => 不属于任何分组
            self.shuffle_groups[current_group_name]["items"].remove(name)
        else:
            # 不在当前分组，可能在别的分组，也可能无分组
            self.remove_name_from_all_groups(name)
            self.shuffle_groups[current_group_name]["items"].append(name)

        self.refresh_list()

    def remove_name_from_all_groups(self, name):
        """保证同一个字符只能在一个分组中，将其从所有分组中移除"""
        for gname, ginfo in self.shuffle_groups.items():
            if name in ginfo["items"]:
                ginfo["items"].remove(name)

    def refresh_list(self):
        """根据搜索、分组状态，刷新列表显示，每个字符都有其对应分组的颜色"""
        self.list_widget.clear()

        search_text = self.search_edit.text().strip().lower()

        for name in self.all_names:
            # 搜索过滤
            if search_text and (search_text not in name.lower()):
                continue

            item = QListWidgetItem(name)

            # 找到该 name 所在的分组，如果有，就显示该分组颜色
            found_group_name = self.find_group_by_item(name)
            if found_group_name is not None:
                color_hex = self.shuffle_groups[found_group_name]["color"]
                item.setBackground(QColor(color_hex))
            else:
                # 不在任何分组，则保持默认
                pass

            self.list_widget.addItem(item)

    def find_group_by_item(self, name):
        """找到 name 所在的分组并返回分组名；如果没有所在分组则返回 None"""
        for gname, ginfo in self.shuffle_groups.items():
            if name in ginfo["items"]:
                return gname
        return None

    def on_save(self):
        """保存分组信息到 settings 并关闭对话框"""
        # 将更新后的 shuffle_groups 写回 settings
        self.settings[self.all_names_str] = self.shuffle_groups
        self.accept()

    def on_cancel(self):
        """直接关闭对话框，不保存"""
        self.reject()
