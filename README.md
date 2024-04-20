## yolov5的训练助手，选择相关路径后可以快速开始训练

## 使用pyinstaller进行打包
```
pyinstaller --onefile --windowed --add-data 'get_conda_envs.bat;.' -i 'app.ico' -n 'YOLO训练助手' main.py
```