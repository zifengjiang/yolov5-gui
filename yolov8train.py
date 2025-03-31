from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolov8n-cls.pt")

    train_results = model.train(
        name=r"F:\毕业论文实验\液晶屏分类模型",  # model name
        data=r"F:\毕业论文实验\分类任务数据集",  # path to dataset YAML
        epochs=150,  # number of training epochs
        imgsz=32,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=32,  # batch size
        workers=8,
        patience=100,
        # number of workers
    )

    # Load a model
    model = YOLO("yolov5s.pt")

    train_results = model.train(
        name=r"F:\毕业论文实验\数据增强实验\实验结果\yolov5s_粗粒度模型",  # model name
        data=r"F:\Projects\PycharmProjects\yolo5\data\meter.yaml",  # path to dataset YAML
        epochs=50,  # number of training epochs
        imgsz=1024,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=8,  # batch size
        workers=2,
        patience=100,
         # number of workers
    )

    train_results = model.train(
        name=r"F:\毕业论文实验\数据增强实验\实验结果\yolov5s_铭牌",  # model name
        data=r"F:\Projects\PycharmProjects\yolo5\data\nameplate.yaml",  # path to dataset YAML
        epochs=50,  # number of training epochs
        imgsz=1024,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=8,  # batch size
        workers=2,
        patience=100,
         # number of workers
    )

    # Train the model
    train_results = model.train(
        name=r"F:\毕业论文实验\数据增强实验\实验结果\yolov5s_baseline",  # model name
        data=r"data\数据增强实验\baseline.yaml",  # path to dataset YAML
        epochs=300,  # number of training epochs
        imgsz=1024,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=4,  # batch size
        workers=2,
        patience=100,
         # number of workers
    )

    model = YOLO("yolov5s.pt")

    train_results = model.train(
        name=r"F:\毕业论文实验\数据增强实验\实验结果\yolov5s_imgAug",  # model name
        data=r"data\数据增强实验\imgAug.yaml",  # path to dataset YAML
        epochs=300,  # number of training epochs
        imgsz=1024,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=4,  # batch size
        workers=2,
        patience=100,
         # number of workers
    )

    model = YOLO("yolov5s.pt")

    train_results = model.train(
        name=r"F:\毕业论文实验\数据增强实验\实验结果\yolov5s_imgAug+shuffle",  # model name
        data=r"data\数据增强实验\imgAug+shuffle.yaml",  # path to dataset YAML
        epochs=300,  # number of training epochs
        imgsz=1024,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=4,  # batch size
        workers=2,
        patience=100,
         # number of workers
    )

    model = YOLO("yolov5s.pt")

    train_results = model.train(
        name=r"F:\毕业论文实验\数据增强实验\实验结果\yolov5s_shuffle",  # model name
        data=r"data\数据增强实验\shuffle.yaml",  # path to dataset YAML
        epochs=300,  # number of training epochs
        imgsz=1024,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=4,  # batch size
        workers=2,
        patience=100,
         # number of workers
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    