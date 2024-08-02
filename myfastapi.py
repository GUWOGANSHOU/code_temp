# -*- coding:utf-8 -*-
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel # pydantic：用于数据接口定义检查与设置管理的库
import time
import torch
import numpy as np
from numpy import random
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized
from models.experimental import attempt_load
from ultralytics import YOLO, YOLOWorld
import PaddleOCR.ppstructure.kie.predict_kie_token_ser as predict_kie_token_ser
import PaddleOCR.ppstructure.kie.predict_kie_token_ser_re as predict_kie_token_ser_re
from PaddleOCR.ppstructure.utility import parse_args
from PaddleOCR.ppstructure.utility_id_kie import parse_args as parse_args_id_kie


app = FastAPI(
    title="Yolo Helmet Detection Model API",
    description="A simple API that use Yolov7 model to detect helmet or head",
    version="0.1",
)

# 定义目标检测数据格式
class Data(BaseModel):
    img: tuple
    conf_thres: float
    iou_thres: float

# 定义 OCR 数据格式
class DataKie(BaseModel):
    img: tuple

t0 = time.time()
# 加载模型
model_helmet = attempt_load('./weights/helmet_detection_best.pt')
print(f'Load helmet Model Done. ({time.time() - t0:.3f})')

t0 = time.time()
# 加载模型
model_dw = YOLO('./weights/dw_best.pt')
print(f'Load dw Model Done. ({time.time() - t0:.3f})')

t0 = time.time()
# 加载模型
model_pcb = YOLO('./weights/pcb_best.pt')
print(f'Load pcb Model Done. ({time.time() - t0:.3f})')

t0 = time.time()
# 加载模型
model_asset = YOLOWorld('./weights/yolov8s_worldv2_coco6_dispenser.pt')
model_asset.set_classes(["person", "chair", "couch", "dining table", "tv", "laptop", "dispenser"])
print(f'Load asset Model Done. ({time.time() - t0:.3f})')


t0 = time.time()
# 加载模型
model_tobacco = YOLO('./weights/tobacco_best_2.pt')
print(f'Load tobacco Model Done. ({time.time() - t0:.3f})')

# 安全帽检测
@app.post('/helmet_detection')
async def detect(data_helmet: Data):
    img0 = data_helmet.img
    # 转为 numpy array
    img0 = np.array(img0)
    img0 = img0.astype("uint8")

    stride = int(model_helmet.stride.max())
    imgsz = check_img_size(640, s=stride)

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Get names and colors
    names = model_helmet.module.names if hasattr(model_helmet, 'module') else model_helmet.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model_helmet(img, augment=False)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, data_helmet.conf_thres, data_helmet.iou_thres, agnostic=False)
    t3 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s = ''
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
        # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    # print(f'Done. ({time.time() - t0:.3f})')
    result = {"prediction": img0.tolist()}
    return result

# 电网绝缘手套检测
@app.post('/dw_detection')
async def detect(data_dw: Data):
    img0 = data_dw.img
    # 转为 numpy array
    img0 = np.array(img0)
    img0 = img0.astype("uint8")

    t1 = time.time()
    results = model_dw(img0, conf=data_dw.conf_thres, iou=data_dw.iou_thres)  # 对图像进行预测
    print(f'Predict Done. ({time.time() - t1:.3f})')

    t1 = time.time()
    im_array = results[0].plot()  # plot a BGR numpy array of predictions
    print(f'Plot Done. ({time.time() - t1:.3f})')

    result = {"prediction": im_array.tolist()}
    return result

# pcb 缺陷检测
@app.post('/pcb_detection')
async def detect(data_pcb: Data):
    img0 = data_pcb.img
    # 转为 numpy array
    img0 = np.array(img0)
    img0 = img0.astype("uint8")
    t1 = time.time()
    results = model_pcb(img0, conf=data_pcb.conf_thres, iou=data_pcb.iou_thres)  # 对图像进行预测
    print(f'Predict Done. ({time.time() - t1:.3f})')

    t1 = time.time()
    im_array = results[0].plot()  # plot a BGR numpy array of predictions
    print(f'Plot Done. ({time.time() - t1:.3f})')

    result = {"prediction": im_array.tolist()}
    return result

# 关键信息提取
@app.post('/kie')
async def detect(data_kie: DataKie):
    img0 = data_kie.img
    # 转为 numpy array
    img0 = np.array(img0)
    img0 = img0.astype("uint8")
    t1 = time.time()
    img_res = predict_kie_token_ser_re.main(parse_args(), img0) # 对图像进行预测
    # img_res = predict_kie_token_ser.main(parse_args(), img0) # 对图像进行预测
    print(f'Predict Done. ({time.time() - t1:.3f})')
    result = {"prediction": img_res.tolist()}
    return result

# 身份证信息抽取
@app.post('/id_kie')
async def detect(data_kie: DataKie):
    img0 = data_kie.img
    # 转为 numpy array
    img0 = np.array(img0)
    img0 = img0.astype("uint8")
    t1 = time.time()
    img_res = predict_kie_token_ser.main(parse_args_id_kie(), img0) # 对图像进行预测
    # img_res = predict_kie_token_ser.main(parse_args(), img0) # 对图像进行预测
    print(f'Predict Done. ({time.time() - t1:.3f})')
    result = {"prediction": img_res.tolist()}
    return result


# 资产异常行为识别
@app.post('/asset_detection')
async def detect(data_asset: Data):
    img0 = data_asset.img
    # 转为 numpy array
    img0 = np.array(img0)
    img0 = img0.astype("uint8")
    t1 = time.time()
    results = model_asset.predict(img0, conf=data_asset.conf_thres, iou=data_asset.iou_thres)  # 对图像进行预测
    print(f'Predict Done. ({time.time() - t1:.3f})')

    t1 = time.time()
    im_array = results[0].plot()  # plot a BGR numpy array of predictions
    print(f'Plot Done. ({time.time() - t1:.3f})')

    result = {"prediction": im_array.tolist()}
    return result

# 检测
@app.post('/license_detection')
async def detect(data_license: Data):
    img0 = data_license.img
    # 转为 numpy array
    img0 = np.array(img0)
    img0 = img0.astype("uint8")
    t1 = time.time()
    results = model_tobacco(img0, conf=data_license.conf_thres, iou=data_license.iou_thres)  # 对图像进行预测
    print(f'Predict Done. ({time.time() - t1:.3f})')

    t1 = time.time()
    im_array = results[0].plot()  # plot a BGR numpy array of predictions
    print(f'Plot Done. ({time.time() - t1:.3f})')

    result = {"prediction": im_array.tolist()}
    return result

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8501)

