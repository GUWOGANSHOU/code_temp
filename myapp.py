import time
import numpy as np
import streamlit as st
import requests
from PIL import Image, ImageOps

# st.sidebar 下的内容会被渲染到侧边栏
sidebar = st.sidebar
# 使用对象表示法添加选择框
option  = st.sidebar.selectbox(
    "",
    ("安全帽检测", "电网绝缘手套检测", "PCB缺陷检测", "关键信息抽取", "身份证识别", "资产异常行为识别", "经营户亮证识别"),
    label_visibility='collapsed'
)
# 默认渲染到主界面
# 设置标题
st.title(option)


if option == '安全帽检测':
    c1, c2 = st.columns(spec=2)
    pred_img = ''
    with c1:
        # 上传图片并展示
        uploaded_file = st.sidebar.file_uploader("", type=["jpg", "png", "bmp", "jpeg"])

        if uploaded_file:
            # # 将传入的文件转为Opencv格式
            # file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            # print("File size:", len(file_bytes))
            # opencv_image = cv2.imdecode(file_bytes, 1)
            original_image = Image.open(uploaded_file).convert("RGB")
            print("压缩前文件大小：", original_image.size)
            original_image.thumbnail((640, 640), Image.BICUBIC)
            print(original_image)
            print("压缩后文件大小：", original_image.size)
            original_image = ImageOps.exif_transpose(original_image) #恢复正常角度的图像
            print("恢复正常角度后文件大小：", original_image.size)
            st.image(original_image)
            img = np.array(original_image)
            print(img.shape)
            # # 展示图片
            # image_cv2 = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            # st.image(image_cv2, channels="RGB")
            # 保存图片
            # cv2.imwrite('test.jpg', opencv_image)
            conf_thres = st.slider('conf_thres', 0.1, 0.9, 0.25)
            iou_thres = st.slider('iou_thres', 0.1, 0.9, 0.45)

            if st.button("检测") & (img.size != 0):
                data_bin = {'img': img.tolist(), 'conf_thres': conf_thres, 'iou_thres': iou_thres}
                t1 = time.time()
                # 请求/相应
                json_response = requests.post("http://127.0.0.1:8501/helmet_detection", json=data_bin).json()
                pred_img = json_response.get("prediction")
                print(f'Post Done. ({time.time() - t1:.3f})')
    with c2:
        if pred_img != '':
            st.image(np.array(pred_img), channels="RGB")

elif option == '电网绝缘手套检测':
    c1, c2 = st.columns(spec=2)
    pred_img = ''
    with c1:
        # 上传图片并展示
        uploaded_file = st.sidebar.file_uploader("", type=["jpg", "png", "bmp", "jpeg"])

        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            print("压缩前文件大小：", original_image.size)
            original_image.thumbnail((640, 640), Image.BICUBIC)
            print(original_image)
            print("压缩后文件大小：", original_image.size)
            original_image = ImageOps.exif_transpose(original_image)  # 恢复正常角度的图像
            print("恢复正常角度后文件大小：", original_image.size)
            st.image(original_image)
            img = np.array(original_image)
            print(img.shape)

            conf_thres = st.slider('conf_thres', 0.1, 0.9, 0.25)
            iou_thres = st.slider('iou_thres', 0.1, 0.9, 0.7)
            if st.button("检测") & (img.size != 0):
                data_bin = {'img': img.tolist(), 'conf_thres': conf_thres, 'iou_thres': iou_thres}
                t1 = time.time()
                # 请求/相应
                json_response = requests.post("http://127.0.0.1:8501/dw_detection", json=data_bin).json()
                pred_img = json_response.get("prediction")
                print(f'Post Done. ({time.time() - t1:.3f})')
    with c2:
        if pred_img != '':
            t1 = time.time()
            st.image(np.array(pred_img), channels="RGB")
            print(f'st plot image Done. ({time.time() - t1:.3f})')


elif option == 'PCB缺陷检测':
    c1, c2 = st.columns(spec=2)
    pred_img = ''
    with c1:
        # 上传图片并展示
        uploaded_file = st.sidebar.file_uploader("", type=["jpg", "png", "bmp", "jpeg"])
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            print("压缩前文件大小：", original_image.size)
            original_image.thumbnail((640, 640), Image.BICUBIC)
            print(original_image)
            print("压缩后文件大小：", original_image.size)
            original_image = ImageOps.exif_transpose(original_image)  # 恢复正常角度的图像
            print("恢复正常角度后文件大小：", original_image.size)
            st.image(original_image)
            img = np.array(original_image)
            print(img.shape)


            conf_thres = st.slider('conf_thres', 0.1, 0.9, 0.25)
            iou_thres = st.slider('iou_thres', 0.1, 0.9, 0.7)
            if st.button("检测") & (img.size != 0):
                data_bin = {'img': img.tolist(), 'conf_thres': conf_thres, 'iou_thres': iou_thres}
                # 请求/响应
                t1 = time.time()
                json_response = requests.post("http://127.0.0.1:8501/pcb_detection", json=data_bin).json()
                print(f'Post Done. ({time.time() - t1:.3f})')
                pred_img = json_response.get("prediction")
    with c2:
        if pred_img != '':
            t1 = time.time()
            st.image(np.array(pred_img), channels="RGB")
            print(f'st plot image Done. ({time.time() - t1:.3f})')

elif option == '关键信息抽取':
    c1, c2 = st.columns(spec=2)
    pred_img = ''
    with c1:
        # 上传图片并展示
        uploaded_file = st.sidebar.file_uploader("", type=["jpg", "png", "bmp", "jpeg"])
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            print("压缩前文件大小：", original_image.size)
            original_image.thumbnail((640, 640), Image.BICUBIC)
            print(original_image)
            print("压缩后文件大小：", original_image.size)
            original_image = ImageOps.exif_transpose(original_image)  # 恢复正常角度的图像
            print("恢复正常角度后文件大小：", original_image.size)
            st.image(original_image)
            img = np.array(original_image)
            print(img.shape)

            if st.button("检测") & (img.size != 0):
                data_bin = {'img': img.tolist()}
                # 请求/响应
                t1 = time.time()
                json_response = requests.post("http://127.0.0.1:8501/kie", json=data_bin).json()
                print(f'Post Done. ({time.time() - t1:.3f})')
                pred_img = json_response.get("prediction")
    with c2:
        if pred_img != '':
            t1 = time.time()
            st.image(np.array(pred_img), channels="RGB")
            print(f'st plot image Done. ({time.time() - t1:.3f})')

elif option == '身份证识别':
    c1, c2 = st.columns(spec=2)
    pred_img = ''
    with c1:
        # 上传图片并展示
        uploaded_file = st.sidebar.file_uploader("", type=["jpg", "png", "bmp", "jpeg"])
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            print("压缩前文件大小：", original_image.size)
            original_image.thumbnail((640, 640), Image.BICUBIC)
            print(original_image)
            print("压缩后文件大小：", original_image.size)
            original_image = ImageOps.exif_transpose(original_image)  # 恢复正常角度的图像
            print("恢复正常角度后文件大小：", original_image.size)
            st.image(original_image)
            img = np.array(original_image)
            print(img.shape)

            if st.button("检测") & (img.size != 0):
                data_bin = {'img': img.tolist()}
                # 请求/响应
                t1 = time.time()
                json_response = requests.post("http://127.0.0.1:8501/id_kie", json=data_bin).json()
                print(f'Post Done. ({time.time() - t1:.3f})')
                pred_img = json_response.get("prediction")
    with c2:
        if pred_img != '':
            t1 = time.time()
            st.image(np.array(pred_img), channels="RGB")
            print(f'st plot image Done. ({time.time() - t1:.3f})')

elif option == '资产异常行为识别':
    c1, c2 = st.columns(spec=2)
    pred_img = ''
    with c1:
        # 上传图片并展示
        uploaded_file = st.sidebar.file_uploader("", type=["jpg", "png", "bmp", "jpeg"])
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            print("压缩前文件大小：", original_image.size)
            original_image.thumbnail((640, 640), Image.BICUBIC)
            print(original_image)
            print("压缩后文件大小：", original_image.size)
            original_image = ImageOps.exif_transpose(original_image)  # 恢复正常角度的图像
            print("恢复正常角度后文件大小：", original_image.size)
            st.image(original_image)
            img = np.array(original_image)
            print(img.shape)


            conf_thres = st.slider('conf_thres', 0.1, 0.9, 0.25)
            iou_thres = st.slider('iou_thres', 0.1, 0.9, 0.7)
            if st.button("检测") & (img.size != 0):
                data_bin = {'img': img.tolist(), 'conf_thres': conf_thres, 'iou_thres': iou_thres}
                # 请求/响应
                t1 = time.time()
                json_response = requests.post("http://127.0.0.1:8501/asset_detection", json=data_bin).json()
                print(f'Post Done. ({time.time() - t1:.3f})')
                pred_img = json_response.get("prediction")
    with c2:
        if pred_img != '':
            t1 = time.time()
            st.image(np.array(pred_img), channels="RGB")
            print(f'st plot image Done. ({time.time() - t1:.3f})')

elif option == '经营户亮证识别':
    c1, c2 = st.columns(spec=2)
    pred_img = ''
    with c1:
        # 上传图片并展示
        uploaded_file = st.sidebar.file_uploader("", type=["jpg", "png", "bmp", "jpeg"])
        if uploaded_file:
            original_image = Image.open(uploaded_file).convert("RGB")
            print("压缩前文件大小：", original_image.size)
            original_image.thumbnail((640, 640), Image.BICUBIC)
            print(original_image)
            print("压缩后文件大小：", original_image.size)
            original_image = ImageOps.exif_transpose(original_image)  # 恢复正常角度的图像
            print("恢复正常角度后文件大小：", original_image.size)
            st.image(original_image)
            img = np.array(original_image)
            print(img.shape)


            conf_thres = st.slider('conf_thres', 0.1, 0.9, 0.25)
            iou_thres = st.slider('iou_thres', 0.1, 0.9, 0.7)
            if st.button("检测") & (img.size != 0):
                data_bin = {'img': img.tolist(), 'conf_thres': conf_thres, 'iou_thres': iou_thres}
                # 请求/响应
                t1 = time.time()
                json_response = requests.post("http://127.0.0.1:8501/license_detection", json=data_bin).json()
                print(f'Post Done. ({time.time() - t1:.3f})')
                pred_img = json_response.get("prediction")
    with c2:
        if pred_img != '':
            t1 = time.time()
            st.image(np.array(pred_img), channels="RGB")
            print(f'st plot image Done. ({time.time() - t1:.3f})')