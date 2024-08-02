from PIL import Image
from ultralytics import YOLO

# Load a model
model = YOLO('./weights/dw_best.pt')  # pretrained YOLOv8n model
results = model("./datasets/dw/test/1_test_imagesa/0b8e3b35_b409_4f36_a13e_af138f9dde01.jpg", conf=0.25, iou=0.75)  # 对图像进行预测
i = 0
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    # im.save('results.jpg')  # save image
    print(i)
    i = i+1
    img = im_array[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    print(img.shape)
print(img.shape)


