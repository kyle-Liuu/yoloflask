from flask import Flask, request, jsonify  
import numpy as np  
import cv2  
from ultralytics import YOLO  
import base64  
import requests  
from io import BytesIO  
  
app = Flask(__name__)  
  
# 加载YOLO模型  
model = YOLO("yolo11n.pt")  # 可以替换为您的模型路径  
  
def process_image(img):  
    # 使用YOLO模型进行检测  
    results = model(img)  
      
    # 提取检测结果  
    detections = []  
    for result in results:  
        boxes = result.boxes  
        for i, box in enumerate(boxes):  
            # 获取四角坐标 (xyxy格式)  
            x1, y1, x2, y2 = box.xyxy[0].tolist()  
              
            # 获取置信度分数  
            conf = float(box.conf[0])  
              
            # 获取类别ID和名称  
            cls_id = int(box.cls[0])  
            cls_name = result.names[cls_id]  
              
            # 添加到结果列表  
            detections.append({  
                "bbox_2d": [x1, y1, x2, y2],  # 左上角和右下角坐标  
                "confidence": conf,  
                "class_id": cls_id,  
                "class_name": cls_name  
            })  
      
    return detections  
  
@app.route('/detect', methods=['POST'])  
def detect():  
    # 检查请求类型  
    if 'image' in request.files:  
        # 文件上传  
        file = request.files['image']  
        img_bytes = file.read()  
        nparr = np.frombuffer(img_bytes, np.uint8)  
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
      
    elif 'url' in request.form:  
        # URL图像  
        url = request.form['url']  
        response = requests.get(url)  
        img_bytes = BytesIO(response.content).read()  
        nparr = np.frombuffer(img_bytes, np.uint8)  
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
      
    elif 'base64' in request.form:  
        # Base64编码图像  
        encoded_img = request.form['base64']  
        img_bytes = base64.b64decode(encoded_img)  
        nparr = np.frombuffer(img_bytes, np.uint8)  
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
      
    else:  
        return jsonify({"error": "No image provided"}), 400  
      
    # 处理图像并返回结果  
    detections = process_image(img)  
    return jsonify({"detections": detections})  
  
if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0', port=5000)