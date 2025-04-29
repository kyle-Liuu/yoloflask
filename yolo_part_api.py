from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# 加载 YOLO 模型
model = YOLO('yolo11n.pt')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # 获取上传的图像文件
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()

        # 将图像字节转换为 OpenCV 图像
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 进行目标检测
        results = model(image)

        # 提取检测结果
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # 四角坐标
            scores = result.boxes.conf.cpu().numpy()  # 分数
            classes = result.boxes.cls.cpu().numpy().astype(int)  # 类别索引
            names = result.names  # 类别名称

            for box, score, cls in zip(boxes, scores, classes):
                detection = {
                    'bbox_2d': box.tolist(),
                    'confidence': float(score),
                    'class_id': cls,  
                    'class_name': names[cls]
                }
                detections.append(detection)

        return jsonify({'detections': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)