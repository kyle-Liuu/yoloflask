import requests
import cv2
import numpy as np

def detect_and_draw_boxes(image_path, api_url="http://localhost:5000/detect"):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image file {image_path}")
        return

    # 将图片编码为 JPEG 格式
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    # 发送 POST 请求到 API
    files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        # 解析检测结果
        detections = response.json().get('detections', [])
        print(f"Detected {len(detections)} objects")

        # 绘制边界框
        for detection in detections:
            bbox = detection['bbox_2d']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # 解析边界框坐标
            x1, y1, x2, y2 = list(map(int, bbox))

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制类别和置信度
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 保存结果图片
        result_path = "result.jpg"
        cv2.imwrite(result_path, image)
        print(f"Result saved to {result_path}")

    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    image_path = "bus.jpg"  # 替换为你的图片路径
    detect_and_draw_boxes(image_path)