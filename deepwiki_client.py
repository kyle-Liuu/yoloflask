import requests  
import json  
import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
  
def call_detection_api(image_path):  
    # 准备图像文件  
    with open(image_path, 'rb') as f:  
        files = {'image': f}  
          
        # 发送请求到API  
        response = requests.post('http://localhost:5000/detect', files=files)  
          
    # 解析响应  
    if response.status_code == 200:  
        result = response.json()  
        return result['detections']  
    else:  
        print(f"Error: {response.status_code}")  
        return None  
  
def visualize_detections(image_path, detections, output_path=None):  
    # 读取图像  
    img = cv2.imread(image_path)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
      
    # 创建图形  
    fig, ax = plt.subplots(1)  
    ax.imshow(img)  
      
    # 绘制检测框  
    for det in detections:  
        x1, y1, x2, y2 = det['bbox_2d']  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')  
        ax.add_patch(rect)  
          
        # 添加标签  
        label = f"{det['class_name']} ({det['confidence']:.2f})"  
        ax.text(x1, y1-5, label, color='white', backgroundcolor='red', fontsize=8)  
      
    plt.axis('off')
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"图片已保存至: {output_path}")
    
    plt.show()  
  
# 使用示例  
image_path = 'bus.jpg'  
detections = call_detection_api(image_path)  
if detections:  
    # 添加输出路径参数
    output_path = 'output_' + image_path
    visualize_detections(image_path, detections, output_path)  
    print(json.dumps(detections, indent=2))