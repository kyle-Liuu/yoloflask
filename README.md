# YOLO 目标检测 API 服务

这是一个基于 YOLO 和 Flask 的目标检测 API 服务项目。该项目提供了一个简单的 HTTP API 接口，可以对上传的图像进行目标检测，并返回检测结果。

## 功能特点

- 支持多种图像输入方式：
  - 本地文件上传
  - 图片 URL
  - Base64 编码图像
- 返回 JSON 格式的检测结果
- 提供可视化工具展示检测结果
- 支持批量处理
- 高性能实时检测

## 环境要求

- Python 3.8+
- Flask 3.1.0
- OpenCV 4.11.0
- Ultralytics 8.3.119
- 其他依赖见 `requirements.txt`

## 安装说明

1. 克隆项目到本地：

```bash
git clone [项目地址]
cd yoloflask
```

2. 创建并激活虚拟环境（推荐）：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 启动服务器

1. 运行 all_API 服务：支持 all 和 part 两种客户端

```bash
python yolo_all_api.py
```

2. 运行 part_API 服务：仅支持 part 客户端

```bash
python yolo_part_api.py
```

服务器将在 `http://localhost:5000` 启动。

### 使用 API

#### 1. 使用 curl 发送请求：

```bash
curl -X POST -F "image=@bus.jpg" http://localhost:5000/detect
```

#### 2. 使用 Python 客户端：

简单客户端（yolo_part_client.py）：

```python
python yolo_part_client.py
```

简单客户端（yolo_part_client_result.py）：自动保存检测结果

```python
python yolo_part_client_result.py
```

完整客户端（yolo_all_client.py）：

```python
python yolo_all_client.py
```

### API 响应格式

成功响应示例：

```json
{
  "detections": [
    {
      "bbox_2d": [x1, y1, x2, y2],
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95
    }
  ]
}
```

## 项目说明

```
yoloflask/
│
├── API 服务器
│   ├── yolo_all_api.py        # 完整版 API 服务器，支持多种输入方式
│   └── yolo_part_api.py       # 简化版 API 服务器
│
├── 客户端程序
│   ├── yolo_all_client.py              # 完整版客户端示例
│   ├── yolo_part_client.py             # 简化版客户端示例
│   ├── yolo_part_client_result.py      # 简化保存版客户端示例
│   └── curl.txt                        # curl 命令示例
│
├── 示例文件
│   ├── bus.jpg                 # 检测图片示例
│   ├── output_bus.jpg          # 完整版客户端检测结果示例
│   └── result.jpg              # 简化保存版客户端检测结果示例
│
├── 配置文件
│   ├── requirements.txt         # 项目依赖列表
│   └── yolo11n.pt               # YOLO 模型文件
│
└── README.md                # 项目说明文档
```

## 注意事项

1. 确保有足够的系统内存和 GPU（如果使用）
2. 第一次运行时会自动下载 YOLO 模型
3. 建议在处理大量图片时使用批处理模式
4. API 服务默认监听所有网络接口，如需更改请修改相应的 host 和 port 参数

## 常见问题

1. 如果遇到模型加载错误，请检查模型文件是否存在
2. 如果遇到内存错误，请减小批处理大小
3. 如果需要自定义模型，请修改模型路径参数
