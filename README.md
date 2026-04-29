# Digit Recognizer - CNN Handwriting Recognition

基于 CNN 的手写数字识别系统，在 MNIST 数据集上达到 **99.96%** 验证准确率，支持 Gradio Web 界面。

## 项目成果

- **Kaggle 评分**: 0.997+
- **模型验证准确率**: 99.96% (5000 样本测试)
- **支持**: 手写画板 + 图片上传两种输入方式

## 项目结构

```
simple/
├── model_cnn_fold2.pth      # 最佳模型 (99.96% acc)
├── model_cnn_fold1.pth       # Top 3 模型
├── model_cnn_fold4.pth       # Top 3 模型
├── archived_models/          # 已归档的旧模型
├── web_app/                  # Gradio Web 应用
│   ├── app.py                # 主程序
│   ├── model_best.pth        # 当前使用模型
│   └── README.md             # Web 应用说明
├── src/                      # 训练代码
│   ├── train_cnn.py          # CNN 训练
│   ├── train_advanced.py     # 高级训练 (Label Smoothing + 增强)
│   ├── model_cnn.py          # CNN 模型定义
│   └── dataset.py            # 数据加载
├── digit-recognizer/         # Kaggle 数据集
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── CNNs手写数字识别实验报告.md  # 实验报告
├── .gitignore
└── README.md
```

## 模型架构

3 层卷积 + BatchNorm + Global Average Pooling:

```
Input (1×28×28)
├── Conv Block 1: Conv(1→32) + BN + ReLU + Conv(32→32) + BN + ReLU + MaxPool + Dropout(0.25)
├── Conv Block 2: Conv(32→64) + BN + ReLU + Conv(64→64) + BN + ReLU + MaxPool + Dropout(0.25)
├── Conv Block 3: Conv(64→128) + BN + ReLU + Conv(128→128) + BN + ReLU + MaxPool + Dropout(0.25)
├── Global Average Pooling
├── FC (128→256) + BN + ReLU + Dropout(0.5)
└── FC (256→10)
```

## 训练配置

| 配置项 | 值 |
|--------|-----|
| 优化器 | Adam |
| 学习率 | 0.001 |
| Batch Size | 64 |
| 训练 Epoch | 35 |
| 数据增强 | RandomAffine + RandomPerspective + Label Smoothing |
| Early Stopping | 是 (patience=5) |
| 学习率调度 | CosineAnnealingLR |
| 验证方法 | 5 折交叉验证 |

## 快速开始

### 1. 安装依赖
```bash
pip install torch numpy pandas pillow gradio
```

### 2. 启动 Web 应用
```bash
cd web_app
python app.py
```
访问 http://localhost:7860

### 3. 或直接使用模型进行预测
```python
import torch
from PIL import Image
import numpy as np

# 加载模型
model = CNN()
model.load_state_dict(torch.load('model_cnn_fold2.pth', weights_only=True))
model.eval()

# 预处理函数
def preprocess(img):
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype('float32') / 255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

# 预测
img = Image.open('digit.png').convert('L')
tensor = preprocess(img)
with torch.no_grad():
    pred = model(tensor).argmax().item()
print(f'Predicted: {pred}')
```

## Web 应用功能

- **手写画板**: 实时手写输入，auto-invert 自动适配白底/黑底
- **图片上传**: 支持上传手写数字图片
- **Top-3 预测**: 显示前三候选及置信度
- **概率分布**: 可视化各类别概率

## 模型对比

| 模型 | 验证准确率 | 状态 |
|------|-----------|------|
| model_cnn_fold2.pth | 99.96% | **使用** |
| model_cnn_fold4.pth | 99.90% | 保留 |
| model_cnn_fold1.pth | 99.90% | 保留 |
| 其他模型 | - | 已归档 |

## 技术栈

- Python 3.8+
- PyTorch 2.0+
- Gradio (Web 界面)
- scikit-learn (交叉验证)
- PIL (图像处理)

## License

MIT