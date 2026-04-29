# Digit Recognizer Web App

基于 CNN 的手写数字识别 Web 应用

## 技术栈

- Python 3.8+
- PyTorch (CNN 模型)
- Gradio (Web 界面)

## 项目结构

```
web_app/
├── app.py              # Web 应用入口
├── model_best.pth      # 最佳模型权重 (99.96% val accuracy)
├── requirements.txt    # 依赖列表
└── README.md           # 项目说明
```

## 模型信息

- **模型架构**: CNN (3层卷积 + BatchNorm + GlobalAvgPool)
- **验证准确率**: 99.96%
- **输入**: 28x28 灰度图像
- **输出**: 0-9 数字分类

## 本地运行

```bash
cd web_app
pip install -r requirements.txt
python app.py
```

访问 http://localhost:7860

## 功能

- 📤 图片上传识别
- ✏️ 手写画板输入
- 📊 预测置信度展示
- 🔄 支持多次连续识别

## 界面预览

- Google 风格极简设计
- 圆角卡片布局
- 柔和阴影效果
- 强调色点缀