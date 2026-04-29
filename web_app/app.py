"""
Digit Recognizer Web App
基于 CNN 的手写数字识别 - Google 风格界面
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr
import os

# ============== CNN 模型定义 ==============

class CNN(nn.Module):
    """CNN for MNIST digit classification."""
    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============== 模型加载 ==============

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path: str = 'model_best.pth'):
    """加载预训练模型"""
    model = CNN().to(DEVICE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights.")

    model.eval()
    return model


MODEL = load_model()


# ============== 图像预处理 ==============

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """预处理图像（用于手写画板，自动适应白底黑字/黑底白字）"""
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28), Image.LANCZOS)
    img_array = np.array(image)

    # 检查整体灰度来判断是白底黑字还是黑底白字
    # MNIST是黑底白字(背景0,前景255)，用户手写通常是白底黑字(背景255,前景0)
    # 如果平均灰度 > 127，说明是白底，需要反转成黑底白字
    if img_array.mean() > 127:
        img_array = 255 - img_array

    img_array = img_array.astype('float32') / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor


def preprocess_upload_image(image: Image.Image) -> torch.Tensor:
    """预处理上传图片（复杂版 - 处理各种格式）"""
    # 转换为灰度
    if image.mode != 'L':
        image = image.convert('L')

    img_array = np.array(image)

    # 1. 检测并裁剪白边（找到数字所在区域）
    row_sums = (img_array < 250).sum(axis=1)
    col_sums = (img_array < 250).sum(axis=0)
    rows = np.where(row_sums > 0)[0]
    cols = np.where(col_sums > 0)[0]

    if len(rows) > 0 and len(cols) > 0:
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        digit = img_array[r_min:r_max+1, c_min:c_max+1]

        # 2. 检查裁剪区域的mean来决定是否反转（白底黑字 vs 黑底白字）
        digit_mean = digit.mean()
        # MNIST需要白字黑底，所以黑底白字不需要处理，白底黑字需要反转
        if digit_mean <= 127:  # 黑字白底 → 反转成白字黑底
            digit = 255 - digit

        h, w = digit.shape
        max_dim = max(h, w)
        scale = 20.0 / max_dim

        new_h, new_w = int(h * scale), int(w * scale)
        digit_pil = Image.fromarray(digit)
        digit_small = np.array(digit_pil.resize((new_w, new_h), Image.LANCZOS))

        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_h = (28 - new_h) // 2
        start_w = (28 - new_w) // 2
        canvas[start_h:start_h+new_h, start_w:start_w+new_w] = digit_small
        img_array = canvas
    else:
        image = image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(image)
        mean_val = img_array.mean()
        if mean_val > 127:
            img_array = 255 - img_array

    # 3. 固定阈值二值化
    img_array = (img_array > 128).astype(np.uint8) * 255

    # 4. 归一化
    img_array = img_array.astype('float32') / 255.0
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor


# ============== 预测函数 ==============

def predict(model, image_tensor: torch.Tensor):
    """执行预测"""
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)

        top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
        top3_probs = top3_probs.squeeze().cpu().numpy()
        top3_indices = top3_indices.squeeze().cpu().numpy()

        return {
            'digit': int(predicted.item()),
            'confidence': float(conf.item()),
            'top3': [(int(idx), float(prob)) for idx, prob in zip(top3_indices, top3_probs)]
        }


def make_result_html(digit, conf, top3):
    """生成结果 HTML"""
    result_html = f"""
    <div style='text-align: center; padding: 3rem 2rem; background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px) saturate(180%); border-radius: 32px; box-shadow: 0 12px 40px rgba(94, 112, 84, 0.08); animation: float 6s ease-in-out infinite; border: 1px solid rgba(255, 255, 255, 0.8); margin-bottom: 2rem;'>
        <div style='font-size: 10rem; font-weight: 300; color: #2d3a27; line-height: 1; letter-spacing: -4px;'>
            {digit}
        </div>
        <div style='color: #5e7054; margin-top: 1.5rem; font-size: 1.2rem; font-weight: 400; letter-spacing: 2px;'>
            置信度 <span style='color: #2d3a27; font-weight: 600;'>{conf:.2%}</span>
        </div>
    </div>
    """

    top3_html = "<div style='padding: 2rem; background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(24px) saturate(180%); border-radius: 24px; box-shadow: 0 8px 32px rgba(94, 112, 84, 0.06); border: 1px solid rgba(255, 255, 255, 0.8); animation: float 7s ease-in-out infinite alternate;'>"
    top3_html += "<div style='font-weight: 500; color: #4a5a43; margin-bottom: 1.5rem; font-size: 1rem; letter-spacing: 2px; text-align: center;'>前三名预测</div>"
    for i, (d, p) in enumerate(top3):
        bar_width = int(p * 100)
        fill_color = '#2d3a27' if i == 0 else ('#5e7054' if i == 1 else '#8b9d77')
        top3_html += f"""
        <div style='margin: 1.2rem 0; display: flex; align-items: center; gap: 1.5rem;'>
            <div style='color: {fill_color}; font-size: 1.5rem; width: 30px; display: flex; justify-content: center; align-items: center; font-weight: 400;'>{d}</div>
            <div style='flex: 1; height: 6px; background: rgba(94, 112, 84, 0.1); border-radius: 10px; overflow: hidden;'>
                <div style='width: {bar_width}%; height: 100%; background: {fill_color}; border-radius: 10px; transition: width 1.2s cubic-bezier(0.22, 1, 0.36, 1);'></div>
            </div>
            <span style='color: #5e7054; width: 60px; text-align: right; font-weight: 500; font-family: monospace; font-size: 1.1rem;'>{p:.1%}</span>
        </div>
        """
    top3_html += "</div>"

    return result_html, top3_html


# ============== Gradio 界面 ==============

custom_css = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root, .dark, .gradio-container, .gradio-container.dark {
    --background-fill-primary: transparent !important;
    --background-fill-secondary: rgba(255, 255, 255, 0.4) !important;
    --border-color-primary: rgba(94, 112, 84, 0.2) !important;
    --block-background-fill: rgba(255, 255, 255, 0.45) !important;
    --block-border-color: rgba(255, 255, 255, 0.6) !important;
    --panel-background-fill: rgba(255, 255, 255, 0.45) !important;
    --input-background-fill: rgba(255, 255, 255, 0.5) !important;
    --input-background-fill-hover: rgba(255, 255, 255, 0.7) !important;
    --body-text-color: #2d3a27 !important;
    --block-label-text-color: #5e7054 !important;
    --block-title-text-color: #2d3a27 !important;
    --color-accent: #5e7054 !important;
    --color-accent-soft: rgba(94, 112, 84, 0.2) !important;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #eef2e6 0%, #d6e0ce 100%) !important;
    min-height: 100vh !important;
    color: #2d3a27 !important;
}

h1 {
    text-align: center !important;
    color: #2d3a27 !important;
    font-weight: 500 !important;
    font-size: 2.5rem !important;
    letter-spacing: -1px !important;
    margin-bottom: 0.5rem !important;
    animation: float 6s ease-in-out infinite alternate !important;
}

.description {
    text-align: center !important;
    color: #5e7054 !important;
    font-size: 1.1rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.5px !important;
    animation: float 7s ease-in-out infinite alternate-reverse !important;
}

/* Premium Frosted Glass Panels */
.panel, .tabs, .tab-nav, .gr-box, .gr-block {
    background: rgba(255, 255, 255, 0.45) !important;
    backdrop-filter: blur(24px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
    border-radius: 28px !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    box-shadow: 0 12px 40px rgba(94, 112, 84, 0.08) !important;
}

/* Floating effect for tabs */
.tabs {
    animation: float 8s ease-in-out infinite !important;
    transition: all 0.6s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.tabs:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 20px 50px rgba(94, 112, 84, 0.12) !important;
}

/* Minimalist Buttons - Matcha Theme */
button.primary {
    background: #4a5a43 !important;
    border: none !important;
    border-radius: 50px !important;
    color: white !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    box-shadow: 0 8px 24px rgba(74, 90, 67, 0.25) !important;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button.primary:hover {
    transform: translateY(-4px) scale(1.02) !important;
    box-shadow: 0 12px 32px rgba(74, 90, 67, 0.35) !important;
    background: #2d3a27 !important;
}

button.secondary {
    border-radius: 50px !important;
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    color: #4a5a43 !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button.secondary:hover {
    transform: translateY(-2px) !important;
    background: rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 8px 24px rgba(94, 112, 84, 0.1) !important;
}

/* Clean layout without fighting Gradio's internal DOM */
.gradio-container .bg-gray-800, 
.gradio-container .dark\:bg-gray-900 {
    background-color: transparent !important;
}

div[data-testid="sketchpad"] span, div[data-testid="sketchpad"] button, div[data-testid="image"] button, div[data-testid="image"] span {
    color: #4a5a43 !important;
}

/* Hide footer */
footer {
    display: none !important;
}

/* Apple Design Split Panels */
.apple-panel {
    background: rgba(255, 255, 255, 0.45) !important;
    backdrop-filter: blur(40px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(40px) saturate(200%) !important;
    border-radius: 36px !important;
    padding: 2.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
    box-shadow: 0 24px 48px rgba(94, 112, 84, 0.08), inset 0 2px 4px rgba(255, 255, 255, 0.4) !important;
}

.empty-result {
    height: 100%;
    min-height: 400px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #8b9d77;
    font-size: 1.2rem;
    font-weight: 400;
    letter-spacing: 1px;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 24px;
    border: 2px dashed rgba(94, 112, 84, 0.2);
}
"""


def predict_image_wrapper(img):
    """处理上传图片"""
    if img is None:
        return "<div class='empty-result'>请上传图片</div>", ""

    try:
        img_tensor = preprocess_upload_image(img)
        result = predict(MODEL, img_tensor)
        return make_result_html(result['digit'], result['confidence'], result['top3'])
    except Exception as e:
        return f"<div style='color: #d93025;'>预测失败: {str(e)}</div>", ""


def predict_sketch_wrapper(canvas):
    """处理手写画板"""
    if canvas is None:
        return "<div class='empty-result'>请在手写板上绘制数字</div>", ""

    try:
        # 如果是 dict，尝试获取 image
        if isinstance(canvas, dict):
            for key in ['image', 'composite', 'background']:
                if key in canvas and canvas[key] is not None:
                    canvas = canvas[key]
                    break
            else:
                return "<div class='empty-result'>请在手写板上绘制数字</div>", ""

        # 如果是 numpy array，转换为 PIL Image
        if isinstance(canvas, np.ndarray):
            if len(canvas.shape) == 3 and canvas.shape[2] == 3:
                canvas = Image.fromarray(canvas.astype('uint8'), 'RGB')
            elif len(canvas.shape) == 2:
                canvas = Image.fromarray(canvas.astype('uint8'), 'L')
            else:
                canvas = Image.fromarray(canvas.astype('uint8'))

        # 确保是灰度图
        if isinstance(canvas, Image.Image):
            if canvas.mode != 'L':
                canvas = canvas.convert('L')
        else:
            return f"<div style='color: #d93025;'>预测失败: 未知数据类型</div>", ""

        img_tensor = preprocess_image(canvas)
        result = predict(MODEL, img_tensor)
        return make_result_html(result['digit'], result['confidence'], result['top3'])
    except Exception as e:
        return f"<div style='color: #d93025;'>预测失败: {str(e)}</div>", ""


# ============== 构建界面 ==============

with gr.Blocks(title="手写数字识别系统") as app:
    gr.Markdown("""
    <div style="padding: 3rem 0 2rem 0;">
        <h1>手写数字识别</h1>
        <p class='description'>
            基于卷积神经网络 (CNN) 的智能手写数字识别与验证平台
        </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        # Left Panel - Inputs
        with gr.Column(scale=1, elem_classes=["apple-panel"]):
            gr.Markdown("<h3 style='color: #2d3a27; margin-bottom: 1rem; font-weight: 500; font-size: 1.5rem; text-align: center; letter-spacing: 2px;'>输入源</h3>")
            
            with gr.Tabs(elem_classes=["floating-element"]):
                with gr.TabItem("手写画板"):
                    with gr.Column(elem_classes="input-wrapper", scale=1):
                        canvas_input = gr.Sketchpad(
                            label="",
                            brush=gr.Brush(colors=["#2d3a27"]),
                            height=400,
                            width=400
                        )
                    with gr.Row():
                        clear_btn = gr.Button("清空画板", variant="secondary")
                        sketch_btn = gr.Button("开始识别", variant="primary", size="lg")

                with gr.TabItem("图片上传"):
                    with gr.Column(elem_classes="input-wrapper", scale=1):
                        img_input = gr.Image(label="", type="pil", height=400, width=400)
                    img_btn = gr.Button("开始识别", variant="primary", size="lg")

        # Right Panel - Shared Results
        with gr.Column(scale=1, elem_classes=["apple-panel"]):
            gr.Markdown("<h3 style='color: #2d3a27; margin-bottom: 1rem; font-weight: 500; font-size: 1.5rem; text-align: center; letter-spacing: 2px;'>分析结果</h3>")
            shared_result = gr.HTML("<div class='empty-result'>等待输入以进行识别...</div>")
            shared_top3 = gr.HTML("")

    # ============== 交互事件 ==============
    img_btn.click(
        fn=predict_image_wrapper,
        inputs=[img_input],
        outputs=[shared_result, shared_top3]
    )

    clear_btn.click(
        fn=lambda: (None, "<div class='empty-result'>等待输入以进行识别...</div>", ""),
        outputs=[canvas_input, shared_result, shared_top3]
    )

    sketch_btn.click(
        fn=predict_sketch_wrapper,
        inputs=[canvas_input],
        outputs=[shared_result, shared_top3]
    )

    gr.Markdown("""
    <div style='text-align: center; margin-top: 4rem; padding: 2rem; background: rgba(255,255,255,0.45); backdrop-filter: blur(24px) saturate(180%); border-radius: 28px; border: 1px solid rgba(255,255,255,0.8); box-shadow: 0 12px 40px rgba(94, 112, 84, 0.08); animation: float 8s ease-in-out infinite alternate-reverse;'>
        <div style='color: #2d3a27; font-weight: 500; font-size: 1rem; letter-spacing: 2px;'>准确率 99.5%+</div>
        <div style='color: #5e7054; font-size: 0.85rem; margin-top: 0.8rem; font-weight: 300; letter-spacing: 1px;'>基于 PyTorch CNN · Gradio 强力驱动</div>
    </div>
    """)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=custom_css,
        theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inter"), "sans-serif"])
    )