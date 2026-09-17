"""
簡化版 Demo - 人臉年齡性別偵測
只有上傳圖片和顯示結果
"""
import argparse
import os
import gradio as gr
from ultralytics import YOLO
from PIL import Image


def predict(image, conf_threshold):
    """預測函數"""
    if image is None:
        return None

    # 執行預測
    results = model(image, conf=conf_threshold)

    # 繪製結果
    result_image = results[0].plot()

    # 轉換顏色格式 (BGR to RGB)
    result_image = Image.fromarray(result_image[..., ::-1])

    return result_image


def create_demo():
    """建立 Gradio 介面"""

    with gr.Blocks(title="人臉年齡性別偵測") as demo:

        gr.Markdown("# 👤 人臉年齡性別偵測")
        gr.Markdown("上傳圖片，系統會自動偵測人臉並預測年齡和性別")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="上傳圖片")
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="信心度閾值"
                )
                submit_btn = gr.Button("開始偵測", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="偵測結果")

        # 按鈕事件
        submit_btn.click(
            fn=predict,
            inputs=[input_image, conf_slider],
            outputs=output_image
        )

        # 範例（如果有的話）
        gr.Markdown("---")
        gr.Markdown("### 📝 說明")
        gr.Markdown("""
        - **類別**：10 個類別（男/女 × 5 個年齡組）
        - **信心度閾值**：越高表示只顯示更確定的預測結果
        - **顏色框**：不同類別會用不同顏色標示
        """)

    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人臉年齡性別偵測 Demo')
    parser.add_argument('--model', type=str,
                       default='results/train/weights/best.pt',
                       help='模型路徑')
    parser.add_argument('--port', type=int, default=7860,
                       help='埠號')

    args = parser.parse_args()

    # 檢查模型是否存在
    if not os.path.exists(args.model):
        print(f"[ERROR] 找不到模型: {args.model}")
        print("請先執行訓練: python train.py")
        exit(1)

    # 載入模型（全域變數，避免重複載入）
    print(f"[LOAD] 載入模型: {args.model}")
    model = YOLO(args.model)
    print("[OK] 模型載入完成")

    # 啟動 Demo
    print(f"[START] 啟動 Demo...")
    demo = create_demo()
    demo.launch(
        server_port=args.port,
        server_name="127.0.0.1",
        inbrowser=True
    )
