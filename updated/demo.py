"""
Face Age-Gender Detection - Gradio Demo
互動式人臉年齡性別偵測 Web 介面

用法:
    python demo.py
    python demo.py --model results/train/weights/best.pt
"""
import argparse
import os
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np


# 類別名稱和顏色
CLASS_INFO = {
    0: {'name': '男性 0-20歲', 'color': '#4A90E2'},
    1: {'name': '男性 21-40歲', 'color': '#5C6BC0'},
    2: {'name': '男性 41-60歲', 'color': '#7E57C2'},
    3: {'name': '男性 61-80歲', 'color': '#9575CD'},
    4: {'name': '男性 81歲以上', 'color': '#B39DDB'},
    5: {'name': '女性 0-20歲', 'color': '#EC407A'},
    6: {'name': '女性 21-40歲', 'color': '#F06292'},
    7: {'name': '女性 41-60歲', 'color': '#F48FB1'},
    8: {'name': '女性 61-80歲', 'color': '#F8BBD0'},
    9: {'name': '女性 81歲以上', 'color': '#FCE4EC'}
}


class FaceAgeGenderDetector:
    def __init__(self, model_path):
        """初始化偵測器"""
        print(f"📦 載入模型: {model_path}")
        self.model = YOLO(model_path)

    def detect(self, image, conf_threshold=0.25):
        """
        偵測人臉並分類年齡性別

        Args:
            image: PIL Image
            conf_threshold: 信心度閾值

        Returns:
            result_image: 標註後的圖片
            summary: 文字說明
            statistics: 統計資訊
        """
        # 預測
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        # 準備輸出
        result_img = result.plot()  # 繪製偵測框
        total_faces = len(boxes)

        # 統計資訊
        stats = {
            'total': total_faces,
            'male': 0,
            'female': 0,
            'age_groups': {
                '0-20': 0,
                '21-40': 0,
                '41-60': 0,
                '61-80': 0,
                '81+': 0
            }
        }

        # 詳細結果
        summary_lines = [f"🔍 偵測到 {total_faces} 張人臉\n"]

        if total_faces == 0:
            summary_lines.append("❌ 未偵測到人臉，請上傳包含人臉的清晰圖片")
            return result_img, "\n".join(summary_lines), self._format_stats(stats)

        # 處理每個偵測結果
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # 取得類別資訊
            class_info = CLASS_INFO.get(cls, {'name': f'Class {cls}'})
            class_name = class_info['name']

            # 更新統計
            if cls < 5:  # Male
                stats['male'] += 1
            else:  # Female
                stats['female'] += 1

            # 年齡組統計
            age_group_idx = cls % 5
            age_groups = ['0-20', '21-40', '41-60', '61-80', '81+']
            stats['age_groups'][age_groups[age_group_idx]] += 1

            # 記錄結果
            gender_emoji = "👨" if cls < 5 else "👩"
            summary_lines.append(
                f"{gender_emoji} 人臉 {i+1}: {class_name} (信心度: {conf:.1%})"
            )

        return result_img, "\n".join(summary_lines), self._format_stats(stats)

    def _format_stats(self, stats):
        """格式化統計資訊"""
        if stats['total'] == 0:
            return "無統計資料"

        lines = [
            "📊 統計資訊\n",
            f"總人數: {stats['total']} 人",
            f"👨 男性: {stats['male']} 人 ({stats['male']/stats['total']:.1%})",
            f"👩 女性: {stats['female']} 人 ({stats['female']/stats['total']:.1%})",
            "\n年齡分佈:"
        ]

        for age_group, count in stats['age_groups'].items():
            if count > 0:
                percentage = count / stats['total']
                lines.append(f"  {age_group}歲: {count} 人 ({percentage:.1%})")

        return "\n".join(lines)


def create_demo(model_path):
    """建立 Gradio 介面"""
    detector = FaceAgeGenderDetector(model_path)

    def predict(image, conf_threshold):
        """預測函數"""
        if image is None:
            return None, "請上傳圖片", ""

        result_img, summary, stats = detector.detect(image, conf_threshold)
        return result_img, summary, stats

    # 建立介面
    with gr.Blocks(theme=gr.themes.Soft(), title="人臉年齡性別偵測") as demo:
        gr.Markdown("""
        # 👤 Face Age-Gender Detection
        ### 人臉年齡性別偵測系統

        使用 YOLOv8m 模型，可同時偵測人臉並分類 **10 個年齡性別類別**：
        - 👨 男性：0-20、21-40、41-60、61-80、81+ 歲
        - 👩 女性：0-20、21-40、41-60、61-80、81+ 歲
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="上傳圖片")
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="信心度閾值",
                    info="調整偵測的信心度門檻"
                )
                predict_btn = gr.Button("🔍 開始偵測", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="偵測結果")

        with gr.Row():
            with gr.Column():
                summary_text = gr.Textbox(
                    label="偵測詳情",
                    lines=10,
                    placeholder="偵測結果將顯示在這裡..."
                )
            with gr.Column():
                stats_text = gr.Textbox(
                    label="統計資訊",
                    lines=10,
                    placeholder="統計資訊將顯示在這裡..."
                )

        # 事件綁定
        predict_btn.click(
            fn=predict,
            inputs=[input_image, conf_slider],
            outputs=[output_image, summary_text, stats_text]
        )

        gr.Markdown("""
        ---
        **技術規格:**
        - 模型: YOLOv8m
        - 訓練資料: IMDB-WIKI Face Chips
        - 類別數: 10 (5 age groups × 2 genders)

        **使用說明:**
        1. 上傳包含人臉的圖片
        2. 調整信心度閾值（可選）
        3. 點擊「開始偵測」
        4. 查看標註結果和統計資訊
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(
        description='啟動人臉年齡性別偵測 Gradio Demo'
    )
    parser.add_argument('--model', type=str,
                       default='results/train/weights/best.pt',
                       help='模型路徑')
    parser.add_argument('--share', action='store_true',
                       help='建立公開分享連結')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port 號')

    args = parser.parse_args()

    # 檢查模型
    if not os.path.exists(args.model):
        print(f"❌ 找不到模型: {args.model}")
        print(f"請先執行訓練: python train.py")
        return

    # 建立並啟動 demo
    print(f"🚀 啟動 Gradio Demo...")
    demo = create_demo(args.model)
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == '__main__':
    main()
