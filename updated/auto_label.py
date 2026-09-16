"""
Auto-Labeling Tool for Face Detection
從 IMDB-WIKI 人臉圖片檔名自動產生 YOLO 格式標註檔

檔名格式: {age}_{gender}_{other}_{timestamp}.jpg.chip.jpg
範例: 60_1_0_20170110141405608.jpg.chip.jpg
  - age: 60
  - gender: 1 (female) or 0 (male)

用法:
    python auto_label.py --input data/raw_images --output data/labels
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import cv2


def parse_filename(filename):
    """
    從檔名解析年齡和性別資訊

    Args:
        filename: 檔名，格式如 "60_1_0_20170110141405608.jpg.chip.jpg"

    Returns:
        age: 年齡
        gender: 性別 (0=male, 1=female)
        class_id: YOLO 類別 ID (0-9)
    """
    try:
        # 去掉副檔名
        name = filename.replace('.jpg.chip.jpg', '').replace('.jpg', '')

        # 分割
        parts = name.split('_')
        if len(parts) < 2:
            return None, None, None

        age = int(parts[0])
        gender = int(parts[1])

        # 判斷年齡組 (0-4)
        if age <= 20:
            age_group = 0
        elif age <= 40:
            age_group = 1
        elif age <= 60:
            age_group = 2
        elif age <= 80:
            age_group = 3
        else:
            age_group = 4

        # 計算類別 ID
        # Male: 0-4, Female: 5-9
        if gender == 0:  # Male
            class_id = age_group
        else:  # Female
            class_id = age_group + 5

        return age, gender, class_id

    except Exception as e:
        print(f"⚠️ 無法解析檔名 {filename}: {e}")
        return None, None, None


def create_yolo_label(image_path, class_id, output_dir):
    """
    為圖片建立 YOLO 格式標註檔

    YOLO 格式: <class_id> <x_center> <y_center> <width> <height>
    所有值都是相對於圖片大小的比例 (0-1)

    由於是人臉 chip (已裁切的人臉)，我們假設整張圖就是一個人臉
    """
    try:
        # 讀取圖片取得尺寸
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️ 無法讀取圖片: {image_path}")
            return False

        # 假設整張圖就是人臉，bbox 占整張圖的中心區域 (留一點邊界)
        x_center = 0.5
        y_center = 0.5
        width = 0.9  # 90% 的圖片寬度
        height = 0.9  # 90% 的圖片高度

        # 建立標註檔
        label_filename = image_path.stem + '.txt'
        label_path = output_dir / label_filename

        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        return True

    except Exception as e:
        print(f"❌ 建立標註失敗 {image_path}: {e}")
        return False


def batch_auto_label(input_dir, output_dir, limit=None):
    """
    批次自動標註

    Args:
        input_dir: 輸入圖片目錄
        output_dir: 輸出標註目錄
        limit: 處理數量限制 (None = 全部處理)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 取得所有圖片
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

    if limit:
        image_files = image_files[:limit]

    print(f"📁 輸入目錄: {input_dir}")
    print(f"📁 輸出目錄: {output_dir}")
    print(f"🖼️  找到 {len(image_files)} 張圖片")

    # 統計
    stats = {
        'success': 0,
        'failed': 0,
        'class_distribution': {i: 0 for i in range(10)}
    }

    # 處理每張圖片
    for img_file in tqdm(image_files, desc="自動標註"):
        age, gender, class_id = parse_filename(img_file.name)

        if class_id is None:
            stats['failed'] += 1
            continue

        # 建立標註
        if create_yolo_label(img_file, class_id, output_path):
            stats['success'] += 1
            stats['class_distribution'][class_id] += 1
        else:
            stats['failed'] += 1

    # 顯示統計
    print(f"\n✅ 標註完成！")
    print(f"   成功: {stats['success']}")
    print(f"   失敗: {stats['failed']}")
    print(f"\n📊 類別分佈:")

    class_names = [
        'Male 0-20', 'Male 21-40', 'Male 41-60', 'Male 61-80', 'Male 81+',
        'Female 0-20', 'Female 21-40', 'Female 41-60', 'Female 61-80', 'Female 81+'
    ]

    for class_id, count in stats['class_distribution'].items():
        if count > 0:
            percentage = count / stats['success'] * 100 if stats['success'] > 0 else 0
            print(f"   Class {class_id} ({class_names[class_id]}): {count} ({percentage:.1f}%)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='自動產生 YOLO 格式標註檔'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='輸入圖片目錄')
    parser.add_argument('--output', type=str, required=True,
                       help='輸出標註目錄')
    parser.add_argument('--limit', type=int, default=None,
                       help='處理數量限制 (測試用)')

    args = parser.parse_args()

    batch_auto_label(args.input, args.output, args.limit)

    print("🎉 完成！")


if __name__ == '__main__':
    main()
