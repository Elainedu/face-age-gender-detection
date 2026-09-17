"""
處理 WIKI 資料集
從 wiki.mat 讀取年齡性別資訊，並產生 YOLO 格式標註
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2
import scipy.io
import argparse


def calculate_age(dob, photo_taken):
    """計算年齡"""
    try:
        ordinal = int(dob) - 366
        if ordinal < 1:
            return None
        birth_date = datetime.fromordinal(ordinal)
        age = photo_taken - birth_date.year
        if age < 0 or age > 120:
            return None
        return age
    except Exception:
        return None


def get_age_group(age):
    """
    判斷年齡組 (0-4)
    0: 0-20
    1: 21-40
    2: 41-60
    3: 61-80
    4: 81+
    """
    if age <= 20:
        return 0
    elif age <= 40:
        return 1
    elif age <= 60:
        return 2
    elif age <= 80:
        return 3
    else:
        return 4


def get_class_id(age, gender):
    """
    計算 YOLO 類別 ID
    Male (gender=1): 0-4
    Female (gender=0): 5-9
    """
    age_group = get_age_group(age)

    if gender == 1.0:  # Male
        return age_group
    else:  # Female (gender == 0.0)
        return age_group + 5


def create_yolo_label(image_path, class_id, output_dir):
    """建立 YOLO 格式標註檔"""
    try:
        # 讀取圖片
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        # 整張圖作為人臉區域
        x_center = 0.5
        y_center = 0.5
        width = 0.9
        height = 0.9

        # 建立標註檔
        label_filename = image_path.stem + '.txt'
        label_path = output_dir / label_filename

        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        return True

    except Exception as e:
        return False


def process_wiki_dataset(wiki_dir='data/raw_images',
                         output_dir='data/processed',
                         limit=None,
                         min_age=0,
                         max_age=100):
    """
    處理 WIKI 資料集

    Args:
        wiki_dir: WIKI 資料集目錄（包含 wiki.mat）
        output_dir: 輸出目錄
        limit: 處理數量限制
        min_age: 最小年齡
        max_age: 最大年齡
    """
    wiki_path = Path(wiki_dir)
    output_path = Path(output_dir)

    # 讀取 metadata
    mat_file = wiki_path / 'wiki.mat'
    if not mat_file.exists():
        print(f"[ERROR] 找不到 wiki.mat 檔案: {mat_file}")
        return

    print("[INFO] 讀取 wiki.mat...")
    mat = scipy.io.loadmat(str(mat_file))

    # 提取資料
    wiki = mat['wiki'][0, 0]
    dob = wiki['dob'][0]  # 出生日期
    photo_taken = wiki['photo_taken'][0]  # 拍攝年份
    full_path = wiki['full_path'][0]  # 圖片路徑
    gender = wiki['gender'][0]  # 性別

    print(f"[INFO] 資料集包含 {len(full_path)} 筆資料")

    # 建立輸出目錄
    img_output = output_path / 'images'
    label_output = output_path / 'labels'
    img_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    # 統計
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'invalid_age': 0,
        'missing_file': 0,
        'class_distribution': {i: 0 for i in range(10)}
    }

    # 處理每張圖片
    process_count = min(len(full_path), limit) if limit else len(full_path)

    print(f"[INFO] 開始處理 {process_count} 張圖片...")

    for i in tqdm(range(process_count), desc="處理圖片"):
        stats['total'] += 1

        # 取得圖片路徑
        img_path_str = full_path[i][0]
        img_path = wiki_path / img_path_str

        # 檢查檔案是否存在
        if not img_path.exists():
            stats['missing_file'] += 1
            continue

        # 計算年齡
        if dob[i] == 0 or photo_taken[i] == 0:
            stats['invalid_age'] += 1
            continue

        age = calculate_age(dob[i], photo_taken[i])

        # 過濾年齡
        if age is None or age < min_age or age > max_age:
            stats['invalid_age'] += 1
            continue

        # 取得性別
        img_gender = gender[i]
        if img_gender != 0.0 and img_gender != 1.0:
            stats['failed'] += 1
            continue

        # 計算類別 ID
        class_id = get_class_id(age, img_gender)

        # 複製圖片
        new_img_name = f"{img_path.stem}.jpg"
        new_img_path = img_output / new_img_name

        try:
            shutil.copy2(img_path, new_img_path)

            # 建立標註
            if create_yolo_label(new_img_path, class_id, label_output):
                stats['success'] += 1
                stats['class_distribution'][class_id] += 1
            else:
                stats['failed'] += 1

        except Exception as e:
            stats['failed'] += 1

    # 顯示統計
    print(f"\n[OK] 處理完成！")
    print(f"   總數: {stats['total']}")
    print(f"   成功: {stats['success']}")
    print(f"   失敗: {stats['failed']}")
    print(f"   年齡無效: {stats['invalid_age']}")
    print(f"   檔案缺失: {stats['missing_file']}")

    print(f"\n[STATS] 類別分佈:")
    class_names = [
        'Male 0-20', 'Male 21-40', 'Male 41-60', 'Male 61-80', 'Male 81+',
        'Female 0-20', 'Female 21-40', 'Female 41-60', 'Female 61-80', 'Female 81+'
    ]

    for class_id, count in stats['class_distribution'].items():
        if count > 0:
            percentage = count / stats['success'] * 100 if stats['success'] > 0 else 0
            print(f"   Class {class_id} ({class_names[class_id]}): {count} ({percentage:.1f}%)")

    print(f"\n[DIR] 輸出位置:")
    print(f"   圖片: {img_output}")
    print(f"   標籤: {label_output}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='處理 WIKI 資料集')
    parser.add_argument('--input', type=str, default='data/raw_images',
                       help='WIKI 資料集目錄')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='輸出目錄')
    parser.add_argument('--limit', type=int, default=None,
                       help='處理數量限制')
    parser.add_argument('--min-age', type=int, default=0,
                       help='最小年齡')
    parser.add_argument('--max-age', type=int, default=100,
                       help='最大年齡')

    args = parser.parse_args()

    process_wiki_dataset(
        wiki_dir=args.input,
        output_dir=args.output,
        limit=args.limit,
        min_age=args.min_age,
        max_age=args.max_age
    )

    print("\n[DONE] 完成！")


if __name__ == '__main__':
    main()
