"""
修正 YOLO 標籤格式

舊格式 (錯誤): coco_class_id  gender  age  x  y  w  h   ← 7個欄位，像素座標
新格式 (正確): age_gender_class  x_center  y_center  width  height  ← 5個欄位，正規化 0~1

類別對應:
  男性 (gender=0): male0-20=0, male21-40=1, male41-60=2, male61-80=3, male81+=4
  女性 (gender=1): female0-20=5, female21-40=6, female41-60=7, female61-80=8, female81+=9
"""
import os
from PIL import Image


def get_age_gender_class(age_int, gender_int):
    if age_int <= 20:
        age_group = 0
    elif age_int <= 40:
        age_group = 1
    elif age_int <= 60:
        age_group = 2
    elif age_int <= 80:
        age_group = 3
    else:
        age_group = 4

    return age_group + 5 if gender_int == 1 else age_group


def find_image(label_name, images_dir):
    """找對應的圖片檔案"""
    for f in os.listdir(images_dir):
        name_no_ext = f.replace('.jpg.chip.jpg', '').replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        if name_no_ext == label_name:
            return os.path.join(images_dir, f)
    return None


def fix_label_file(label_path, images_dir):
    """修正單一標籤檔案"""
    label_name = os.path.basename(label_path).replace('.txt', '')
    parts = label_name.split('_')

    # 從檔名解析年齡和性別
    try:
        age = int(parts[0])
        gender = int(parts[1])
    except (ValueError, IndexError):
        print(f"  [SKIP] 無法解析檔名: {label_name}")
        return False

    new_cls = get_age_gender_class(age, gender)

    # 取得圖片尺寸
    image_path = find_image(label_name, images_dir)
    if image_path:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    else:
        img_w, img_h = 200, 200  # chip 圖片預設尺寸

    # 讀取舊標籤
    with open(label_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        vals = line.strip().split()
        if len(vals) < 7:
            continue

        # 舊格式: coco_cls  gender  age  x  y  w  h
        # 其中 x = x1 + x2/2，w = x2，y = y1 + y2/2，h = y2（計算錯誤的原始碼）
        try:
            x_val = float(vals[3])
            y_val = float(vals[4])
            w_val = float(vals[5])  # 實際上是 x2
            h_val = float(vals[6])  # 實際上是 y2
        except ValueError:
            continue

        # 還原 xyxy 座標
        x2 = w_val
        y2 = h_val
        x1 = x_val - x2 / 2
        y1 = y_val - y2 / 2

        # 正規化到 0~1
        x_center = max(0.0, min(1.0, (x1 + x2) / 2 / img_w))
        y_center = max(0.0, min(1.0, (y1 + y2) / 2 / img_h))
        width    = max(0.0, min(1.0, (x2 - x1) / img_w))
        height   = max(0.0, min(1.0, (y2 - y1) / img_h))

        new_lines.append(f"{new_cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 如果偵測結果為空，用整張圖作為 bbox（chip 圖片本身就是人物）
    if not new_lines:
        new_lines.append(f"{new_cls} 0.500000 0.500000 1.000000 1.000000\n")

    with open(label_path, 'w') as f:
        f.writelines(new_lines)

    return True


def fix_labels_dir(labels_dir, images_dir):
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    success, failed = 0, 0

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        if fix_label_file(label_path, images_dir):
            success += 1
        else:
            failed += 1

    return success, failed


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))

    for split in ['train', 'val']:
        labels_dir = os.path.join(base, 'labels', split)
        images_dir = os.path.join(base, 'images', split)

        if not os.path.exists(labels_dir):
            print(f"[SKIP] 找不到: {labels_dir}")
            continue

        print(f"\n[FIX] 修正 {split} 標籤...")
        success, failed = fix_labels_dir(labels_dir, images_dir)
        print(f"[DONE] 成功: {success}  失敗: {failed}")

    # 驗證：顯示修正後的範例
    sample_dir = os.path.join(base, 'labels', 'train')
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.txt')][:3]
    print("\n[CHECK] 修正後的標籤範例:")
    for sf in sample_files:
        with open(os.path.join(sample_dir, sf)) as f:
            content = f.read().strip()
        print(f"  {sf}: {content}")

    print("\n修正完成！請重新訓練:")
    print("  python train.py")
