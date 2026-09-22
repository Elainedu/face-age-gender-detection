"""
平衡資料集 - 對少數類別進行 oversampling
讓每個類別都達到 target_count 數量
"""
import os
import shutil

CLASSES = [
    'male0-20', 'male21-40', 'male41-60', 'male61-80', 'male81~',
    'female0-20', 'female21-40', 'female41-60', 'female61-80', 'female81~'
]
TARGET = 150  # 每個類別目標數量

base = os.path.dirname(os.path.abspath(__file__))


def get_class_from_label(label_path):
    """從 label 檔案讀取類別 ID"""
    with open(label_path, 'r') as f:
        line = f.readline().strip()
    if line:
        return int(line.split()[0])
    return None


def balance_split(split):
    src_img = os.path.join(base, 'images', split)
    src_lbl = os.path.join(base, 'labels', split)
    dst_img = os.path.join(base, 'images', f'{split}_balanced')
    dst_lbl = os.path.join(base, 'labels', f'{split}_balanced')

    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    # 先按類別分組所有檔案
    class_files = {i: [] for i in range(10)}

    for lbl_file in os.listdir(src_lbl):
        if not lbl_file.endswith('.txt'):
            continue
        lbl_path = os.path.join(src_lbl, lbl_file)
        cls = get_class_from_label(lbl_path)
        if cls is None or cls >= 10:
            continue

        # 找對應圖片
        name = lbl_file.replace('.txt', '')
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.jpg.chip.jpg']:
            candidate = os.path.join(src_img, name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path:
            class_files[cls].append((img_path, lbl_path))

    # 統計並 oversampling
    print(f"\n[{split}] 原始分布:")
    for cls, files in class_files.items():
        print(f"  Class {cls} ({CLASSES[cls]}): {len(files)}")

    total_copied = 0
    for cls, files in class_files.items():
        if not files:
            print(f"  [WARN] Class {cls} 沒有資料，跳過")
            continue

        count = 0
        idx = 0
        while count < TARGET:
            src_i, src_l = files[idx % len(files)]
            ext = os.path.splitext(src_i)[1]
            new_name = f"cls{cls}_{count:04d}{ext}"
            new_lbl  = f"cls{cls}_{count:04d}.txt"
            shutil.copy2(src_i, os.path.join(dst_img, new_name))
            shutil.copy2(src_l, os.path.join(dst_lbl, new_lbl))
            count += 1
            idx += 1

        total_copied += count

    print(f"[{split}] 平衡後每類 {TARGET} 張，共 {total_copied} 張")
    return dst_img, dst_lbl


if __name__ == '__main__':
    balance_split('train')
    balance_split('val')
    print("\n完成！請用以下指令重新訓練:")
    print("  python train_balanced.py")
