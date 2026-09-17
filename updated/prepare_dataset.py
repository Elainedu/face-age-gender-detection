"""
数据集准备脚本
自动整理图片和标签到正确的目录结构
"""
import os
import shutil
from pathlib import Path
import random

def prepare_dataset(raw_images_dir='data/raw_images',
                   output_base='data',
                   train_ratio=0.8):
    """
    整理数据集结构

    Args:
        raw_images_dir: 原始图片目录
        output_base: 输出基础目录
        train_ratio: 训练集比例
    """
    raw_path = Path(raw_images_dir)

    # 检查原始图片目录是否存在
    if not raw_path.exists():
        print(f"[ERROR] Directory not found: {raw_images_dir}")
        print(f"[INFO] Please place images in {raw_images_dir}/")
        return

    # 获取所有图片文件
    image_files = list(raw_path.glob('*.jpg')) + list(raw_path.glob('*.png'))

    if len(image_files) == 0:
        print(f"[ERROR] No images found in {raw_images_dir}")
        print(f"[INFO] Please make sure image files exist in this directory")
        return

    print(f"[INFO] Found {len(image_files)} images")

    # 随机打乱
    random.shuffle(image_files)

    # 分割训练集和验证集
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"[INFO] Train: {len(train_files)} images")
    print(f"[INFO] Val: {len(val_files)} images")

    # 创建目录结构
    base = Path(output_base)
    dirs = {
        'train_img': base / 'images' / 'train',
        'val_img': base / 'images' / 'val',
        'train_label': base / 'labels' / 'train',
        'val_label': base / 'labels' / 'val'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # 复制文件
    print("\n[START] Copying training set...")
    copy_dataset(train_files, dirs['train_img'], dirs['train_label'])

    print("\n[START] Copying validation set...")
    copy_dataset(val_files, dirs['val_img'], dirs['val_label'])

    print("\n[OK] Dataset prepared!")
    print(f"[DIR] Images:")
    print(f"   Train: {dirs['train_img']}")
    print(f"   Val: {dirs['val_img']}")
    print(f"[DIR] Labels:")
    print(f"   Train: {dirs['train_label']}")
    print(f"   Val: {dirs['val_label']}")

def copy_dataset(image_files, img_output_dir, label_output_dir):
    """复制图片和对应的标签文件"""
    for img_file in image_files:
        # 复制图片
        shutil.copy2(img_file, img_output_dir / img_file.name)

        # 复制对应的标签文件（如果存在）
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            shutil.copy2(label_file, label_output_dir / label_file.name)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='准备数据集')
    parser.add_argument('--raw-images', type=str, default='data/raw_images',
                       help='原始图片目录')
    parser.add_argument('--output', type=str, default='data',
                       help='输出基础目录')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例 (0-1)')

    args = parser.parse_args()

    prepare_dataset(args.raw_images, args.output, args.train_ratio)
