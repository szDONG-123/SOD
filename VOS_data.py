# -*- coding=utf-8 -*-
import os
import cv2
import numpy as np
from config import Config as cg

"""
数据加载 - 视频序列版本（嵌套目录结构）
数据组织方式:
    ./VOS/train/image/
        ├── 0a49f5265b/          # 视频1
        │   ├── 00000.jpg
        │   ├── 00005.jpg
        │   ├── 00010.jpg
        │   └── ...
        ├── video_id2/           # 视频2
        │   └── ...
    ./VOS/train/label/
        ├── 0a49f5265b/          # 视频1的mask
        │   ├── 00000.png
        │   ├── 00005.png
        │   └── ...
        └── video_id2/
            └── ...

    ./VOS/test/image/ 和 ./VOS/test/label/ 结构相同

每个子文件夹代表一个完整的视频序列
"""


def image_resize(image, image_channel=3):
    """
    图像预处理：resize到统一尺寸并padding
    """
    ht, wd, ch = image.shape
    new_ht = 0
    new_wd = 0
    imagec = 0

    if ht > wd:
        new_ht = cg.image_size
        new_wd = int(wd * new_ht / ht)
        pad_zeros = np.zeros((cg.image_size, cg.image_size - new_wd, image_channel), dtype=image.dtype)
        imagec = cv2.resize(image, (new_wd, new_ht))
        imagec = np.hstack((imagec, pad_zeros))
    else:
        new_wd = cg.image_size
        new_ht = int(ht * new_wd / wd)
        pad_zeros = np.zeros((cg.image_size - new_ht, cg.image_size, image_channel), dtype=image.dtype)
        imagec = cv2.resize(image, (new_wd, new_ht))
        imagec = np.vstack((imagec, pad_zeros))

    return imagec


def load_data_one(path, images_path, masks_path, is_augmented=False):
    """
    加载视频序列数据（每个子文件夹是一个视频）
    返回:
        images: [N, T, image_size*image_size*image_channel]  # N个视频，每个T帧
        masks: [N, T, image_size*image_size]
        其中T可能因视频而异，但这里假设每个视频帧数相同或做padding
    """
    img_base_dir = os.path.join(path, images_path)
    mask_base_dir = os.path.join(path, masks_path)

    # 获取所有视频ID子目录
    video_dirs = sorted([d for d in os.listdir(img_base_dir)
                         if os.path.isdir(os.path.join(img_base_dir, d))])

    print("发现 %d 个视频目录" % len(video_dirs))

    images_all = []
    masks_all = []
    video_count = 0

    # 遍历每个视频目录
    for video_id in video_dirs:
        img_dir = os.path.join(img_base_dir, video_id)
        mask_dir = os.path.join(mask_base_dir, video_id)

        # 检查mask目录是否存在
        if not os.path.exists(mask_dir):
            print("警告: 找不到对应的mask目录: %s" % mask_dir)
            continue

        # 获取该视频下所有图像文件并排序
        all_files = sorted(os.listdir(img_dir))
        # 过滤只保留图像文件
        all_files = [f for f in all_files if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.jpeg'))]

        if len(all_files) == 0:
            print("警告: 视频 %s 中没有图像文件" % video_id)
            continue

        print("  视频 %s: %d 帧" % (video_id, len(all_files)))

        imgs = []
        msks = []
        valid_video = True

        # 读取该视频的所有帧
        for img_file in all_files:
            img_path = os.path.join(img_dir, img_file)

            # mask文件名：尝试多种后缀
            base_name = os.path.splitext(img_file)[0]
            mask_path = None
            for ext in ['.png', '.jpg', '.PNG', '.JPG']:
                candidate = os.path.join(mask_dir, base_name + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                print("  警告: 找不到mask: %s/%s" % (video_id, base_name))
                valid_video = False
                break

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            if image is None:
                print("  警告: 读取图像失败: %s" % img_path)
                valid_video = False
                break
            if mask is None:
                print("  警告: 读取mask失败: %s" % mask_path)
                valid_video = False
                break

            # resize
            imagec = image_resize(image, cg.image_channel)
            maskc = image_resize(mask, cg.image_channel)
            maskc = maskc[:, :, 0]  # 取单通道
            th, maskc = cv2.threshold(maskc, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            imgs.append(imagec)
            msks.append(maskc)

        # 确保视频完整读取
        if not valid_video or len(imgs) == 0:
            print("  跳过视频 %s" % video_id)
            continue

        # 堆叠成 [T, H, W, C] 和 [T, H, W]
        imgs_stack = np.stack(imgs, axis=0)  # [T, 256, 256, 3]
        msks_stack = np.stack(msks, axis=0)  # [T, 256, 256]

        images_all.append(imgs_stack)
        masks_all.append(msks_stack)

        # 数据增强
        if is_augmented:
            # 水平翻转
            imgs_hf = np.stack([cv2.flip(img, 0) for img in imgs], axis=0)
            msks_hf = np.stack([cv2.flip(msk, 0) for msk in msks], axis=0)
            images_all.append(imgs_hf)
            masks_all.append(msks_hf)

            # 垂直翻转
            imgs_vf = np.stack([cv2.flip(img, 1) for img in imgs], axis=0)
            msks_vf = np.stack([cv2.flip(msk, 1) for msk in msks], axis=0)
            images_all.append(imgs_vf)
            masks_all.append(msks_vf)

            # 双向翻转
            imgs_bf = np.stack([cv2.flip(img, -1) for img in imgs], axis=0)
            msks_bf = np.stack([cv2.flip(msk, -1) for msk in msks], axis=0)
            images_all.append(imgs_bf)
            masks_all.append(msks_bf)

        video_count += 1
        if video_count % 10 == 0:
            print("  已加载 %d 个视频" % video_count)

    print("----------------------")
    print("原始视频数: %d" % video_count)
    if is_augmented:
        print("增强后视频数: %d" % (video_count * 4))

    # 注意：这里假设所有视频帧数相同，如果不同需要padding或其他处理
    # 如果帧数不同，下面的np.array会失败，需要特殊处理
    try:
        images = np.array(images_all, dtype=np.uint8)  # [N, T, 256, 256, 3]
        masks = np.array(masks_all, dtype=np.uint8)  # [N, T, 256, 256]
        print("images shape=", images.shape)
        print("masks shape=", masks.shape)
    except ValueError as e:
        print("错误: 视频帧数不一致，无法直接转换为numpy数组")
        print("需要对视频进行padding或截断处理")
        # 这里可以添加padding逻辑
        # 找出最大帧数
        max_frames = max([img.shape[0] for img in images_all])
        print("最大帧数: %d" % max_frames)

        # Padding到最大帧数
        images_padded = []
        masks_padded = []
        for imgs, msks in zip(images_all, masks_all):
            num_frames = imgs.shape[0]
            if num_frames < max_frames:
                # 使用最后一帧进行padding
                pad_frames = max_frames - num_frames
                imgs_pad = np.repeat(imgs[-1:], pad_frames, axis=0)
                msks_pad = np.repeat(msks[-1:], pad_frames, axis=0)
                imgs = np.concatenate([imgs, imgs_pad], axis=0)
                msks = np.concatenate([msks, msks_pad], axis=0)
            images_padded.append(imgs)
            masks_padded.append(msks)

        images = np.array(images_padded, dtype=np.uint8)
        masks = np.array(masks_padded, dtype=np.uint8)
        print("padding后 images shape=", images.shape)
        print("padding后 masks shape=", masks.shape)

    # reshape为 [N, T, H*W*C] 和 [N, T, H*W]
    n_videos = images.shape[0]
    n_frames = images.shape[1]
    images = np.reshape(images, [n_videos, n_frames,
                                 cg.image_size * cg.image_size * cg.image_channel])
    masks = np.reshape(masks, [n_videos, n_frames,
                               cg.image_size * cg.image_size])

    print("reshape后:")
    print("images shape=", images.shape)
    print("masks shape=", masks.shape)

    return images, masks


def load_data():
    """
    加载训练和测试数据（视频序列，嵌套目录结构）
    数据路径: ./VOS/train/ 和 ./VOS/test/
    """
    # 训练数据路径
    train_path = "./VOS/train"
    train_images_path = "image"
    train_masks_path = "label"

    # 测试数据路径
    test_path = "./VOS/test"
    test_images_path = "image"
    test_masks_path = "label"

    print("======================")
    print("加载训练数据: %s" % train_path)
    print("======================")
    train_images, train_masks = load_data_one(
        path=train_path,
        images_path=train_images_path,
        masks_path=train_masks_path,
        is_augmented=True
    )

    # 保存为视频序列版本
    np.save("train_images_video", train_images)
    np.save("train_masks_video", train_masks)
    print("训练数据已保存: train_images_video.npy, train_masks_video.npy")

    print("\n======================")
    print("加载测试数据: %s" % test_path)
    print("======================")
    test_images, test_masks = load_data_one(
        path=test_path,
        images_path=test_images_path,
        masks_path=test_masks_path,
        is_augmented=False
    )

    # 保存为视频序列版本
    np.save("test_images_video", test_images)
    np.save("test_masks_video", test_masks)
    print("测试数据已保存: test_images_video.npy, test_masks_video.npy")

    return train_images, train_masks, test_images, test_masks


if __name__ == '__main__':
    load_data()