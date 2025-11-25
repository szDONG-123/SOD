# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import os
from config import Config as cg
from upsample_skimage import upsample


# ==================== 基础工具层 ====================
def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initial(shape=shape))


def weight_variable_3d(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape=shape))


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv


def conv3d(input, in_features, out_features, kernel_size, with_bias=False):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size, kernel_size]
    W = weight_variable_3d([kernel_size[0], kernel_size[1], kernel_size[2], in_features, out_features])
    conv = tf.nn.conv3d(input, W, [1, 1, 1, 1, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv


def diated_conv2d(input, in_features, out_features, kernel_size, dilated_rate, with_bias=False):
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.atrous_conv2d(input, W, dilated_rate, padding='SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv


def avg_pool(input, s):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], 'VALID')


def batchnorm(x, is_training, center=True, scale=True, epsilon=0.001, decay=0.95):
    shape = x.get_shape().as_list()
    mean, var = tf.nn.moments(x, axes=list(range(len(shape) - 1)))
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([mean, var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean), tf.identity(var)

    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(mean), ema.average(var)))
    shift_v = tf.Variable(tf.zeros(shape[-1])) if center else None
    scale_v = tf.Variable(tf.ones(shape[-1])) if scale else None
    output = tf.nn.batch_normalization(x, mean, var, shift_v, scale_v, epsilon)
    return output


def batchnorm_3d(x, is_training, center=True, scale=True, epsilon=0.001, decay=0.95):
    shape = x.get_shape().as_list()
    mean, var = tf.nn.moments(x, axes=[0, 1, 2, 3])
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([mean, var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean), tf.identity(var)

    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(mean), ema.average(var)))
    shift_v = tf.Variable(tf.zeros(shape[-1])) if center else None
    scale_v = tf.Variable(tf.ones(shape[-1])) if scale else None
    output = tf.nn.batch_normalization(x, mean, var, shift_v, scale_v, epsilon)
    return output


# ==================== 模块定义 ====================
def batch_activ_conv(current, in_features, out_features, kernel_size, dilated_rate, is_training, keep_prob):
    current = batchnorm(current, is_training=is_training)
    current = tf.nn.relu(current)
    current = diated_conv2d(current, in_features, out_features, kernel_size, dilated_rate)
    current = tf.nn.dropout(current, keep_prob)
    return current


def block(input, layers, in_features, growth, dilated_rate, is_training, keep_prob):
    current = input
    features = in_features
    for idx in range(layers):
        tmp = batch_activ_conv(current, features, growth, 3, dilated_rate, is_training, keep_prob)
        current = tf.concat([current, tmp], 3)
        features += growth
    return current, features


def block_3d(input_5d, is_training, keep_prob):
    print("\n===== 3D-block =====")
    print("输入:", input_5d)

    current = input_5d

    # Block 1: 80 -> 64 channels
    with tf.variable_scope("block1"):
        current = conv3d(current, 80, 64, [3, 3, 3])
        current = batchnorm_3d(current, is_training=is_training)
        current = tf.nn.relu(current)
        current = tf.nn.dropout(current, keep_prob)
        print("Block1:", current)

    # Block 2: 64 -> 48 channels
    with tf.variable_scope("block2"):
        current = conv3d(current, 64, 48, [3, 3, 3])
        current = batchnorm_3d(current, is_training=is_training)
        current = tf.nn.relu(current)
        current = tf.nn.dropout(current, keep_prob)
        print("Block2:", current)

    # Block 3: 48 -> 32 channels
    with tf.variable_scope("block3"):
        current = conv3d(current, 48, 32, [3, 3, 3])
        current = batchnorm_3d(current, is_training=is_training)
        current = tf.nn.relu(current)
        current = tf.nn.dropout(current, keep_prob)
        print("Block3:", current)

    # Block 4: 32 -> 16 channels
    with tf.variable_scope("block4"):
        current = conv3d(current, 32, 16, [3, 3, 3])
        current = batchnorm_3d(current, is_training=is_training)
        current = tf.nn.relu(current)
        current = tf.nn.dropout(current, keep_prob)
        print("Block4:", current)

    print("最终输出:", current)
    return current


# ==================== 核心图构建函数 ====================
def build_graph(xs, is_training, keep_prob):
    """
    构建计算图的核心函数，包含所有 print 语句以保持原样输出
    """
    layers = 12
    fuse_channels = 16
    theta = 0.5

    B = tf.shape(xs)[0]
    T = tf.shape(xs)[1]

    print("===== 2D DenseBlock Encoder =====")

    # Reshape: [B, T, H*W*C] -> [B*T, H, W, C]
    input_2d = tf.reshape(xs, [B * T, cg.image_size, cg.image_size, cg.image_channel])
    # 原始代码有 tf.summary.image，这里保留逻辑但不一定要print，因为log里没显示image的tensor

    # 初始卷积
    current = conv2d(input_2d, 3, 16, 3)
    print("Initial:", current)

    # Block1: 256x256, d=1
    current, features = block(current, layers, 16, 12, 1, is_training, keep_prob)
    scale_256 = conv2d(current, features, fuse_channels, 3, True)
    print("scale_256:", scale_256)

    current = batch_activ_conv(current, features, int(features * theta), 1, 1, is_training, keep_prob)
    current = avg_pool(current, 2)

    # Block2: 128x128, d=1
    current, features = block(current, layers, int(features * theta), 12, 1, is_training, keep_prob)
    scale_128 = conv2d(current, features, fuse_channels, 3, True)
    print("scale_128:", scale_128)

    current = batch_activ_conv(current, features, int(features * theta), 1, 1, is_training, keep_prob)
    current = avg_pool(current, 2)

    # Block3: 64x64, d=2
    current, features = block(current, layers, int(features * theta), 12, 2, is_training, keep_prob)
    scale_64_1 = conv2d(current, features, fuse_channels, 3, True)
    print("scale_64_1:", scale_64_1)

    current = batch_activ_conv(current, features, int(features * theta), 1, 1, is_training, keep_prob)

    # Block4: 64x64, d=4
    current, features = block(current, layers, int(features * theta), 12, 4, is_training, keep_prob)
    scale_64_2 = conv2d(current, features, fuse_channels, 3, True)
    print("scale_64_2:", scale_64_2)

    current = batch_activ_conv(current, features, int(features * theta), 1, 1, is_training, keep_prob)

    # Block5: 64x64, d=8
    current, features = block(current, layers, int(features * theta), 12, 8, is_training, keep_prob)
    scale_64_3 = conv2d(current, features, fuse_channels, 3, True)
    print("scale_64_3:", scale_64_3)

    # ========== 所有特征上采样到256×256 ==========
    print("\n===== Feature Upsampling to 256x256 =====")

    scale_256_up = scale_256
    scale_128_up = upsample(scale_128, 2, fuse_channels)
    scale_64_1_up = upsample(scale_64_1, 4, fuse_channels)
    scale_64_2_up = upsample(scale_64_2, 4, fuse_channels)
    scale_64_3_up = upsample(scale_64_3, 4, fuse_channels)

    print("scale_128_up:", scale_128_up)
    print("scale_64_1_up:", scale_64_1_up)
    print("scale_64_2_up:", scale_64_2_up)
    print("scale_64_3_up:", scale_64_3_up)

    # ========== 通道拼接 ==========
    fused_features = tf.concat([scale_256_up, scale_128_up, scale_64_1_up, scale_64_2_up, scale_64_3_up], axis=3)
    print("fused_features (2D):", fused_features)

    # ========== Reshape为5D: [B*T, H, W, C] -> [B, T, H, W, C] ==========
    fused_features_5d = tf.reshape(fused_features, [B, T, 256, 256, fuse_channels * 5])
    print("fused_features_5d:", fused_features_5d)

    # ========== 3D-block ==========
    logits_5d = block_3d(fused_features_5d, is_training, keep_prob)
    logits_5d = conv3d(logits_5d, 16, 1, [1, 1, 1], with_bias=True)

    # ========== 输出 ==========
    logits_2d = tf.reshape(logits_5d, [B * T, 256, 256, 1])
    yp_2d = tf.nn.sigmoid(logits_2d, name="yp_2d")
    print("yp_2d:", yp_2d)

    return yp_2d, logits_2d


# ==================== 辅助函数 (Loss, Metrics, Preprocess) ====================
def images_preprocessing(images, masks):
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    batchs = images.shape[0]
    T = images.shape[1]

    images = np.reshape(images, [batchs, T, cg.image_size, cg.image_size, cg.image_channel])

    # 归一化逻辑
    for i in range(batchs):
        for t in range(T):
            images[i, t, :, :, 2] -= np.mean(images[i, t, :, :, 2])
            images[i, t, :, :, 1] -= np.mean(images[i, t, :, :, 1])
            images[i, t, :, :, 0] -= np.mean(images[i, t, :, :, 0])

            images[i, t, :, :, 2] /= (np.std(images[i, t, :, :, 2]) + 1e-12)
            images[i, t, :, :, 1] /= (np.std(images[i, t, :, :, 1]) + 1e-12)
            images[i, t, :, :, 0] /= (np.std(images[i, t, :, :, 0]) + 1e-12)

    images = np.reshape(images, [batchs, T, cg.image_size * cg.image_size * cg.image_channel])
    masks = masks / 255.0
    return images, masks


def fused_loss(yp, gt, batch_size_actual):
    # MAE loss
    mae_loss = tf.reduce_mean(tf.log(1 + tf.exp(tf.abs(yp - gt))))

    # Generalized Dice loss
    mask_front = gt
    mask_background = 1 - gt
    pro_front = yp
    pro_background = 1 - yp

    w1 = 1 / (tf.pow(tf.reduce_sum(mask_front), 2) + 1e-12)
    w2 = 1 / (tf.pow(tf.reduce_sum(mask_background), 2) + 1e-12)
    numerator = w1 * tf.reduce_sum(mask_front * pro_front) + w2 * tf.reduce_sum(mask_background * pro_background)
    denominator = w1 * tf.reduce_sum(mask_front + pro_front) + w2 * tf.reduce_sum(mask_background + pro_background)
    dice_loss = 1 - 2 * numerator / (denominator + 1e-12)

    # Cross entropy loss
    total_pixels = tf.cast(cg.image_size * cg.image_size * batch_size_actual, tf.float32)
    w = (total_pixels - tf.reduce_sum(yp)) / (tf.reduce_sum(yp) + 1e-12)
    cross_entropy_loss = -tf.reduce_mean(
        0.1 * w * mask_front * tf.log(pro_front + 1e-12) + mask_background * tf.log(pro_background + 1e-12))

    total_loss = dice_loss + mae_loss + cross_entropy_loss
    return total_loss, mae_loss, dice_loss, cross_entropy_loss


def F_measure(gt, pred_map):
    mask = tf.greater(pred_map, 0.5)
    mask = tf.cast(mask, tf.float32)
    gtCnt = tf.reduce_sum(gt)
    hitMap = tf.where(gt > 0, mask, tf.zeros(tf.shape(mask)))
    hitCnt = tf.reduce_sum(hitMap)
    algCnt = tf.reduce_sum(mask)
    prec = hitCnt / (algCnt + 1e-12)
    recall = hitCnt / (gtCnt + 1e-12)
    beta_square = 0.3
    F_score = (1 + beta_square) * prec * recall / (beta_square * prec + recall + 1e-12)
    return prec, recall, F_score


def visualize_results(session, yp_op, xs_pl, ys_pl, is_training_pl, keep_prob_pl,
                      test_data, test_labels, epoch, save_dir, num_samples=3):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epoch_dir = os.path.join(save_dir, "epoch_%03d" % epoch)
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)

    num_samples = min(num_samples, len(test_data))
    sample_data = test_data[:num_samples]
    sample_labels = test_labels[:num_samples]

    sample_data_proc, sample_labels_proc = images_preprocessing(sample_data.copy(), sample_labels.copy())

    predictions = session.run(yp_op, feed_dict={
        xs_pl: sample_data_proc,
        ys_pl: sample_labels_proc,
        is_training_pl: False,
        keep_prob_pl: 1.0
    })

    for video_idx in range(num_samples):
        T = sample_data.shape[1]
        orig_imgs = sample_data[video_idx]
        orig_imgs = np.reshape(orig_imgs, [T, cg.image_size, cg.image_size, cg.image_channel])

        gt_masks = sample_labels[video_idx]
        gt_masks = np.reshape(gt_masks, [T, cg.image_size, cg.image_size])

        start_idx = sum([sample_data.shape[1] for _ in range(video_idx)])
        pred_masks = predictions[start_idx: start_idx + T]
        pred_masks = np.squeeze(pred_masks, axis=-1)

        for t in range(T):
            img = orig_imgs[t].astype(np.uint8)
            gt = (gt_masks[t]).astype(np.uint8)
            pred = (pred_masks[t] * 255).astype(np.uint8)

            gt_3ch = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
            pred_3ch = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

            comparison = np.hstack([img, gt_3ch, pred_3ch])

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, 'Input', (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, 'GT', (cg.image_size + 10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, 'Pred', (cg.image_size * 2 + 10, 30), font, 1, (0, 255, 0), 2)

            save_path = os.path.join(epoch_dir, "video%02d_frame%03d.png" % (video_idx, t))
            cv2.imwrite(save_path, comparison)

    print("可视化结果已保存到: %s" % epoch_dir)