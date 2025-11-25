# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import VOS_model as model
from config import Config as cg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
vis_dir = "./visualization_test_result"
model_path = "./XXXXX.ckpt"  # 请确保这里指向你想要测试的模型编号


def test():
    # 1. 加载测试数据
    test_images = np.load("test_images_video.npy", mmap_mode='r')
    test_masks = np.load("test_masks_video.npy", mmap_mode='r')

    print("Test videos shape:", test_images.shape)
    print("Test masks shape:", test_masks.shape)

    # 2. 构建图
    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, None, cg.image_size * cg.image_size * cg.image_channel], name="xs")
        ys = tf.placeholder("float", shape=[None, None, cg.image_size * cg.image_size], name="ys")
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        is_training = tf.placeholder("bool", shape=[], name='is_training')

        # 构建网络
        yp_2d, _ = model.build_graph(xs, is_training, keep_prob)

        B = tf.shape(xs)[0]
        T = tf.shape(xs)[1]
        ysc_2d = tf.reshape(ys, [B * T, cg.image_size, cg.image_size, 1])

        # ========== 新增：计算 Loss 和 其他指标 (与 Train 保持一致) ==========
        # 计算 Loss (Dice, MAE, CE)
        total_loss, mae_loss, dice_loss, ce_loss = model.fused_loss(yp_2d, ysc_2d, B * T)

        # 计算 MAE 指标
        MAE = tf.reduce_mean(tf.abs(yp_2d - ysc_2d))

        # 计算 P, R, F-score
        prec, recall, F_score = model.F_measure(ysc_2d, yp_2d)

        saver = tf.train.Saver()

    # 3. 执行 Session
    with tf.Session(graph=graph) as sess:
        print("Restoring model from %s ..." % model_path)
        saver.restore(sess, model_path)

        num_samples = len(test_images)
        print("Start testing on %d samples..." % num_samples)

        batch_size_test = 1
        total_f_score = 0
        total_mae = 0

        for i in range(0, num_samples, batch_size_test):
            b_data = test_images[i:i + batch_size_test]
            b_label = test_masks[i:i + batch_size_test]

            b_data_proc, b_label_proc = model.images_preprocessing(b_data.copy(), b_label.copy())

            # ========== 修改：运行所有指标 ==========
            res = sess.run(
                [total_loss, mae_loss, dice_loss, ce_loss, MAE, prec, recall, F_score, yp_2d],
                feed_dict={
                    xs: b_data_proc,
                    ys: b_label_proc,
                    is_training: False,
                    keep_prob: 1.0
                }
            )

            # 解包结果
            t_loss, m_loss, d_loss, c_loss, mae_val, p_val, r_val, f_val, _ = res

            total_f_score += f_val
            total_mae += mae_val

            # ========== 修改：打印格式与 Train 一致 ==========
            print("Test Sample %d: loss=%.4f (mae=%.4f, dice=%.4f, ce=%.4f), " %
                  (i, t_loss, m_loss, d_loss, c_loss))

        avg_f = total_f_score / (num_samples / batch_size_test)
        avg_mae = total_mae / (num_samples / batch_size_test)

        print("=" * 60)
        print("Average Test Results: MAE=%.4f, F-score=%.4f" % (avg_mae, avg_f))
        print("=" * 60)

        # 可视化 (最后保存一部分结果)
        model.visualize_results(sess, yp_2d, xs, ys, is_training, keep_prob,
                                test_images, test_masks, 999, vis_dir, num_samples=3)


if __name__ == '__main__':
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    test()