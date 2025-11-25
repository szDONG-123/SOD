# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import VOS_model as model
from config import Config as cg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_dir = "./log_video"
# vis_dir = "./visualization_video" # 训练时不再需要可视化测试集
model_save_dir = "./model_video"


def train():
    # 1. 加载数据 (只加载训练数据)
    print("Loading training data...")
    train_images = np.load("train_images_video.npy", mmap_mode='r')
    train_masks = np.load("train_masks_video.npy", mmap_mode='r')

    # 打印形状
    print('Train videos shape:', train_images.shape)
    print('Train masks shape:', train_masks.shape)

    # 2. 定义 Placeholder
    graph = tf.Graph()
    with graph.as_default():
        xs = tf.placeholder("float", shape=[None, None, cg.image_size * cg.image_size * cg.image_channel], name="xs")
        ys = tf.placeholder("float", shape=[None, None, cg.image_size * cg.image_size], name="ys")
        lr = tf.placeholder("float", shape=[], name='lr')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        is_training = tf.placeholder("bool", shape=[], name='is_training')

        # 3. 构建模型 (这里依然会打印 VOS_model 中的网络结构信息)
        yp_2d, logits_2d = model.build_graph(xs, is_training, keep_prob)

        # 准备 GT 用于 Loss 计算
        B = tf.shape(xs)[0]
        T = tf.shape(xs)[1]
        ysc_2d = tf.reshape(ys, [B * T, cg.image_size, cg.image_size, 1])

        # 4. 计算损失和指标
        total_loss, mae_loss, dice_loss, ce_loss = model.fused_loss(yp_2d, ysc_2d, B * T)

        # Summary
        tf.summary.scalar("total_loss", total_loss)
        tf.summary.scalar("mae_loss", mae_loss)
        tf.summary.scalar("dice_loss", dice_loss)
        tf.summary.scalar("cross_entropy_loss", ce_loss)

        MAE = tf.reduce_mean(tf.abs(yp_2d - ysc_2d), name="mae")
        tf.summary.scalar("MAE", MAE)

        prec, recall, F_score = model.F_measure(ysc_2d, yp_2d)
        tf.summary.scalar("precision", prec)
        tf.summary.scalar("recall", recall)
        tf.summary.scalar("F_score", F_score)

        # 5. 优化器
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        weight_decay = 0.0001
        train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(total_loss + weight_decay * l2)

        saver = tf.train.Saver(max_to_keep=50)
        merged_summary_op = tf.summary.merge_all()

    # 6. 会话 Session
    with tf.Session(graph=graph) as session:
        batch_size_train = 2
        learning_rate = 0.1 / 50

        session.run(tf.global_variables_initializer())

        if tf.gfile.Exists(log_dir):
            tf.gfile.DeleteRecursively(log_dir)
        tf.gfile.MakeDirs(log_dir)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # 只保留训练日志 Writer
        train_summary_writer = tf.summary.FileWriter(log_dir + "/train", graph)

        for epoch in range(1, 31):
            if epoch == 5: learning_rate = 0.01 / 50
            if epoch == 10: learning_rate = 0.001 / 50

            # Shuffle
            pi = np.random.permutation(len(train_images))
            t_data, t_label = train_images[pi], train_masks[pi]
            batch_count = len(t_data) // batch_size_train

            print("Epoch %d, Batch count: %d, LR: %.6f" % (epoch, batch_count, learning_rate))

            for batch_idx in range(batch_count):
                b_data = t_data[batch_idx * batch_size_train: (batch_idx + 1) * batch_size_train]
                b_label = t_label[batch_idx * batch_size_train: (batch_idx + 1) * batch_size_train]

                b_data_proc, b_label_proc = model.images_preprocessing(b_data.copy(), b_label.copy())

                batch_res = session.run(
                    [merged_summary_op, train_step, total_loss, MAE, prec, recall, F_score,
                     mae_loss, dice_loss, ce_loss],
                    feed_dict={
                        xs: b_data_proc, ys: b_label_proc,
                        lr: learning_rate, is_training: True, keep_prob: 0.8
                    }
                )

                if batch_idx % 50 == 0:
                    train_summary_writer.add_summary(batch_res[0], epoch * batch_count + batch_idx)
                    print(
                                "Epoch %d, Batch %d: loss=%.4f (mae=%.4f, dice=%.4f, ce=%.4f), MAE=%.4f, P=%.4f, R=%.4f, F=%.4f" %
                                (epoch, batch_idx, batch_res[2], batch_res[7], batch_res[8], batch_res[9],
                                 batch_res[3], batch_res[4], batch_res[5], batch_res[6]))

            # 移除了测试验证和可视化代码，直接保存模型
            save_path = saver.save(session, os.path.join(model_save_dir, 'dense_3dunet_video_%d.ckpt' % epoch))
            print("Model saved: %s" % save_path)


if __name__ == '__main__':
    train()