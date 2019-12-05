# coding:utf-8
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import lenet5_forward
import os
import time
import matplotlib.pyplot as plt

STEPS = 5000
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

def backward(mnist):
    x = tf.placeholder(tf.float32, [
    BATCH_SIZE,
    lenet5_forward.IMAGE_SIZE,
    lenet5_forward.IMAGE_SIZE,
    lenet5_forward.NUM_CHANNELS])

    y_ = tf.placeholder(tf.float32, [None, lenet5_forward.OUTPUT_NODE])
    y = lenet5_forward.forward(x, True, REGULARIZER) #前向传播网络
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(    #指数衰减学习率
        LEARNING_RATE_BASE,
        global_step,    #用于记录当前训练轮数，需设置为不可训练
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)    #阶梯下降/平滑下降

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        init_op = tf.global_variables_initializer()    #初始化
        sess.run(init_op)

        graph_def = tf.get_default_graph() .as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add_1'])

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        #if ckpt and ckpt.model_checkpoint_path:
        # saver.restore(sess, ckpt.model_checkpoint_path)

        fig_loss = np.zeros([STEPS])
        fig_accuracy = np.zeros([STEPS])

        start_time = time.time()

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
            BATCH_SIZE,
            lenet5_forward.IMAGE_SIZE,
            lenet5_forward.IMAGE_SIZE,
            lenet5_forward.NUM_CHANNELS))
            _, fig_loss[i], fig_accuracy[i], step = sess.run([train_op, loss, accuracy, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print("After %d step(s), loss = %g, accuracy = %g" % (step, fig_loss[i], fig_accuracy[i]))
        # 显示训练速度
        end_time = time.time()
        rate = (STEPS * BATCH_SIZE) / (end_time - start_time)
        print("Average Training Rate: %.1f examples/sec" % rate)

        # 存ckpt
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        # 存pb文件
        with tf.gfile.GFile("./combined_model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
        # 绘制曲线
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns1 = ax1.plot(np.arange(STEPS), fig_loss, label="Loss")
        # 按一定间隔显示实现方法
        # ax2.plot(200 * np.arange(len(fig_accuracy)), fig_accuracy, 'r')
        lns2 = ax2.plot(np.arange(STEPS), fig_accuracy, 'r', label="Accuracy")
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('training loss')
        ax2.set_ylabel('training accuracy')
        # 合并图例
        lns = lns1 + lns2
        labels = ["Loss", "Accuracy"]
        # labels = [l.get_label() for l in lns]
        plt.legend(lns, labels, loc=7)
        plt.show()


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)

if __name__=='__main__':
    main()