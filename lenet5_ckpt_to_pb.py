# coding:utf-8
import tensorflow as tf
import numpy as np
import lenet5_forward
import lenet5_backward
from tensorflow.python.framework import graph_util

with tf.Graph().as_default() as g:
    x = tf.placeholder(tf.float32, [
        1,
        lenet5_forward.IMAGE_SIZE,
        lenet5_forward.IMAGE_SIZE,
        lenet5_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, lenet5_forward.OUTPUT_NODE])
    y = lenet5_forward.forward(x, False, None)

    ema = tf.train.ExponentialMovingAverage(lenet5_backward.MOVING_AVERAGE_DECAY)
    ema_restore = ema.variables_to_restore()
    saver = tf.train.Saver(ema_restore)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(lenet5_backward.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            graph_def = tf.get_default_graph().as_graph_def()
            output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add_1'])
            # 存pb文件
            with tf.gfile.GFile("./lenet5_model.pb", "wb") as f:
                f.write(output_graph_def.SerializeToString())
        else:
            print("No checkpoint file found")
