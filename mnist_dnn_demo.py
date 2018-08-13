"""
    A dnn demo using MNIST dataset.
    Create by: FlyingBlade
    Create Time: 2018/6/11 21:50
"""
import tensorflow as tf
import os
import numpy as np

# config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
sess = tf.Session(config=config)

# download MNIST
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

# Dataset params
input_node = 784
output_node = 10

# NN params
hide_layers = [500, ]  # 784 -> 500 -> 10
batch_size = 64
lr = 0.8  # base learning  rate
lr_decay = 0.99  # lr decay rate
reg_rate = 0.0001  # regularization rate
moving_avg_decay = 0.99  # moving average decay


# input_tensor -> (weights1,bias1) relu x-> (weights2,bias2) relu.
def interface(input_tensor, avg_class, weights1, bias1, weights2, bias2):
    if avg_class:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(bias1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(bias2)
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + bias1)
        return tf.matmul(layer1, weights2) + bias2


# model train func
def train(sess, mnist):
    x = tf.placeholder(tf.float32, shape=[None, input_node], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, output_node], name='y-input')

    # hidden layer
    # todo: Attention, initialize influences our result to a large extend. when stddev=1, will train longer and score worse.
    weights1 = tf.Variable(tf.truncated_normal([input_node, hide_layers[0]], stddev=0.1))
    bias1 = tf.Variable(tf.constant(.1, shape=[hide_layers[0], ]))
    weights2 = tf.Variable(tf.truncated_normal([hide_layers[0], output_node], stddev=0.1))
    bias2 = tf.Variable(tf.constant(.1, shape=[output_node], ))

    # forward(Without moving average)
    y = interface(x, None, weights1, bias1, weights2, bias2)

    # step
    global_steps = tf.Variable(0, trainable=False)  # Notice that this var is not trainable.

    # moving avg
    moving_avg = tf.train.ExponentialMovingAverage(moving_avg_decay, global_steps)
    moving_avg_step = moving_avg.apply(tf.trainable_variables())  # using on all trainable variable

    # forward(with moving average)
    average_y = interface(x, moving_avg, weights1, bias1, weights2, bias2)

    # cross_entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # reg_l2
    regularizer = tf.contrib.layers.l2_regularizer(reg_rate)
    reg = regularizer(weights1) + regularizer(weights2)  # usually used on weights, bias not.
    loss = cross_entropy_mean + reg

    # learning rate decay
    learning_rate_decay = tf.train.exponential_decay(lr, global_steps, mnist.train.num_examples / batch_size, lr_decay,
                                                     staircase=True)

    # Optimizer
    # train_step = tf.train.AdagradOptimizer(learning_rate_decay).minimize(loss, global_step=global_steps)
    train_step = tf.train.GradientDescentOptimizer(learning_rate_decay).minimize(loss, global_step=global_steps)
    # Multi steps one time.
    train_op = tf.group(train_step, moving_avg_step)
    # calculate acc
    corrects = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

    # run
    with sess.as_default():
        tf.global_variables_initializer().run()  # initialize
        val_feed = {x: mnist.validation.images,
                    y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}

        for i in range(30000):
            if i % 1000 == 0:
                val_acc = sess.run(accuracy, feed_dict=val_feed)
                print('rounds:', i, 'val acc:', val_acc)

            xs, ys = mnist.train.next_batch(batch_size, shuffle=True)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('test acc:', test_acc)


def main(argv=None):
    train(sess, mnist)


if __name__ == "__main__":
    tf.app.run()