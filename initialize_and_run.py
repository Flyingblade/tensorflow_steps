"""
    Create by: FlyingBlade
    Create Time: 2018/6/6 16:25
"""
import tensorflow as tf
import numpy as np
import os

# config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
sess = tf.Session(config=config)

# h-params
batch_size = 8
lr = 0.001
# network define
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# x = tf.constant([[0.2,0.5]])

# todo Attention: dtype & shape
x = tf.placeholder(tf.float32, shape=(None, 2), name='xinput')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='yinput')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)

# loss func
#
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
# Opt
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

# load data
rdm = np.random.RandomState(1)
dataset_size = 128
df_x = rdm.rand(dataset_size, 2)
df_y = np.reshape([float(x[0] + x[1] < 1) for x in df_x], (-1, 1))
# initialize
global_initial = tf.global_variables_initializer()
sess.run(global_initial)

# start train
rounds = 10000
for i in range(rounds):
    # todo Attention: here needs "%dataset_size", if not, will only train 1 epoch.
    start = (i * batch_size) % dataset_size
    end = min(dataset_size, start + batch_size)

    sess.run(train_step, feed_dict={x: df_x[start:end], y_: df_y[start:end]})
    if i % 1000 == 0:
        loss = sess.run(cross_entropy, feed_dict={x: df_x, y_: df_y})
        print(i, loss)
